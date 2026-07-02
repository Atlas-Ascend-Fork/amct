# coding=utf-8
# ----------------------------------------------------------------------------
# Adapted from
# https://github.com/ruikangliu/FlatQuant
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# Copyright (c) 2024 ruikangliu. Licensed under MIT.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------

import torch
import torch.nn as nn

from amct_pytorch.algorithms.registry_factory import ALGO_REGISTRY


def _random_orthogonal(size: int, *, device=None, dtype=None) -> torch.Tensor:
    matrix = torch.randn(size, size, device=device, dtype=torch.float32)
    q, r = torch.linalg.qr(matrix)
    diag = torch.sign(torch.diag(r))
    diag = torch.where(diag == 0, torch.ones_like(diag), diag)
    q = q @ torch.diag(diag)
    if dtype is not None:
        q = q.to(dtype)
    return q


def _kronecker_matmul(x: torch.Tensor, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    init_shape = x.shape
    x = x.reshape(-1, left.shape[0], right.shape[0])
    x = torch.matmul(x, right.to(x))
    x = torch.matmul(left.to(x).transpose(0, 1), x)
    return x.reshape(init_shape)


class _InvFlatSingleTransform(nn.Module):
    def __init__(self, size: int):
        super().__init__()
        self.linear = nn.Linear(size, size, bias=False)
        self.linear.weight.data.copy_(_random_orthogonal(size).to(self.linear.weight))

    def forward(self, x: torch.Tensor, inv_t: bool = False) -> torch.Tensor:
        init_shape = x.shape
        matrix = self.linear.weight
        if inv_t:
            matrix = torch.linalg.inv(matrix).transpose(0, 1)
        matrix = matrix.to(device=x.device, dtype=x.dtype)
        x = x.reshape(-1, matrix.shape[0])
        return x.matmul(matrix).reshape(init_shape)


class _InvFlatDecomposeTransform(nn.Module):
    def __init__(self, left_size: int, right_size: int, add_diag: bool):
        super().__init__()
        self.add_diag = add_diag

        self.linear_left = nn.Linear(left_size, left_size, bias=False)
        self.linear_left.weight.data.copy_(_random_orthogonal(left_size).to(self.linear_left.weight))

        self.linear_right = nn.Linear(right_size, right_size, bias=False)
        self.linear_right.weight.data.copy_(_random_orthogonal(right_size).to(self.linear_right.weight))

        if add_diag:
            self.diag_scale = nn.Parameter(torch.ones(left_size * right_size, dtype=torch.float32))

    def forward(self, x: torch.Tensor, inv_t: bool = False) -> torch.Tensor:
        if self.add_diag:
            diag_scale = self.diag_scale.to(device=x.device, dtype=x.dtype)
            if inv_t:
                x = x / diag_scale
            else:
                x = x * diag_scale

        matrix_left = self.linear_left.weight
        matrix_right = self.linear_right.weight
        if inv_t:
            matrix_left = torch.linalg.inv(matrix_left).transpose(0, 1)
            matrix_right = torch.linalg.inv(matrix_right).transpose(0, 1)
        return _kronecker_matmul(x, matrix_left, matrix_right)


@ALGO_REGISTRY.register(
    name="flatquant",
    description="Learnable affine structure transform inspired by FlatQuant",
    targets=("structure",),
)
class FlatQuant(nn.Module):
    """
    Minimal FlatQuant core for the current framework:
    learn a structure-level affine transform on activations and apply its
    inverse-transpose to linear weights.

    This keeps the main idea from FlatQuant's fast learnable affine transforms,
    but does not port their full wrapper/model-tool/training stack.
    """

    def __init__(self, args, ctx):
        super().__init__()
        self.args = args
        self.dim_size = ctx.dim_size
        self.matrix_size = ctx.matrix_size or 128
        self.add_diag = bool(getattr(args, "flat_add_diag", True))

        if self.dim_size is None:
            raise ValueError("FlatQuant requires AlgoBuildContext.dim_size.")

        use_decompose = (
            self.matrix_size > 1
            and self.dim_size > self.matrix_size
            and self.dim_size % self.matrix_size == 0
        )

        if use_decompose:
            left_size = self.dim_size // self.matrix_size
            right_size = self.matrix_size
            self.transform = _InvFlatDecomposeTransform(left_size, right_size, self.add_diag)
        else:
            self.transform = _InvFlatSingleTransform(self.dim_size)

    def forward(self, x: torch.Tensor, inv_t: bool = False, name: str = None) -> torch.Tensor:
        return self.transform(x, inv_t=inv_t)

    def trainable_params(self):
        return list(self.parameters())

    def export_ptq_params(self):
        return {
            name: param.detach().cpu()
            for name, param in self.named_parameters()
        }

    def load_ptq_params(self, params):
        named_params = dict(self.named_parameters())
        for name, value in params.items():
            if name not in named_params:
                continue
            param = named_params[name]
            param.data.copy_(value.to(device=param.device, dtype=param.dtype))
