# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
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
#
# The code snippet comes from FlatQuant (https://github.com/ruikangliu/FlatQuant).
#
# Copyright (c) 2024 ruikangliu. All rights reserved.
#
# This file is part of FlatQuant, which is released under the MIT License.
# See the LICENSE file in the root directory of this source tree
# or at https://opensource.org/licenses/MIT for details.
#
# --------------------------------------------------------------------------------
"""Learnable invertible transforms for FlatQuant.

Two structures, both SVD-parameterized with Cayley-orthogonal factors:
- SVDSingleTransMatrix: one matrix, acts on a single structural dim
  (o_trans on num_heads, v/kcache_trans on head_dim)
- SVDDecomposeTransMatrix: Kronecker pair + optional diag scale,
  acts on the whole hidden dim (ln_trans / up_gate_trans / down_trans)"""

import math

import numpy as np
import torch
from scipy.linalg import qr


def get_init_weight(size):
    """Random orthogonal init for the Cayley factors."""
    mat = np.random.randn(size, size)
    q_mat, r_mat = qr(mat)
    q_modified = q_mat @ np.diag(np.sign(np.diag(r_mat)))
    return torch.from_numpy(q_modified).float()


def get_decompose_dim(n):
    """Split hidden dim n into (d1, d2) with d1*d2 == n and d1 ~ d2 ~ sqrt(n)."""
    if n % 4 == 2:
        # no integers a, b satisfy a^2 - b^2 == n; the search below would
        # never terminate, so fail fast with the offending dim
        raise ValueError(
            f'cannot decompose dim {n} into (a-b)(a+b) factors: n % 4 == 2 '
            'has no integer solution. Kronecker transform dims must not be '
            'of form 4k+2 (e.g. 6, 10, 14, ...).'
        )
    a = int(math.sqrt(n))
    if a * a < n:
        a += 1
    while True:
        tmp = a * a - n
        b = int(math.sqrt(tmp))
        if b * b == tmp:
            break
        a += 1
    return a - b, a + b


def get_init_scale(w_smax, x_smax, alpha=0.5):
    """Init diag scale from weight/activation statistics (smoothquant-like)."""
    return (w_smax.pow(1 - alpha) / x_smax.pow(alpha)).clamp(min=1e-5)


def kronecker_matmul(x, left, right):
    """x @ kron(left, right) without materializing the Kronecker product.

    Equivalent to:
        had = torch.kron(left, right)
        x.reshape(-1, had.shape[0]).matmul(had).reshape(init_shape)
    """
    init_shape = x.shape
    had = torch.kron(left, right)
    x = x.reshape(-1, had.shape[0])
    x = x.matmul(had)
    return x.reshape(init_shape)


class SVDSingleTransMatrix(torch.nn.Module):
    """P = U @ diag(sigma) @ V^T on one dim; U/V Cayley-orthogonal, sigma init 1."""

    def __init__(self, size):
        super().__init__()
        self.linear_u = torch.nn.Linear(size, size, bias=False, dtype=torch.float32)
        self.linear_u.weight.data = get_init_weight(size).to(self.linear_u.weight)
        self.linear_u = torch.nn.utils.parametrizations.orthogonal(
            self.linear_u, orthogonal_map='cayley', use_trivialization=False
        )
        self.linear_v = torch.nn.Linear(size, size, bias=False, dtype=torch.float32)
        self.linear_v.weight.data = get_init_weight(size).to(self.linear_v.weight)
        self.linear_v = torch.nn.utils.parametrizations.orthogonal(
            self.linear_v, orthogonal_map='cayley', use_trivialization=False
        )
        self.linear_diag = torch.nn.Parameter(
            torch.ones(size, dtype=torch.float32), requires_grad=True
        )
        self._eval_mode = False

    def forward(self, inp, inv_t=False):
        init_shape = inp.shape
        matrix = self.get_matrix(inv_t=inv_t).to(inp)
        inp = inp.reshape(-1, matrix.shape[0])
        return inp.matmul(matrix).reshape(init_shape)

    def get_matrix(self, inv_t=False):
        if self._eval_mode:
            return self.matrix_inv_t if inv_t else self.matrix
        orthog_u, orthog_v = self.linear_u.weight, self.linear_v.weight
        linear_diag = self.linear_diag
        if inv_t:
            linear_diag = 1 / linear_diag
        return orthog_u @ torch.diag(linear_diag) @ orthog_v.t()

    def to_eval_mode(self):
        """Freeze the factorized form into plain matrices, drop factor params."""
        if self._eval_mode:
            return
        self.matrix = torch.nn.Parameter(
            self.linear_u.weight
            @ torch.diag(self.linear_diag)
            @ self.linear_v.weight.t(),
            requires_grad=False,
        )
        self.matrix_inv_t = torch.nn.Parameter(
            self.linear_u.weight
            @ torch.diag(1 / self.linear_diag)
            @ self.linear_v.weight.t(),
            requires_grad=False,
        )
        self._eval_mode = True
        del self.linear_u, self.linear_diag, self.linear_v

    def __repr__(self):
        res = f'SVDSingleTransMatrix(eval_mode={self._eval_mode})'
        return res


class SVDDecomposeTransMatrix(torch.nn.Module):
    """P = P_left (x) P_right on (d1, d2), plus optional per-channel diag scale.

    diag_scale is the piece that reparameterizes into the preceding Norm's
    weight at convert time (gamma <- gamma * c); use_diag switches it off once
    merged."""

    def __init__(self, left_size, right_size, add_diag=False, diag_init_para=None):
        super().__init__()
        self.linear_u_left = torch.nn.Linear(
            left_size, left_size, bias=False, dtype=torch.float32
        )
        self.linear_u_left.weight.data = get_init_weight(left_size).to(
            self.linear_u_left.weight
        )
        self.linear_u_left = torch.nn.utils.parametrizations.orthogonal(
            self.linear_u_left, orthogonal_map='cayley', use_trivialization=False
        )
        self.linear_v_left = torch.nn.Linear(
            left_size, left_size, bias=False, dtype=torch.float32
        )
        self.linear_v_left.weight.data = get_init_weight(left_size).to(
            self.linear_v_left.weight
        )
        self.linear_v_left = torch.nn.utils.parametrizations.orthogonal(
            self.linear_v_left, orthogonal_map='cayley', use_trivialization=False
        )
        self.linear_diag_left = torch.nn.Parameter(
            torch.ones(left_size, dtype=torch.float32), requires_grad=True
        )

        self.linear_u_right = torch.nn.Linear(
            right_size, right_size, bias=False, dtype=torch.float32
        )
        self.linear_u_right.weight.data = get_init_weight(right_size).to(
            self.linear_u_right.weight
        )
        self.linear_u_right = torch.nn.utils.parametrizations.orthogonal(
            self.linear_u_right, orthogonal_map='cayley', use_trivialization=False
        )
        self.linear_v_right = torch.nn.Linear(
            right_size, right_size, bias=False, dtype=torch.float32
        )
        self.linear_v_right.weight.data = get_init_weight(right_size).to(
            self.linear_v_right.weight
        )
        self.linear_v_right = torch.nn.utils.parametrizations.orthogonal(
            self.linear_v_right, orthogonal_map='cayley', use_trivialization=False
        )
        self.linear_diag_right = torch.nn.Parameter(
            torch.ones(right_size, dtype=torch.float32), requires_grad=True
        )

        self.add_diag = add_diag
        self.use_diag = True
        if self.add_diag:
            device = self.linear_u_left.weight.device
            if diag_init_para is None:
                self.diag_scale = torch.nn.Parameter(
                    torch.ones((left_size * right_size), dtype=torch.float32).to(
                        device
                    ),
                    requires_grad=True,
                )
            else:
                self.diag_scale = torch.nn.Parameter(
                    diag_init_para.to(device), requires_grad=True
                )
        self._eval_mode = False

    def forward(self, inp, inv_t=False):
        if self.add_diag and self.use_diag:
            if inv_t:
                inp = inp / self.diag_scale.to(inp)
            else:
                inp = inp * self.diag_scale.to(inp)
        if not self._eval_mode:
            diag_left = self.linear_diag_left
            diag_right = self.linear_diag_right
            if inv_t:
                diag_left = 1 / diag_left
                diag_right = 1 / diag_right
            matrix_left = (
                self.linear_u_left.weight
                @ torch.diag(diag_left)
                @ self.linear_v_left.weight.t()
            )
            matrix_right = (
                self.linear_u_right.weight
                @ torch.diag(diag_right)
                @ self.linear_v_right.weight.t()
            )
        else:
            matrix_left, matrix_right = self.matrix_left, self.matrix_right
            if inv_t:
                matrix_left, matrix_right = (
                    self.matrix_left_inv,
                    self.matrix_right_inv,
                )
        return kronecker_matmul(inp, matrix_left.to(inp), matrix_right.to(inp))

    def to_eval_mode(self):
        if self._eval_mode:
            return
        self.matrix_left = torch.nn.Parameter(
            self.linear_u_left.weight
            @ torch.diag(self.linear_diag_left)
            @ self.linear_v_left.weight.t(),
            requires_grad=False,
        )
        self.matrix_right = torch.nn.Parameter(
            self.linear_u_right.weight
            @ torch.diag(self.linear_diag_right)
            @ self.linear_v_right.weight.t(),
            requires_grad=False,
        )
        self.matrix_left_inv = torch.nn.Parameter(
            self.linear_u_left.weight
            @ torch.diag(1 / self.linear_diag_left)
            @ self.linear_v_left.weight.t(),
            requires_grad=False,
        )
        self.matrix_right_inv = torch.nn.Parameter(
            self.linear_u_right.weight
            @ torch.diag(1 / self.linear_diag_right)
            @ self.linear_v_right.weight.t(),
            requires_grad=False,
        )
        self._eval_mode = True
        del (
            self.linear_u_left,
            self.linear_diag_left,
            self.linear_v_left,
            self.linear_u_right,
            self.linear_diag_right,
            self.linear_v_right,
        )

    def __repr__(self):
        res = f'SVDDecomposeTransMatrix(_eval_mode={self._eval_mode})'
        return res
