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

"""Learnable Hadamard Transform (LHT) for structure-level PTQ.

Uses ``torch.nn.utils.parametrizations.orthogonal`` with the Cayley map to
keep ``linear.weight`` on the orthogonal group during training.  Because
``inv(Q).T == Q`` for any orthogonal ``Q``, the same matrix serves both the
activation and weight transforms without an explicit matrix inverse.

Training starts from a random rotation: the default dynamic trivialization
framework stores an orthogonal ``base`` and zeroes the Cayley parameter, so
``Q = base``.  ``base`` is part of the parametrization state (a buffer, not a
trainable parameter); ``export_ptq_params`` always exports the effective
``Q = base @ cayley(A)``, so the exported matrix is orthogonal by construction.
"""

from typing import Optional

import torch
from torch import nn
from torch.nn.utils.parametrizations import orthogonal
from torch.nn.utils.parametrize import is_parametrized, remove_parametrizations

from amct_pytorch.algorithms.quant.base import QuantAlgorithmBase
from amct_pytorch.algorithms.registry_factory import ALGO_REGISTRY


def _validate_input(x: torch.Tensor, size: int) -> None:
    if x.ndim == 0 or x.shape[-1] % size != 0:
        raise ValueError(
            f"input last dimension must be divisible by {size}, got shape {tuple(x.shape)}"
        )


class _LearnableTransform(nn.Module):
    """Trainable orthogonal transform via Cayley parametrization."""

    def __init__(self, size: int):
        super().__init__()
        self.linear = nn.Linear(size, size, bias=False)
        # Keep the default ``use_trivialization=True``: it stores an orthogonal
        # ``base`` and initializes the Cayley parameter to zero, so training
        # starts from a random rotation instead of the near-identity matrix that
        # a zero-initialized Cayley parameter would give.  Do not disable it to
        # "avoid the base buffer": with ``use_trivialization=False`` the Cayley
        # ``right_inverse`` raises NotImplementedError, which torch silently
        # swallows, leaving the near-identity initialization in place.
        orthogonal(self.linear, "weight", orthogonal_map="cayley")

    def forward(self, x: torch.Tensor, inv_t: bool = False) -> torch.Tensor:
        # inv_t satisfies the structure-transform contract but is a no-op here:
        # for an orthogonal Q, F.linear's transpose semantics make the inverse
        # transform the same matmul (W' = W @ Q^{-T} = W @ Q).
        _validate_input(x, self.linear.in_features)
        initial_shape = x.shape
        matrix = self.linear.weight.to(device=x.device, dtype=x.dtype)
        return x.reshape(-1, matrix.shape[0]).matmul(matrix).reshape(initial_shape)


@ALGO_REGISTRY.register(
    name="learnable_had",
    description="Learnable orthogonal structure transform (Cayley)",
    targets=("structure",),
)
class LearnableHadamard(QuantAlgorithmBase):
    """Learnable Hadamard Transform (LHT)."""

    def __init__(self, args, ctx):
        super().__init__()
        if not isinstance(ctx.matrix_size, int) or ctx.matrix_size <= 0:
            raise ValueError(
                f"matrix_size must be a positive integer, got {ctx.matrix_size!r}"
            )
        dim_size = getattr(ctx, "dim_size", None)
        if dim_size is not None and dim_size % ctx.matrix_size != 0:
            raise ValueError(
                f"dim_size {dim_size} must be divisible by matrix_size {ctx.matrix_size}"
            )
        self.args = args
        self.size = ctx.matrix_size
        self.transform = _LearnableTransform(self.size)

    def forward(
        self, x: torch.Tensor, inv_t: bool = False, name: Optional[str] = None
    ) -> torch.Tensor:
        if self.is_observe:
            return self.calib_forward(x, inv_t=inv_t, name=name)
        return self.transform(x, inv_t=inv_t)

    def trainable_params(self):
        # Once the parametrization is gone (i.e. the module was restored through
        # ``load_ptq_params``) ``linear.weight`` is a free matrix that no longer
        # lies on the orthogonal group, so continuing to train it would silently
        # break the ``inv_t`` no-op and hence the linear-equivalence contract.
        # Deriving this from the parametrization state (instead of a load flag)
        # keeps the answer correct across ``state_dict``/``deepcopy`` round-trips.
        if not is_parametrized(self.transform.linear, "weight"):
            return []
        return list(self.parameters())

    def export_ptq_params(self):
        return {"transform_weight": self.transform.linear.weight.detach().cpu()}

    def load_ptq_params(self, params):
        """Load an exported orthogonal matrix for inference reproduction.

        Loading removes the Cayley parametrization, so ``linear.weight``
        becomes a plain parameter whose ``state_dict`` key changes from
        ``transform.linear.parametrizations.weight.original`` to
        ``transform.linear.weight``; the two key layouts are not interchangeable.
        The loaded matrix is validated to stay on the orthogonal group and the
        module is frozen, because loaded modules are meant for eval/deploy only.
        """
        if "transform_weight" not in params:
            raise ValueError(
                f"params must contain 'transform_weight', got keys {sorted(params)}"
            )
        weight = params["transform_weight"]
        if tuple(weight.shape) != tuple(self.transform.linear.weight.shape):
            raise ValueError(
                f"transform_weight shape must be {tuple(self.transform.linear.weight.shape)}, "
                f"got {tuple(weight.shape)}"
            )
        self._check_orthogonal(weight)
        linear = self.transform.linear
        if is_parametrized(linear, "weight"):
            remove_parametrizations(linear, "weight")
        with torch.no_grad():
            linear.weight.copy_(weight.to(linear.weight))
            linear.weight.requires_grad_(False)

    def _check_orthogonal(self, weight: torch.Tensor) -> None:
        """Reject matrices that are not orthogonal enough for ``inv_t`` to be a no-op."""
        if not weight.is_floating_point():
            raise ValueError(
                f"transform_weight must be floating point, got dtype {weight.dtype}"
            )
        # Validate on the CPU source in float64: the check must not depend on
        # device-side float64 support and must not be affected by the operand
        # dtype that the matrices will be cast to later.
        matrix = weight.detach().to(device="cpu", dtype=torch.float64)
        eye = torch.eye(matrix.shape[0], dtype=torch.float64)
        residual = (matrix @ matrix.T - eye).abs().max().item()
        # Same contract as the deploy-side loader (vllm-ascend
        # ``validate_lht_matrix``): a valid export is orthogonal to ~1e-6, while
        # a matrix that has drifted off the group during unconstrained training
        # is off by orders of magnitude more.
        tolerance = max(1e-4, 2 * torch.finfo(weight.dtype).eps)
        if not torch.isfinite(matrix).all() or residual > tolerance:
            raise ValueError(
                f"transform_weight must be orthogonal, got max|W W^T - I|={residual:.3e} "
                f"(tolerance {tolerance:.1e}); export the effective matrix produced by "
                "the Cayley parametrization, not a raw training parameter"
            )
