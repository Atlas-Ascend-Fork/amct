#!/usr/bin/env python3
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

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from amct_pytorch.algorithms.quant import AlgoBuildContext, register_algorithms
from amct_pytorch.algorithms.quant.learnable_hadamard import (
    LearnableHadamard,
    _LearnableTransform,
)
from amct_pytorch.algorithms.registry_factory import ALGO_REGISTRY
from amct_pytorch.common.models.llm.common.ptq_params import PtqParamHandler
from amct_pytorch.quantization.modules.quant_base import build_algorithms_by_target

# ---------- _LearnableTransform ----------


def test_learnable_transform_is_trainable():
    transform = _LearnableTransform(4)
    x = torch.randn(2, 3, 8)
    assert transform(x).shape == x.shape
    assert transform.linear.weight.requires_grad


def test_learnable_transform_weight_is_orthogonal():
    transform = _LearnableTransform(4)
    weight = transform.linear.weight
    assert torch.allclose(weight @ weight.T, torch.eye(4), atol=1e-4)
    assert abs(abs(torch.linalg.det(weight).item()) - 1.0) < 1e-4


def test_initial_transform_is_random_orthogonal_not_identity():
    """Training must start from a random rotation, not from the identity.

    ``use_trivialization=False`` makes torch fall back to the near-identity
    initialization (the Cayley ``right_inverse`` raises NotImplementedError and
    is swallowed), which silently changes the algorithm.  Guard the documented
    behaviour: for a Haar-distributed rotation ``||Q - I||_F / sqrt(2n)`` is
    close to 1, while the near-identity case sits around 0.4.
    """
    torch.manual_seed(0)
    size = 128
    transform = _LearnableTransform(size)
    weight = transform.linear.weight.detach()
    eye = torch.eye(size)
    deviation = (weight - eye).norm() / (2 * size) ** 0.5
    assert 0.9 <= deviation <= 1.1, (
        f"initial transform is not a random rotation: {deviation}"
    )
    assert (weight @ weight.T - eye).abs().max() < 1e-5


def test_learnable_transform_preserves_shape():
    transform = _LearnableTransform(8)
    x = torch.randn(2, 3, 16)
    assert transform(x).shape == x.shape


def test_learnable_transform_inv_t_equals_forward():
    transform = _LearnableTransform(4)
    x = torch.randn(2, 8)
    assert torch.equal(transform(x), transform(x, inv_t=True))


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_learnable_transform_supports_half_dtypes(dtype):
    transform = _LearnableTransform(4)
    x = torch.randn(2, 8).to(dtype)
    out = transform(x)
    assert out.shape == x.shape
    assert out.dtype == dtype
    expected = (
        x.float()
        .reshape(-1, 4)
        .matmul(transform.linear.weight.float())
        .reshape(x.shape)
    )
    torch.testing.assert_close(out.float(), expected, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize(
    "dtype, tol",
    [(torch.bfloat16, 5e-2), (torch.float16, 2e-2)],
)
def test_learnable_transform_half_dtype_equivalence_at_default_size(dtype, tol):
    torch.manual_seed(0)
    size = 128
    transform = _LearnableTransform(size)
    x = torch.randn(2, 256).to(dtype)
    out = transform(x)
    expected = (
        x.float()
        .reshape(-1, size)
        .matmul(transform.linear.weight.float())
        .reshape(x.shape)
    )
    torch.testing.assert_close(out.float(), expected, rtol=tol, atol=tol)


# ---------- Orthogonal invariance ----------


@pytest.mark.parametrize(
    "dtype, tol",
    [(torch.bfloat16, 2e-2), (torch.float16, 5e-3)],
)
def test_linear_equivalence_holds_in_half_precision_at_default_size(dtype, tol):
    """The only invariant LHT relies on: F.linear(x@Q, W@Q) == x@W^T.

    Check it at the production default ``matrix_size=128`` and in the dtypes
    used for real runs.  Casting the orthogonal matrix to half precision is the
    dominant error term, so the tolerance is a per-dtype budget rather than the
    fp32 one used by the small-matrix test below.
    """
    torch.manual_seed(0)
    size = 128
    algorithm = LearnableHadamard(
        SimpleNamespace(), SimpleNamespace(matrix_size=size, dim_size=4 * size)
    )
    x = torch.randn(2, 3, 4 * size)
    weight = torch.randn(8, 4 * size)
    reference = F.linear(x, weight)
    transformed = F.linear(
        algorithm(x.to(dtype)).float(),
        algorithm(weight.to(dtype), inv_t=True).float(),
    )
    relative = (transformed - reference).norm() / reference.norm()
    assert relative < tol, (
        f"{dtype} linear equivalence error {relative:.3e} >= {tol:.1e}"
    )


def test_transform_and_inverse_preserve_linear_result():
    torch.manual_seed(0)
    algorithm = LearnableHadamard(SimpleNamespace(), SimpleNamespace(matrix_size=4))
    x = torch.randn(2, 8)
    weight = torch.randn(3, 8)
    assert torch.allclose(
        F.linear(algorithm(x), algorithm(weight, inv_t=True)),
        F.linear(x, weight),
        atol=1e-4,
    )


# ---------- LHT training step ----------


def test_learnable_transform_stays_orthogonal_after_optimizer_step():
    torch.manual_seed(0)
    algorithm = LearnableHadamard(
        SimpleNamespace(), SimpleNamespace(matrix_size=4, dim_size=8)
    )
    optimizer = torch.optim.AdamW(algorithm.trainable_params(), lr=1e-3)
    x = torch.randn(2, 8)
    loss = algorithm(x).square().mean()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    weight = algorithm.transform.linear.weight
    assert torch.allclose(weight @ weight.T, torch.eye(4), atol=1e-4)


# ---------- Export / Load round-trip ----------


def test_export_load_round_trip():
    torch.manual_seed(0)
    algorithm = LearnableHadamard(SimpleNamespace(), SimpleNamespace(matrix_size=4))
    params = algorithm.export_ptq_params()
    other = LearnableHadamard(SimpleNamespace(), SimpleNamespace(matrix_size=4))
    other.load_ptq_params(params)
    assert torch.allclose(other(torch.eye(4)), algorithm(torch.eye(4)), atol=1e-5)


def test_export_contains_effective_orthogonal_matrix():
    algorithm = LearnableHadamard(SimpleNamespace(), SimpleNamespace(matrix_size=4))
    exported = algorithm.export_ptq_params()["transform_weight"]
    assert torch.allclose(exported @ exported.T, torch.eye(4), atol=1e-4)
    torch.testing.assert_close(exported, algorithm.transform.linear.weight)


def test_load_rejects_wrong_shape():
    algorithm = LearnableHadamard(SimpleNamespace(), SimpleNamespace(matrix_size=4))
    with pytest.raises(ValueError, match="shape"):
        algorithm.load_ptq_params({"transform_weight": torch.eye(8)})


def test_load_rejects_non_orthogonal_matrix():
    """A drifted / arbitrary matrix must not silently disable the inv_t shortcut."""
    algorithm = LearnableHadamard(SimpleNamespace(), SimpleNamespace(matrix_size=4))
    with pytest.raises(ValueError, match="orthogonal"):
        algorithm.load_ptq_params({"transform_weight": 2.0 * torch.eye(4)})


def test_loaded_module_is_frozen():
    """Loaded modules are eval/deploy only: they must not stay trainable."""
    torch.manual_seed(0)
    algorithm = LearnableHadamard(SimpleNamespace(), SimpleNamespace(matrix_size=4))
    algorithm.load_ptq_params(algorithm.export_ptq_params())
    assert algorithm.trainable_params() == []
    assert not algorithm.transform.linear.weight.requires_grad
    # Forward must still work after the parametrization has been removed.
    assert algorithm(torch.eye(4)).shape == (4, 4)


def test_ptq_param_handler_export_load_reexport_consistent():
    torch.manual_seed(0)

    def new_parent():
        parent = nn.Module()
        parent.transform = LearnableHadamard(
            SimpleNamespace(), SimpleNamespace(matrix_size=4)
        )
        return parent

    source = new_parent()
    params = PtqParamHandler.export_module(source)
    assert "transform" in params
    assert "transform_weight" in params["transform"]

    target = new_parent()
    PtqParamHandler.load_module(target, params)

    x = torch.randn(2, 8)
    torch.testing.assert_close(
        target.transform(x), source.transform(x), rtol=0, atol=1e-6
    )

    reexported = PtqParamHandler.export_module(target)
    torch.testing.assert_close(
        reexported["transform"]["transform_weight"],
        params["transform"]["transform_weight"],
        rtol=0,
        atol=1e-6,
    )


# ---------- Input validation ----------


def test_input_rejects_incompatible_block_size():
    algorithm = LearnableHadamard(SimpleNamespace(), SimpleNamespace(matrix_size=4))
    with pytest.raises(ValueError, match="last dimension"):
        algorithm(torch.randn(4, 3))


# ---------- Construction validation ----------


def test_requires_matrix_size():
    with pytest.raises(ValueError, match="matrix_size"):
        LearnableHadamard(SimpleNamespace(), SimpleNamespace(matrix_size=None))


@pytest.mark.parametrize("bad_size", [0, -4])
def test_rejects_non_positive_matrix_size(bad_size):
    with pytest.raises(ValueError, match="positive integer"):
        LearnableHadamard(
            SimpleNamespace(), SimpleNamespace(matrix_size=bad_size, dim_size=16)
        )


def test_rejects_incompatible_dim_size():
    with pytest.raises(ValueError, match="dim_size"):
        LearnableHadamard(
            SimpleNamespace(), SimpleNamespace(matrix_size=4, dim_size=10)
        )


# ---------- Registry & framework integration ----------


def test_algorithm_registers():
    register_algorithms()
    assert ALGO_REGISTRY.get("learnable_had") is LearnableHadamard


def test_framework_builds_learnable_had():
    register_algorithms()
    args = SimpleNamespace(algos=["learnable_had"])
    built = build_algorithms_by_target(
        args, "structure", AlgoBuildContext(matrix_size=4)
    )
    assert isinstance(built, LearnableHadamard)


def test_framework_builds_with_configured_k_size():
    register_algorithms()
    args = SimpleNamespace(algos=["learnable_had"], k_size=8)
    ctx = AlgoBuildContext(matrix_size=args.k_size, dim_size=16)
    built = build_algorithms_by_target(args, "structure", ctx)
    assert built.size == 8


def test_observe_mode_returns_input_unchanged():
    algorithm = LearnableHadamard(SimpleNamespace(), SimpleNamespace(matrix_size=4))
    algorithm.is_observe = True
    x = torch.randn(2, 8)
    out = algorithm(x, name="test.linear")
    assert torch.equal(out, x)
