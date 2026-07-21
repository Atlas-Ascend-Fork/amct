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
#
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

from amct_pytorch.algorithms.quant.flatquant import (
    FlatQuant,
    _InvFlatDecomposeTransform,
    _InvFlatSingleTransform,
    _kronecker_matmul,
    _random_orthogonal,
)


def _make(dim_size=4, matrix_size=2):
    args = SimpleNamespace()
    ctx = SimpleNamespace(dim_size=dim_size, matrix_size=matrix_size)
    return FlatQuant(args=args, ctx=ctx)


def test_random_orthogonal_returns_orthogonal_matrix_with_requested_dtype():
    torch.manual_seed(0)
    matrix = _random_orthogonal(4, dtype=torch.float64)
    assert matrix.dtype == torch.float64
    eye = torch.eye(4, dtype=torch.float64)
    assert torch.allclose(matrix.T @ matrix, eye, atol=1e-6)


def test_kronecker_matmul_with_identity_matrices_is_passthrough():
    x = torch.arange(8, dtype=torch.float32).reshape(1, 8)
    left = torch.eye(2)
    right = torch.eye(4)
    assert torch.equal(_kronecker_matmul(x, left, right), x)


def test_kronecker_matmul_preserves_original_shape():
    x = torch.arange(16, dtype=torch.float32).reshape(2, 1, 8)
    left = torch.eye(2)
    right = torch.eye(4)
    out = _kronecker_matmul(x, left, right)
    assert out.shape == x.shape


def test_single_transform_is_used_when_dim_is_not_decomposable():
    flat = _make(dim_size=5, matrix_size=3)
    assert isinstance(flat.transform, _InvFlatSingleTransform)


def test_single_transform_forward_preserves_shape_and_dtype():
    torch.manual_seed(0)
    transform = _InvFlatSingleTransform(4)
    x = torch.ones(2, 3, 4, dtype=torch.float64)
    out = transform(x)
    assert out.shape == x.shape
    assert out.dtype == torch.float64


def test_decompose_transform_is_used_when_dim_is_multiple_of_matrix_size():
    flat = _make(dim_size=8, matrix_size=4)
    assert isinstance(flat.transform, _InvFlatDecomposeTransform)
    assert flat.transform.linear_left.weight.shape == (2, 2)
    assert flat.transform.linear_right.weight.shape == (4, 4)
    assert flat.transform.diag_scale.shape == (8,)


def test_decompose_transform_without_diag_has_no_diag_scale_parameter():
    transform = _InvFlatDecomposeTransform(left_size=2, right_size=4, add_diag=False)
    assert transform.add_diag is False
    assert not hasattr(transform, "diag_scale")


def test_decompose_transform_without_diag_forward_runs_without_diag_scale():
    transform = _InvFlatDecomposeTransform(left_size=2, right_size=2, add_diag=False)
    with torch.no_grad():
        transform.linear_left.weight.copy_(torch.eye(2))
        transform.linear_right.weight.copy_(torch.eye(2))
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    assert torch.equal(transform(x), x)


def test_forward_uses_diag_scale_for_decomposed_activation_transform():
    flat = _make(dim_size=4, matrix_size=2)
    with torch.no_grad():
        flat.transform.linear_left.weight.copy_(torch.eye(2))
        flat.transform.linear_right.weight.copy_(torch.eye(2))
        flat.transform.diag_scale.copy_(torch.tensor([1.0, 2.0, 3.0, 4.0]))

    x = torch.ones(1, 4)
    out = flat(x, inv_t=False)
    assert torch.allclose(out, torch.tensor([[1.0, 2.0, 3.0, 4.0]]))


def test_forward_inv_t_divides_by_diag_scale_for_decomposed_weight_transform():
    flat = _make(dim_size=4, matrix_size=2)
    with torch.no_grad():
        flat.transform.linear_left.weight.copy_(torch.eye(2))
        flat.transform.linear_right.weight.copy_(torch.eye(2))
        flat.transform.diag_scale.copy_(torch.tensor([1.0, 2.0, 4.0, 8.0]))

    weight = torch.ones(1, 4)
    out = flat(weight, inv_t=True)
    assert torch.allclose(out, torch.tensor([[1.0, 0.5, 0.25, 0.125]]))


def test_flatquant_transform_keeps_linear_result_equivalent():
    flat = _make(dim_size=4, matrix_size=2)
    with torch.no_grad():
        flat.transform.linear_left.weight.copy_(torch.tensor([[0.0, 1.0], [1.0, 0.0]]))
        flat.transform.linear_right.weight.copy_(torch.tensor([[0.0, 1.0], [1.0, 0.0]]))
        flat.transform.diag_scale.copy_(torch.tensor([1.0, 2.0, 3.0, 4.0]))

    x = torch.tensor([[1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 5.0]])
    weight = torch.tensor([[1.0, 0.5, 0.25, 0.125], [2.0, 1.0, 0.5, 0.25]])
    transformed_x = flat(x)
    transformed_weight = flat(weight, inv_t=True)
    assert torch.allclose(F.linear(transformed_x, transformed_weight), F.linear(x, weight), atol=1e-5)


def test_constructor_raises_when_dim_size_is_missing():
    args = SimpleNamespace()
    ctx = SimpleNamespace(dim_size=None, matrix_size=2)
    with pytest.raises(ValueError, match="dim_size"):
        FlatQuant(args=args, ctx=ctx)


def test_default_matrix_size_is_used_when_ctx_matrix_size_is_none():
    args = SimpleNamespace()
    ctx = SimpleNamespace(dim_size=256, matrix_size=None)
    flat = FlatQuant(args=args, ctx=ctx)
    assert flat.matrix_size == 128
    assert isinstance(flat.transform, _InvFlatDecomposeTransform)


def test_trainable_params_returns_all_parameters():
    flat = _make(dim_size=4, matrix_size=2)
    trainable_params = flat.trainable_params()
    parameters = list(flat.parameters())
    assert len(trainable_params) == len(parameters)
    assert all(actual is expected for actual, expected in zip(trainable_params, parameters))


def test_export_load_round_trip_for_decomposed_transform():
    flat = _make(dim_size=4, matrix_size=2)
    with torch.no_grad():
        flat.transform.linear_left.weight.copy_(torch.tensor([[0.0, 1.0], [1.0, 0.0]]))
        flat.transform.linear_right.weight.copy_(torch.tensor([[0.0, 1.0], [1.0, 0.0]]))
        flat.transform.diag_scale.copy_(torch.tensor([1.0, 2.0, 3.0, 4.0]))

    params = flat.export_ptq_params()
    assert set(params) == {
        "transform.linear_left.weight",
        "transform.linear_right.weight",
        "transform.diag_scale",
    }

    other = _make(dim_size=4, matrix_size=2)
    other.load_ptq_params(params)
    for name, value in flat.named_parameters():
        assert torch.equal(dict(other.named_parameters())[name].data, value.data)


def test_load_ptq_params_ignores_unknown_parameter_names():
    flat = _make(dim_size=4, matrix_size=2)
    original = {name: param.detach().clone() for name, param in flat.named_parameters()}
    flat.load_ptq_params({"unknown": torch.ones(1)})
    for name, param in flat.named_parameters():
        assert torch.equal(param.data, original[name])
