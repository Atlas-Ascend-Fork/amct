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

from amct_pytorch.algorithms.quant.learnable_hadamard import (
    InvSingleTransMatrix,
    LearnableHadamard,
    get_random_orthg,
)

TRANSFORM_WEIGHT = 'transform_weight'


# ---- get_random_orthg ----------------------------------------------------


@pytest.mark.parametrize("size", [4, 8, 16])
def test_get_random_orthg_returns_orthogonal_matrix(size):
    q = get_random_orthg(size)
    assert q.shape == (size, size)
    assert torch.allclose(q @ q.T, torch.eye(size), atol=1e-5)


def test_get_random_orthg_respects_dtype_arg():
    q = get_random_orthg(4, dtype=torch.bfloat16)
    assert q.dtype == torch.bfloat16


def test_get_random_orthg_diag_signs_made_positive():
    q = get_random_orthg(8)
    # Determinant should be +1 because diag-sign flip produces SO(n).
    assert torch.allclose(torch.linalg.det(q), torch.tensor(1.0), atol=1e-4)


# ---- InvSingleTransMatrix -----------------------------------------------


def test_inv_single_trans_matrix_init_uses_orthogonal_weight():
    m = InvSingleTransMatrix(size=4)
    w = m.linear.weight
    assert torch.allclose(w @ w.T, torch.eye(4), atol=1e-5)
    assert m._eval_mode is False
    assert m._matrix is None
    assert m._matrix_inv_t is None


def test_inv_single_trans_matrix_forward_equals_inv_t_for_orthogonal_init():
    # forward returns x @ M when inv_t=False and x @ inv(M).T when inv_t=True.
    # For an orthogonal matrix M, inv(M).T == M, so the two outputs match.
    m = InvSingleTransMatrix(size=4)
    x = torch.randn(2, 2, 4)
    fwd = m(x, inv_t=False)
    inv = m(x, inv_t=True)
    assert torch.allclose(fwd, inv, atol=1e-4)


def test_inv_single_trans_matrix_forward_preserves_shape():
    m = InvSingleTransMatrix(size=4)
    x = torch.randn(2, 3, 4)
    out = m(x, inv_t=False)
    assert out.shape == x.shape


def test_inv_single_trans_matrix_inv_t_uses_actual_inverse_transpose_for_general_weight():
    # Replace the orthogonal weight with a general invertible matrix and verify
    # that inv_t=True produces x @ inv(W).T.
    m = InvSingleTransMatrix(size=3)
    w = torch.tensor([[1.0, 2.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    with torch.no_grad():
        m.linear.weight.copy_(w)
    x = torch.randn(2, 3)
    expected = x @ torch.linalg.inv(w).T
    assert torch.allclose(m(x, inv_t=True), expected, atol=1e-4)


def test_inv_single_trans_matrix_to_eval_mode_caches_matrices():
    m = InvSingleTransMatrix(size=4)
    m.to_eval_mode()
    assert m._eval_mode is True
    assert m._matrix is not None
    assert m._matrix_inv_t is not None

    # Calling again is idempotent.
    cached_matrix = m._matrix
    m.to_eval_mode()
    assert m._matrix is cached_matrix


def test_inv_single_trans_matrix_get_matrix_returns_cached_when_eval():
    m = InvSingleTransMatrix(size=4)
    m.to_eval_mode()
    assert torch.equal(m.get_matrix(inv_t=False), m._matrix)
    assert torch.equal(m.get_matrix(inv_t=True), m._matrix_inv_t)


def test_inv_single_trans_matrix_repr_reports_eval_mode():
    m = InvSingleTransMatrix(size=4)
    assert "eval_mode=False" in repr(m)
    m.to_eval_mode()
    assert "eval_mode=True" in repr(m)
    assert "(4, 4)" in repr(m)


# ---- LearnableHadamard --------------------------------------------------


def _make_hadamard(matrix_size=4):
    return LearnableHadamard(
        args=SimpleNamespace(),
        ctx=SimpleNamespace(matrix_size=matrix_size),
    )


def test_learnable_hadamard_forward_equals_inv_t_for_orthogonal_init():
    layer = _make_hadamard(matrix_size=4)
    x = torch.randn(2, 8)  # last dim must be a multiple of matrix_size=4
    fwd = layer(x, inv_t=False)
    inv = layer(x, inv_t=True)
    assert torch.allclose(fwd, inv, atol=1e-4)


def test_learnable_hadamard_forward_preserves_shape():
    layer = _make_hadamard(matrix_size=4)
    x = torch.randn(2, 8)
    assert layer(x).shape == x.shape


def test_learnable_hadamard_trainable_params_includes_transform_weight():
    layer = _make_hadamard(matrix_size=4)
    params = layer.trainable_params()
    assert any(p is layer.transform.linear.weight for p in params)


def test_learnable_hadamard_export_load_round_trip():
    layer = _make_hadamard(matrix_size=4)
    params = layer.export_ptq_params()
    assert TRANSFORM_WEIGHT in params
    assert torch.equal(params[TRANSFORM_WEIGHT], layer.transform.linear.weight.detach().cpu())

    other = _make_hadamard(matrix_size=4)
    other.load_ptq_params(params)
    assert torch.equal(other.transform.linear.weight.data, layer.transform.linear.weight.data)
