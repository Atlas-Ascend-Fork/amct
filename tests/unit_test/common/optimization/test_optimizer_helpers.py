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
import pytest
import torch
import torch.nn as nn

from amct_pytorch.common.optimization.optimizer import (
    SGDG,
    Cayley_loop,
    matrix_norm_one,
    norm,
    qr_retraction,
    unit,
)


def test_norm_returns_l2_keepdim():
    v = torch.tensor([[3.0, 4.0], [0.0, 5.0]])
    out = norm(v, dim=1)
    assert out.shape == (2, 1)
    assert torch.allclose(out, torch.tensor([[5.0], [5.0]]))


def test_norm_asserts_2d_input():
    with pytest.raises(AssertionError):
        norm(torch.zeros(3))


def test_unit_normalizes_to_unit_vectors_and_returns_norm():
    v = torch.tensor([[3.0, 4.0]])
    normalized, n = unit(v)
    assert torch.allclose(n, torch.tensor([[5.0]]))
    assert torch.allclose(normalized.norm(dim=1), torch.tensor([1.0]), atol=1e-5)


def test_unit_handles_zero_vector_via_eps():
    v = torch.zeros(1, 3)
    normalized, n = unit(v, eps=1e-8)
    assert n.item() == pytest.approx(0.0)
    # eps avoids division by zero -> result is near-zero (not NaN).
    assert torch.isfinite(normalized).all()


def test_matrix_norm_one_is_max_column_abs_sum():
    # column sums of |W|: col0=6, col1=15
    w = torch.tensor([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]])
    assert matrix_norm_one(w).item() == pytest.approx(15.0)


def test_cayley_loop_at_zero_step_returns_x_transposed():
    x = torch.eye(3, 2)             # 3x2 (n, p)
    w = torch.zeros(3, 3)
    tan = torch.zeros(3, 2)
    out = Cayley_loop(x, w, tan, t=0.0)
    # When t=0 -> Y = X; the function returns Y.t() with shape (p, n).
    assert out.shape == (2, 3)
    assert torch.allclose(out, x.t())


def test_qr_retraction_yields_orthonormal_rows():
    # qr_retraction expects p-by-n with p<=n. After call it transposes back so
    # the returned tensor has the same orientation.
    tan = torch.randn(3, 5)
    q = qr_retraction(tan.clone())
    # Row vectors should be orthonormal.
    assert q.shape == (3, 5)
    assert torch.allclose(q @ q.t(), torch.eye(3), atol=1e-5)


class TestSGDG:
    def test_init_basic(self):
        model = nn.Linear(4, 4)
        opt = SGDG(model.parameters(), lr=0.01)
        assert opt.defaults["lr"] == 0.01
        assert opt.defaults["stiefel"] is False

    def test_init_nesterov_requires_momentum(self):
        model = nn.Linear(4, 4)
        with pytest.raises(ValueError, match="Nesterov momentum"):
            SGDG(model.parameters(), lr=0.01, nesterov=True)

    def test_setstate_sets_nesterov_default(self):
        model = nn.Linear(4, 4)
        opt = SGDG(model.parameters(), lr=0.01)
        opt.__setstate__({})
        for g in opt.param_groups:
            assert "nesterov" in g

    def test_step_skips_param_without_grad(self):
        model = nn.Linear(4, 4)
        opt = SGDG(model.parameters(), lr=0.01)
        model.weight.grad = None
        loss = opt.step()
        assert loss is None

    def test_step_standard_sgd(self):
        model = nn.Linear(4, 4)
        w_before = model.weight.data.clone()
        model.weight.grad = torch.ones_like(model.weight)
        opt = SGDG(model.parameters(), lr=0.01)
        opt.step()
        assert not torch.equal(model.weight.data, w_before)

    def test_step_stiefel_path(self):
        torch.random.manual_seed(42)
        model = nn.Linear(5, 3)
        model.weight.grad = torch.randn_like(model.weight)
        opt = SGDG(model.parameters(), lr=0.01, stiefel=True)
        w_before = model.weight.data.clone()
        opt.step()
        assert not torch.equal(model.weight.data, w_before)
        assert "momentum_buffer" in opt.state[model.weight]

    def test_step_with_closure(self):
        model = nn.Linear(4, 4)
        model.weight.grad = torch.ones_like(model.weight)
        opt = SGDG(model.parameters(), lr=0.01)
        closure_called = []

        def closure():
            closure_called.append(1)
            return torch.tensor(0.5)
        loss = opt.step(closure=closure)
        assert closure_called == [1]
        assert loss == 0.5

    def test_step_non_stiefel_with_weight_decay(self):
        model = nn.Linear(4, 4)
        model.weight.grad = torch.ones_like(model.weight)
        opt = SGDG(model.parameters(), lr=0.01, weight_decay=0.1)
        w_before = model.weight.data.clone()
        opt.step()
        assert not torch.equal(model.weight.data, w_before)

    def test_step_non_stiefel_stiefel_flag_runs_qr_retraction(self, monkeypatch):
        import random
        monkeypatch.setattr(random, "randint", lambda a, b: 1)
        model = nn.Linear(5, 3)
        model.weight.grad = torch.randn_like(model.weight)
        opt = SGDG(model.parameters(), lr=0.01, stiefel=True, momentum=0.9)
        w_before = model.weight.data.clone()
        opt.step()
        assert not torch.equal(model.weight.data, w_before)


def test_sgd_step_with_weight_decay():
    """Cover the weight_decay != 0 branch in SGD step."""
    model = nn.Linear(4, 4)
    opt = SGDG(model.parameters(), lr=0.01, weight_decay=0.001, momentum=0, stiefel=False)
    inp = torch.randn(2, 4)
    loss = model(inp).sum()
    loss.backward()
    opt.step()


