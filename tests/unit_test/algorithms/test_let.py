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

from amct_pytorch.algorithms.quant.let import LET


def _make(dim_size=4):
    return LET(args=SimpleNamespace(), ctx=SimpleNamespace(dim_size=dim_size))


def test_initial_state():
    algo = _make(dim_size=8)
    assert algo.dim == 8
    assert algo.log_scale.shape == (1, 8)
    assert torch.equal(algo.log_scale.data, torch.zeros(1, 8))
    assert algo.is_observe is False


def test_forward_with_zero_log_scale_is_passthrough_division():
    algo = _make(dim_size=4)
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    # exp(0)=1 -> x / 1 = x.
    y = algo(x)
    assert torch.allclose(y, x)


def test_forward_inv_t_multiplies_by_scale():
    algo = _make(dim_size=4)
    with torch.no_grad():
        algo.log_scale.fill_(0.0)  # scale = 1
    x = torch.tensor([[2.0, 4.0, 6.0, 8.0]])
    y = algo(x, inv_t=True)
    assert torch.allclose(y, x)  # scale=1


def test_forward_with_log_scale_2_divides_input_by_e2():
    algo = _make(dim_size=4)
    with torch.no_grad():
        algo.log_scale.fill_(2.0)  # scale = e^2
    x = torch.ones(1, 4)
    y_div = algo(x, inv_t=False)
    y_mul = algo(x, inv_t=True)
    expected_scale = torch.exp(torch.tensor(2.0))
    assert torch.allclose(y_div, x / expected_scale, atol=1e-4)
    assert torch.allclose(y_mul, x * expected_scale, atol=1e-4)


def test_observe_mode_updates_log_scale_with_running_max():
    algo = _make(dim_size=2)
    algo.is_observe = True
    x = torch.tensor([[3.0, 5.0]])
    out = algo(x)
    assert out is x
    assert torch.equal(out, torch.tensor([[3.0, 5.0]]))
    # log_scale ≈ log(max(|x|)) = log([3, 5])
    assert torch.allclose(
        algo.log_scale.data, torch.log(torch.tensor([[3.0, 5.0]])), atol=1e-4
    )
    # A larger update raises log_scale; smaller does not.
    algo(torch.tensor([[1.0, 6.0]]))
    expected = torch.log(torch.tensor([[3.0, 6.0]]))
    assert torch.allclose(algo.log_scale.data, expected, atol=1e-4)


def test_calib_forward_records_activation_and_returns_input():
    algo = _make(dim_size=2)
    activation = torch.tensor([[3.0, 5.0]])
    snapshot = activation.clone()
    before = algo.log_scale.detach().clone()

    out = algo.calib_forward(activation, inv_t=False)

    assert out is activation
    assert torch.equal(out, snapshot)
    assert not torch.equal(algo.log_scale, before)


def test_calib_forward_skips_weight_recording_and_returns_input():
    algo = _make(dim_size=2)
    with torch.no_grad():
        algo.log_scale.copy_(torch.tensor([[0.2, 0.4]]))
    weight = torch.tensor([[3.0, 5.0]])
    snapshot = weight.clone()
    before = algo.log_scale.detach().clone()

    out = algo.calib_forward(weight, inv_t=True, name="q_proj")

    assert out is weight
    assert torch.equal(out, snapshot)
    assert torch.equal(algo.log_scale, before)


def test_get_scale_clamps_to_finite_range():
    algo = _make(dim_size=2)
    with torch.no_grad():
        algo.log_scale.fill_(50.0)  # exp(50) >> 1e4 -> should be clamped
    scale = algo._get_scale(dtype=torch.float32, device=torch.device("cpu"))
    assert scale.max().item() == pytest.approx(1e4)


def test_get_scale_clamps_low_floor():
    algo = _make(dim_size=2)
    with torch.no_grad():
        algo.log_scale.fill_(-50.0)  # exp(-50) << 1e-4 -> clamped up
    scale = algo._get_scale(dtype=torch.float32, device=torch.device("cpu"))
    assert scale.min().item() == pytest.approx(1e-4)


def test_export_load_round_trip():
    algo = _make(dim_size=4)
    with torch.no_grad():
        algo.log_scale.copy_(torch.tensor([[0.1, 0.2, 0.3, 0.4]]))
    params = algo.export_ptq_params()
    assert "log_scale" in params

    other = _make(dim_size=4)
    other.load_ptq_params(params)
    assert torch.equal(other.log_scale.data, algo.log_scale.data)


def test_load_ptq_params_is_no_op_when_log_scale_missing():
    algo = _make(dim_size=4)
    original = algo.log_scale.data.clone()
    algo.load_ptq_params({"unrelated": torch.zeros(1)})
    assert torch.equal(algo.log_scale.data, original)


def test_trainable_params_returns_log_scale():
    algo = _make(dim_size=4)
    params = algo.trainable_params()
    assert any(p is algo.log_scale for p in params)


def test_forward_preserves_input_dtype():
    algo = _make(dim_size=4)
    x = torch.ones(1, 4, dtype=torch.bfloat16)
    assert algo(x).dtype == torch.bfloat16


def test_transform_is_a_no_op():
    assert _make().transform() is None
