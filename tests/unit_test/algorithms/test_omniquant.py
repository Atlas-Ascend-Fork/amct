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

from amct_pytorch.algorithms.quant.omniquant import OmniQuant


def _make(dim_size=4):
    return OmniQuant(args=SimpleNamespace(), ctx=SimpleNamespace(dim_size=dim_size))


def test_initial_state():
    om = _make(dim_size=8)
    assert om.dim == 8
    assert om.log_scale.shape == (1, 8)
    assert torch.equal(om.log_scale.data, torch.zeros(1, 8))
    assert om.is_observe is False


def test_forward_with_zero_log_scale_is_passthrough_division():
    om = _make(dim_size=4)
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    # exp(0)=1 -> x / 1 = x.
    y = om(x)
    assert torch.allclose(y, x)


def test_forward_inv_t_multiplies_by_scale():
    om = _make(dim_size=4)
    with torch.no_grad():
        om.log_scale.fill_(0.0)  # scale = 1
    x = torch.tensor([[2.0, 4.0, 6.0, 8.0]])
    y = om(x, inv_t=True)
    assert torch.allclose(y, x)  # scale=1


def test_forward_with_log_scale_2_divides_input_by_e2():
    om = _make(dim_size=4)
    with torch.no_grad():
        om.log_scale.fill_(2.0)  # scale = e^2
    x = torch.ones(1, 4)
    y_div = om(x, inv_t=False)
    y_mul = om(x, inv_t=True)
    expected_scale = torch.exp(torch.tensor(2.0))
    assert torch.allclose(y_div, x / expected_scale, atol=1e-4)
    assert torch.allclose(y_mul, x * expected_scale, atol=1e-4)


def test_observe_mode_updates_log_scale_with_running_max():
    om = _make(dim_size=2)
    om.is_observe = True
    out = om(torch.tensor([[3.0, 5.0]]))
    assert torch.equal(out, torch.tensor([[3.0, 5.0]]))  # passthrough in observe mode
    # log_scale ≈ log(max(|x|)) = log([3, 5])
    assert torch.allclose(om.log_scale.data, torch.log(torch.tensor([[3.0, 5.0]])), atol=1e-4)
    # A larger update raises log_scale; smaller does not.
    om(torch.tensor([[1.0, 6.0]]))
    expected = torch.log(torch.tensor([[3.0, 6.0]]))
    assert torch.allclose(om.log_scale.data, expected, atol=1e-4)


def test_get_scale_clamps_to_finite_range():
    om = _make(dim_size=2)
    with torch.no_grad():
        om.log_scale.fill_(50.0)  # exp(50) >> 1e4 -> should be clamped
    scale = om._get_scale(dtype=torch.float32, device=torch.device("cpu"))
    assert scale.max().item() == pytest.approx(1e4)


def test_get_scale_clamps_low_floor():
    om = _make(dim_size=2)
    with torch.no_grad():
        om.log_scale.fill_(-50.0)  # exp(-50) << 1e-4 -> clamped up
    scale = om._get_scale(dtype=torch.float32, device=torch.device("cpu"))
    assert scale.min().item() == pytest.approx(1e-4)


def test_export_load_round_trip():
    om = _make(dim_size=4)
    with torch.no_grad():
        om.log_scale.copy_(torch.tensor([[0.1, 0.2, 0.3, 0.4]]))
    params = om.export_ptq_params()
    assert "log_scale" in params

    other = _make(dim_size=4)
    other.load_ptq_params(params)
    assert torch.equal(other.log_scale.data, om.log_scale.data)


def test_load_ptq_params_is_no_op_when_log_scale_missing():
    om = _make(dim_size=4)
    original = om.log_scale.data.clone()
    om.load_ptq_params({"unrelated": torch.zeros(1)})
    assert torch.equal(om.log_scale.data, original)


def test_trainable_params_returns_log_scale():
    om = _make(dim_size=4)
    params = om.trainable_params()
    assert any(p is om.log_scale for p in params)


def test_forward_preserves_input_dtype():
    om = _make(dim_size=4)
    x = torch.ones(1, 4, dtype=torch.bfloat16)
    assert om(x).dtype == torch.bfloat16


def test_transform_is_a_no_op():
    assert _make().transform() is None
