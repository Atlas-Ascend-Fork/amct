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
"""Reloading a checkpoint must invalidate derived evaluation weights."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from amct_pytorch.quantization.dtypes import register_dtype
from amct_pytorch.quantization.modules.quant_linear import QuantLinear

register_dtype()
pytestmark = pytest.mark.cpu


@pytest.mark.parametrize("assign", [False, True])
@pytest.mark.parametrize("nested", [False, True])
def test_state_reload_rebuilds_eval_cache_from_new_parameters(assign, nested):
    args = SimpleNamespace(algos=[], quant_dtype="int", w_bits=8, quant_target=[])
    layer = QuantLinear(args, nn.Linear(4, 3, bias=False), name="proj")
    layer.eval_mode = True
    with torch.no_grad():
        layer.linear.weight.fill_(1.0)
    model = nn.Sequential(layer) if nested else layer
    inputs = torch.ones(2, 4)
    old_output = model(inputs)
    old_cache = layer.cached_eval_weight
    state = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    weight_key = "0.linear.weight" if nested else "linear.weight"
    state[weight_key].fill_(3.0)
    model.load_state_dict(state, assign=assign)
    assert layer._cached_transform_key is None
    expected = F.linear(inputs, layer.weight_quantizer(layer.linear.weight))
    actual = model(inputs)
    assert not torch.equal(expected, old_output)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert layer.cached_eval_weight is not old_cache
    cached = layer.cached_eval_weight
    torch.testing.assert_close(model(inputs), expected, rtol=0, atol=0)
    assert layer.cached_eval_weight is cached


@pytest.mark.parametrize("assign", [False, True])
@pytest.mark.parametrize("nested", [False, True])
def test_state_reload_clears_populated_transform_key(assign, nested):
    args = SimpleNamespace(algos=[], quant_dtype="int", w_bits=8, quant_target=[])
    layer = QuantLinear(args, nn.Linear(4, 3, bias=False), name="proj")
    layer.eval_mode = True
    with torch.no_grad():
        layer.linear.weight.fill_(1.0)

    def transform(weight, inv_t=False, name=None):
        return weight * 2

    inputs = torch.ones(2, 4)
    layer(inputs, structure_transform=transform)
    assert layer._cached_transform_key == id(transform)
    model = nn.Sequential(layer) if nested else layer
    state = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    weight_key = "0.linear.weight" if nested else "linear.weight"
    state[weight_key].fill_(3.0)

    model.load_state_dict(state, assign=assign)

    assert layer.cached_eval_weight is None
    assert layer._cached_transform_key is None
    actual = layer(inputs, structure_transform=transform)
    expected = F.linear(inputs, layer.weight_quantizer(layer.linear.weight * 2))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert layer._cached_transform_key == id(transform)
