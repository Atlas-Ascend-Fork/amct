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
from torch import nn

from amct_pytorch.quantization.dtypes import register_dtype
from amct_pytorch.quantization.modules.quant_linear import QuantLinear

register_dtype()
pytestmark = pytest.mark.cpu


def make_quant_linear():
    args = SimpleNamespace(algos=[], quant_dtype="int", w_bits=8, quant_target=[])
    linear = nn.Linear(4, 2)
    with torch.no_grad():
        linear.weight.fill_(1.0)
        linear.bias.fill_(0.25)
    layer = QuantLinear(args, linear)
    layer.eval_mode = True
    return layer


@pytest.mark.parametrize("dtype", [torch.float64, torch.float16, torch.bfloat16])
def test_conversion_after_cached_forward_matches_fresh_quantization(dtype):
    layer = make_quant_linear()
    layer(torch.ones(2, 4))
    layer.to(dtype=dtype)
    data = torch.ones(2, 4, dtype=dtype)
    actual = layer(data)
    weight = layer.weight_quantizer(layer.linear.weight).detach()
    expected = torch.nn.functional.linear(data, weight, layer.linear.bias)
    assert layer.cached_eval_weight.dtype == dtype
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_parent_module_conversion_invalidates_nested_cache():
    layer = make_quant_linear()
    model = nn.Sequential(layer)
    model(torch.ones(2, 4))
    model.double()
    torch.testing.assert_close(
        model(torch.ones(2, 4, dtype=torch.float64)),
        torch.full((2, 2), 4.25, dtype=torch.float64),
        rtol=0,
        atol=0,
    )


def test_conversion_resets_transform_key_and_rebuilds_transformed_weight():
    layer = make_quant_linear()

    def transform(weight, inv_t=False, name=None):
        return weight * 2

    layer(torch.ones(2, 4), structure_transform=transform)
    assert layer._cached_transform_key == id(transform)
    layer.double()
    assert layer.cached_eval_weight is None
    assert layer._cached_transform_key is None
    result = layer(torch.ones(2, 4, dtype=torch.float64), structure_transform=transform)
    assert layer._cached_transform_key == id(transform)
    torch.testing.assert_close(result, torch.full_like(result, 8.25), rtol=0, atol=0)


def test_unchanged_module_reuses_eval_cache():
    layer = make_quant_linear()
    data = torch.ones(2, 4)
    first = layer(data)
    cache = layer.cached_eval_weight
    torch.testing.assert_close(layer(data), first, rtol=0, atol=0)
    assert layer.cached_eval_weight is cache


def test_conversion_before_first_forward_preserves_lazy_cache():
    layer = make_quant_linear().double()
    assert layer.cached_eval_weight is None
    result = layer(torch.ones(2, 4, dtype=torch.float64))
    torch.testing.assert_close(result, torch.full_like(result, 4.25), rtol=0, atol=0)
