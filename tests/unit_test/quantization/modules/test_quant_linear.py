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
import torch.nn as nn
import torch.nn.functional as F

from amct_pytorch.quantization.dtypes import register_dtype
from amct_pytorch.quantization.modules.quant_linear import QuantLinear

register_dtype()


def _args(algos=(), quant_dtype="int", w_bits=8):
    return SimpleNamespace(
        algos=list(algos),
        quant_dtype=quant_dtype,
        w_bits=w_bits,
        quant_target=[],
    )


def _make_linear_and_quant(in_features=4, out_features=8):
    linear = nn.Linear(in_features, out_features)
    return linear, QuantLinear(_args(), linear, w_bits=8, name="proj")


def test_quant_linear_init_records_weight_size_in_args():
    linear, q = _make_linear_and_quant(4, 8)
    # args.w_size mutated as side effect
    assert tuple(q.args.w_size) == (8, 4)
    assert q.eval_mode is False
    assert q.cached_eval_weight is None
    assert q._cached_transform_key is None


def test_quant_linear_train_forward_disabled_quantizer_matches_plain_linear():
    linear, q = _make_linear_and_quant(4, 8)
    x = torch.randn(2, 4)
    expected = linear(x)
    assert torch.allclose(q(x), expected, atol=1e-6)


def test_quant_linear_train_forward_enabled_quantizer_changes_output():
    linear, q = _make_linear_and_quant(4, 8)
    q.weight_quantizer.enable = True
    # Make weights large so int8 quantization is visible.
    with torch.no_grad():
        linear.weight.copy_(torch.linspace(-10, 10, steps=linear.weight.numel()).reshape_as(linear.weight))
    x = torch.randn(2, 4)
    plain = linear(x)
    quantized = q(x)
    assert quantized.shape == plain.shape
    # Output should not be exactly equal once quantization kicks in.
    assert not torch.allclose(quantized, plain, atol=1e-7)


def test_quant_linear_eval_mode_caches_quantized_weight():
    linear, q = _make_linear_and_quant(4, 8)
    q.weight_quantizer.enable = True
    q.eval_mode = True
    x = torch.randn(2, 4)
    out_first = q(x)
    cached = q.cached_eval_weight
    assert cached is not None
    assert q._cached_transform_key is None  # no structure_transform
    # Second call with the same transform must reuse the cache.
    out_second = q(x)
    assert q.cached_eval_weight is cached
    assert torch.equal(out_first, out_second)


def test_quant_linear_eval_mode_invalidates_cache_when_transform_changes():
    linear, q = _make_linear_and_quant(4, 8)
    q.weight_quantizer.enable = True
    q.eval_mode = True

    def transform(weight, inv_t=False, name=None):
        return weight  # identity-but-distinct callable

    def transform_other(weight, inv_t=False, name=None):
        return weight

    x = torch.randn(2, 4)
    q(x, structure_transform=transform)
    first_cached = q.cached_eval_weight
    q(x, structure_transform=transform_other)
    assert q.cached_eval_weight is not first_cached


def test_quant_linear_uses_structure_transform_with_inv_t_and_name():
    linear, q = _make_linear_and_quant(4, 8)
    captured = {}

    def transform(weight, inv_t=False, name=None):
        captured["inv_t"] = inv_t
        captured["name"] = name
        return weight * 0  # easy to verify path was taken

    out = q(torch.zeros(1, 4), structure_transform=transform)
    assert captured == {"inv_t": True, "name": "proj"}
    # weight zeroed -> output is bias only.
    assert torch.allclose(out, linear.bias.expand_as(out), atol=1e-6)


def test_quant_linear_export_deploy_returns_quant_dtype_payload():
    _, q = _make_linear_and_quant(4, 8)
    out = q.export_deploy()
    assert "qweight" in out
    assert "weight_scale" in out


def test_quant_linear_export_deploy_applies_structure_transform():
    linear, q = _make_linear_and_quant(4, 8)
    seen = {}

    def transform(weight, inv_t=False, name=None):
        seen["inv_t"] = inv_t
        seen["name"] = name
        return weight

    q.export_deploy(structure_transform=transform)
    assert seen == {"inv_t": True, "name": "proj"}
