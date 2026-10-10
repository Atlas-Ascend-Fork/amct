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

from amct_pytorch.quantization.dtypes.int import QuantDequantInt
from amct_pytorch.quantization.dtypes.int_impl import (
    dynamic_per_token_quant,
    pack_4bit,
    weight_quant,
)

pytestmark = pytest.mark.cpu


@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("value", [0.0, 6e-8, 1e-6])
def test_fp16_weight_fake_quant_matches_float32_reference(bits, value):
    weight = torch.full((2, 4), value, dtype=torch.float16)
    result = QuantDequantInt(bits=bits)(weight)
    reference = QuantDequantInt(bits=bits)(weight.float()).half()
    assert result.dtype == weight.dtype
    assert result.shape == weight.shape
    assert torch.isfinite(result).all()
    torch.testing.assert_close(result, reference, rtol=0, atol=0)


@pytest.mark.parametrize("value", [0.0, 6e-8, 1e-6])
def test_fp16_weight_ste_has_finite_unit_gradient(value):
    weight = torch.full((2, 4), value, dtype=torch.float16, requires_grad=True)
    QuantDequantInt(bits=8)(weight).sum().backward()
    torch.testing.assert_close(weight.grad, torch.ones_like(weight), rtol=0, atol=0)


@pytest.mark.parametrize("value", [0.0, 6e-8, 1e-6])
def test_fp16_int8_payload_matches_float32_reference(value):
    weight = torch.full((2, 4), value, dtype=torch.float16)
    quant = QuantDequantInt(bits=8)
    actual, expected = quant.export_deploy(weight), quant.export_deploy(weight.float())
    assert actual.keys() == expected.keys()
    assert actual["qweight"].dtype == torch.int8
    assert actual["weight_scale"].dtype == torch.float32
    assert actual["weight_scale"].shape == (2, 1)
    assert (actual["weight_scale"] > 0).all()
    assert actual["weight_bias"] is None
    for name in ["qweight", "weight_scale"]:
        torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)


def test_fp16_mixed_rows_do_not_contaminate_valid_rows():
    weight = torch.tensor(
        [[0.0, 0.0, 0.0, 0.0], [1e-6, -1e-6, 0.0, 6e-8], [65504.0, -65504.0, 1.0, 0.0]],
        dtype=torch.float16,
    )
    result = weight_quant(weight)
    assert torch.isfinite(result).all()
    torch.testing.assert_close(
        result, weight_quant(weight.float()).half(), rtol=0, atol=0
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32, torch.float64])
def test_non_fp16_weight_numerics_remain_unchanged(dtype):
    weight = torch.tensor([[0.0, 0.0, 0.0, 0.0], [1.0, -0.5, 0.25, 0.0]], dtype=dtype)
    scale = (weight.abs().max(dim=1, keepdim=True)[0] / 127).clamp_min(1e-9)
    expected = torch.clamp(torch.round(weight / scale), -127, 127) * scale
    torch.testing.assert_close(weight_quant(weight), expected, rtol=0, atol=0)


def test_int16_and_observation_keep_original_object():
    weight = torch.zeros((2, 4), dtype=torch.float16)
    assert QuantDequantInt(bits=16)(weight) is weight
    observed = QuantDequantInt(bits=8)
    observed.is_observe = True
    assert observed(weight) is weight


def test_rounding_offset_and_int4_packing_payload():
    weight = torch.tensor(
        [[0.1, -0.2, 0.3, -0.4], [1.0, -0.5, 0.25, 0.0]], dtype=torch.float16
    )
    offset = torch.full(weight.shape, 0.25, dtype=torch.float32)
    actual, reference = QuantDequantInt(bits=4), QuantDequantInt(bits=4)
    torch.testing.assert_close(
        actual(weight, offset), reference(weight.float(), offset).half(), rtol=0, atol=0
    )
    payload, expected = (
        actual.export_deploy(weight, offset),
        reference.export_deploy(weight.float(), offset),
    )
    for name in ["qweight", "weight_scale", "weight_bias"]:
        torch.testing.assert_close(payload[name], expected[name], rtol=0, atol=0)
    assert payload["qweight"].shape == (1, 4)
    assert payload["weight_scale"].dtype == torch.uint64


def test_activation_path_and_packing_stay_unchanged():
    weight = torch.tensor(
        [[0.0, 0.0, 0.0, 0.0], [1.0, -0.5, 0.25, 0.0]], dtype=torch.float16
    )
    torch.testing.assert_close(
        QuantDequantInt(bits=8, is_act=True)(weight),
        dynamic_per_token_quant(weight, 8),
        rtol=0,
        atol=0,
    )
    packed = pack_4bit(torch.tensor([[1, 2], [-1, -2]], dtype=torch.int8))
    torch.testing.assert_close(packed, torch.tensor([[-15, -30]], dtype=torch.int8))
