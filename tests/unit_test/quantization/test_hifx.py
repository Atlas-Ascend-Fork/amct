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
import pytest
import torch

from amct_pytorch.quantization.dtypes.hifx import QuantDequantHifp


def test_hifp_init_defaults():
    qdq = QuantDequantHifp()
    assert qdq.bits == 8
    assert qdq.is_act is False


def test_hifp_init_explicit_bits():
    qdq = QuantDequantHifp(bits=8, is_act=True)
    assert qdq.bits == 8
    assert qdq.is_act is True


def test_hifp_forward_bits16_is_passthrough():
    qdq = QuantDequantHifp(bits=16)
    x = torch.tensor([1.0, 2.0])
    y = qdq(x)
    assert torch.equal(y, x)


def test_hifp_fake_quant_float32_round_trip():
    qdq = QuantDequantHifp(bits=8)
    x = torch.randn(4, 8, dtype=torch.float32)
    dx = qdq.fake_quant(x)
    assert dx.shape == x.shape
    assert dx.dtype == torch.float32


def test_hifp_fake_quant_float16_round_trip():
    qdq = QuantDequantHifp(bits=8)
    x = torch.randn(4, 8, dtype=torch.float16)
    dx = qdq.fake_quant(x)
    assert dx.shape == x.shape
    assert dx.dtype == torch.float16


def test_hifp_fake_quant_bfloat16_round_trip():
    qdq = QuantDequantHifp(bits=8)
    x = torch.randn(4, 8, dtype=torch.bfloat16)
    dx = qdq.fake_quant(x)
    assert dx.shape == x.shape
    assert dx.dtype == torch.bfloat16


def test_hifp_fake_quant_raises_for_unknown_dtype():
    qdq = QuantDequantHifp(bits=8)
    x = torch.randn(4, 8, dtype=torch.float64)
    with pytest.raises(TypeError, match="Unknown dtype"):
        qdq.fake_quant(x)


def test_hifp_fake_quant_raises_for_unsupported_bits():
    qdq = QuantDequantHifp(bits=4)
    x = torch.randn(4, 8, dtype=torch.float32)
    with pytest.raises(ValueError, match="comming soon"):
        qdq.fake_quant(x)


def test_hifp_forward_bits8_calls_fake_quant(monkeypatch):
    from amct_pytorch.quantization.dtypes.hifx import (
        QuantDequantHifp as quant_dequant_hifp,
    )

    def quantize_level1(self, x):
        return x
    quant_dequant_hifp.quantize_level1 = quantize_level1

    def quantize_level2(self, x):
        return x
    quant_dequant_hifp.quantize_level2 = quantize_level2

    def quantize_level3(self, x):
        return x
    quant_dequant_hifp.quantize_level3 = quantize_level3
    try:
        qdq = quant_dequant_hifp(bits=8)
        x = torch.randn(2, 4, dtype=torch.float32)
        y = qdq(x)
        assert y.shape == x.shape
        assert y.dtype == x.dtype
    finally:
        del quant_dequant_hifp.quantize_level1
        del quant_dequant_hifp.quantize_level2
        del quant_dequant_hifp.quantize_level3


def test_hifp_export_deploy_returns_qweight():
    qdq = QuantDequantHifp(bits=8)
    x = torch.randn(4, 8, dtype=torch.float32)
    result = qdq.export_deploy(x)
    assert "qweight" in result
    assert result["qweight"].shape == x.shape


def test_hifp_registered_in_dtype_registry():
    from amct_pytorch.quantization.dtypes import DTYPE_REGISTRY
    cls = DTYPE_REGISTRY.get("hifp")
    assert cls is QuantDequantHifp



