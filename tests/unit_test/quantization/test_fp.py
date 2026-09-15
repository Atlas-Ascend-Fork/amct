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

from amct_pytorch.quantization.dtypes import DTYPE_REGISTRY
from amct_pytorch.quantization.dtypes.fp import BLOCK_SIZE_ROW, QuantDequantFP


def test_fp_is_registered_under_fp_name():
    assert DTYPE_REGISTRY.get("fp") is QuantDequantFP


def test_fp_constructor_defaults():
    qdq = QuantDequantFP(bits=8)
    assert qdq.bits == 8
    assert qdq.is_act is False
    assert qdq.is_observe is False
    assert qdq.block_size_col == 128
    assert qdq.scale_dtype == "fp32"


def test_fp_bits4_requires_e8m0_scale_dtype():
    with pytest.raises(ValueError, match="only 8-bits quantization"):
        QuantDequantFP(bits=4)


def test_fp_bits4_e8m0_is_allowed():
    qdq = QuantDequantFP(bits=4, scale_dtype="e8m0")
    assert qdq.bits == 4
    assert qdq.scale_dtype == "e8m0"


def test_fp_forward_observe_returns_input_unchanged():
    qdq = QuantDequantFP(bits=8)
    qdq.is_observe = True
    x = torch.linspace(-1.0, 1.0, steps=64).reshape(2, 32)
    out = qdq(x)
    assert out is x
    assert torch.equal(out, x)


def test_fp_forward_bits16_is_passthrough():
    qdq = QuantDequantFP(bits=16)
    x = torch.randn(2, 32)
    assert torch.equal(qdq(x), x)


def test_fp_forward_shape_and_dtype_preserved():
    qdq = QuantDequantFP(bits=8)
    x = torch.randn(8, 64, dtype=torch.float32)
    y = qdq(x)
    assert y.shape == x.shape
    assert y.dtype == x.dtype


def test_fp_forward_with_tensor_v():
    qdq = QuantDequantFP(bits=8)
    x = torch.randn(4, 64)
    v = torch.randn(4, 64)
    y = qdq(x, v=v)
    assert y.shape == x.shape


def test_fp_fake_quant_preserves_shape():
    qdq = QuantDequantFP(bits=8)
    x = torch.randn(4, 64)
    y = qdq.fake_quant(x)
    assert y.shape == x.shape


def test_fp_zero_input_round_trips_to_zero():
    qdq = QuantDequantFP(bits=8)
    x = torch.zeros(1, 32)
    y = qdq(x)
    assert torch.allclose(y, x)


def test_fp_quantization_error_bounded():
    qdq = QuantDequantFP(bits=8)
    x = torch.randn(4, 128, dtype=torch.float32)
    y = qdq(x)
    rel_err = (y - x).abs().mean() / x.abs().mean().clamp_min(1e-6)
    assert rel_err.item() < 0.2


def test_fp_quant_returns_scale_and_ex_mx():
    qdq = QuantDequantFP(bits=8)
    x = torch.randn(4, 64)
    scale, ex_mx = qdq.quant(x)
    assert ex_mx.shape == x.shape
    # 4x64 is padded up to a single 128x128 block -> one fp32 block scale.
    assert scale.shape == (1, 1, 1, 1)
    assert scale.dtype == torch.float32


def test_fp_reshape_block_pads_to_block_multiple():
    qdq = QuantDequantFP(bits=8)
    x = torch.randn(4, 64)
    blocks, pad_shape = qdq.reshape_block(x)
    assert pad_shape == (BLOCK_SIZE_ROW, BLOCK_SIZE_ROW)
    assert blocks.shape == (1, 1, BLOCK_SIZE_ROW, BLOCK_SIZE_ROW)


def test_fp_quantize_fp_block_scale_matches_amax_over_max():
    qdq = QuantDequantFP(bits=8)
    x = torch.full((BLOCK_SIZE_ROW, BLOCK_SIZE_ROW), 448.0)
    w_fp8, scale = qdq.quantize_fp_block(x)
    assert torch.allclose(scale, torch.tensor([[[[1.0]]]]))
    assert torch.allclose(w_fp8, x)


def test_fp_quantize_fp_block_preserves_shape():
    qdq = QuantDequantFP(bits=8)
    x = torch.randn(4, 64)
    w_fp8, scale = qdq.quantize_fp_block(x)
    assert w_fp8.shape == x.shape
    assert scale.shape == (1, 1, 1, 1)


def test_fp_dequantize_fp_block_round_trips():
    qdq = QuantDequantFP(bits=8)
    x = torch.full((BLOCK_SIZE_ROW, BLOCK_SIZE_ROW), 448.0)
    w_fp8, scale = qdq.quantize_fp_block(x)
    out = qdq.dequantize_fp_block(w_fp8, scale)
    assert out.shape == x.shape
    assert torch.allclose(out, x)


def test_fp_block_size_col_one_preserves_shape():
    qdq = QuantDequantFP(bits=8, block_size_col=1)
    x = torch.randn(BLOCK_SIZE_ROW, 64)
    y = qdq(x)
    assert y.shape == x.shape


def test_fp_block_size_col_one_pads_rows_to_block_row():
    qdq = QuantDequantFP(bits=8, block_size_col=1)
    x = torch.randn(64, BLOCK_SIZE_ROW)
    scale, _ = qdq.quant(x)
    # block_m=1 -> one column per block, rows padded/tiled by BLOCK_SIZE_ROW.
    assert scale.shape == (1, BLOCK_SIZE_ROW, 1, 1)


def test_fp_deploy_bits8_returns_fp8_and_scale():
    qdq = QuantDequantFP(bits=8)
    x = torch.randn(4, 64)
    ex_mx, scale = qdq.deploy(x)
    assert ex_mx.dtype == torch.float8_e4m3fn
    assert ex_mx.shape == x.shape
    assert scale.dtype == torch.float32
    assert scale.shape == (1, 1)


def test_fp_deploy_bits4_returns_packed_uint8():
    qdq = QuantDequantFP(bits=4, scale_dtype="e8m0")
    x = torch.randn(4, 64)
    ex_mx, scale = qdq.deploy(x)
    assert ex_mx.dtype == torch.uint8
    assert ex_mx.shape == (4, 32)


def test_fp_deploy_scale_rank_uniform_across_scale_dtype():
    x = torch.randn(256, 256)
    _, scale_fp32 = QuantDequantFP(bits=8).deploy(x)
    _, scale_e8m0 = QuantDequantFP(bits=4, scale_dtype="e8m0").deploy(x)
    assert scale_fp32.shape == scale_e8m0.shape == (2, 2)


def test_fp_export_deploy_returns_dict():
    qdq = QuantDequantFP(bits=8)
    x = torch.randn(4, 64)
    out = qdq.export_deploy(x)
    assert set(out) == {"qweight", "weight_scale"}
    assert out["qweight"].dtype == torch.float8_e4m3fn
    assert out["weight_scale"].shape == (1, 1)


def test_fp_deploy_round_trips_via_weight_dequant():
    from amct_pytorch.quantization.dtypes.fp_impl import weight_dequant

    qdq = QuantDequantFP(bits=8)
    x = torch.full((BLOCK_SIZE_ROW, BLOCK_SIZE_ROW), 448.0)
    qweight, scale = qdq.deploy(x)
    deq = weight_dequant(qweight, scale, block_size=BLOCK_SIZE_ROW)
    assert deq.shape == x.shape
    assert torch.allclose(deq, x)
