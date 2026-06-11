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
import math
import sys
import types
from unittest.mock import patch

import pytest
import torch

from amct_pytorch.common.utils import quant_util as qu
from amct_pytorch.common.utils.vars import (
    FLOAT4_E2M1,
    INT4,
    INT4_MAX,
    INT4_MIN,
    INT8,
    INT8_MAX,
    INT8_MIN,
    MXFP4_E2M1,
)

# ---- pad_zero_by_group / convert_to_per_group_shape -----------------------


def test_pad_zero_by_group_pads_to_multiple():
    t = torch.arange(10, dtype=torch.float32).reshape(2, 5)
    out = qu.pad_zero_by_group(t, group_size=4)
    assert out.shape == (2, 8)
    assert torch.equal(out[:, :5], t)
    assert torch.equal(out[:, 5:], torch.zeros(2, 3))


def test_pad_zero_by_group_no_op_when_already_aligned():
    t = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    out = qu.pad_zero_by_group(t, group_size=4)
    assert out.shape == t.shape
    assert torch.equal(out, t)


def test_convert_to_per_group_shape_aligned():
    t = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    out = qu.convert_to_per_group_shape(t, group_size=2)
    assert out.shape == (2, 2, 2)
    # First row first group should be [0, 1].
    assert torch.equal(out[0, 0], torch.tensor([0.0, 1.0]))


def test_convert_to_per_group_shape_with_padding():
    t = torch.arange(10, dtype=torch.float32).reshape(2, 5)
    out = qu.convert_to_per_group_shape(t, group_size=4)
    assert out.shape == (2, 2, 4)
    # Trailing values should be padded zeros.
    assert torch.equal(out[0, 1, 1:], torch.zeros(3))


# ---- cal_shared_exponent / scale_input_by_shared_exponents ----------------


def test_cal_shared_exponent_handles_zero_block():
    t = torch.zeros(1, 32)
    se = qu.cal_shared_exponent(t, block_size=32)
    assert se.shape == (1, 1)
    assert torch.equal(se, torch.zeros(1, 1))


def test_cal_shared_exponent_marks_inf_as_nan():
    t = torch.full((1, 32), float("inf"))
    se = qu.cal_shared_exponent(t, block_size=32)
    assert torch.isnan(se).all()


def test_cal_shared_exponent_value_for_known_block():
    # Block has max abs = 1.0 -> exponent floor(log2(1)) = 0; mantissa = 1.0 (≤ 1.75)
    # so shared = 0 - FP4E2M1_MAX_EXP(2) = -2.
    t = torch.zeros(1, 32)
    t[0, 0] = 1.0
    se = qu.cal_shared_exponent(t, block_size=32)
    assert se.item() == pytest.approx(-2.0)


def test_cal_shared_exponent_pads_uneven_last_dim():
    t = torch.randn(1, 50)
    se = qu.cal_shared_exponent(t, block_size=32)
    # ceil(50/32) = 2 blocks
    assert se.shape == (1, 2)


def test_scale_input_by_shared_exponents_round_trip_shape():
    t = torch.randn(2, 64)
    se = qu.cal_shared_exponent(t, block_size=32)
    scaled = qu.scale_input_by_shared_exponents(t, -se, block_size=32)
    assert scaled.shape == t.shape


def test_scale_input_by_shared_exponents_truncates_to_input_length():
    # Last dim 50, block=32 → 2 exponents; result must keep last-dim=50.
    t = torch.randn(1, 50)
    se = torch.zeros(1, 2)
    scaled = qu.scale_input_by_shared_exponents(t, se, block_size=32)
    assert scaled.shape == (1, 50)
    # Exponent 0 -> multiplier 1.
    assert torch.allclose(scaled, t)


# ---- convert_dtype --------------------------------------------------------


def test_convert_dtype_int8_clamps_and_rounds():
    t = torch.tensor([-200.0, -1.6, 0.4, 1.5, 200.0])
    out = qu.convert_dtype(t, INT8)
    assert out.dtype == torch.int8
    assert out.tolist() == [INT8_MIN, -2, 0, 2, INT8_MAX]


def test_convert_dtype_int4_clamps_and_rounds():
    t = torch.tensor([-100.0, -1.0, 0.0, 1.0, 100.0])
    out = qu.convert_dtype(t, INT4)
    assert out.dtype == torch.int32
    assert out.tolist() == [INT4_MIN, -1, 0, 1, INT4_MAX]


def test_convert_dtype_unknown_raises():
    with pytest.raises(ValueError, match="Not supported quant_dtype"):
        qu.convert_dtype(torch.zeros(2), "no-such-dtype")


# ---- apply_smooth_weight --------------------------------------------------


def test_apply_smooth_weight_scales_along_input_channel():
    w = torch.ones(3, 4)
    factor = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    out = qu.apply_smooth_weight(factor, w)
    assert out.shape == w.shape
    assert torch.equal(out, factor.expand_as(w))


def test_apply_smooth_weight_rejects_wrong_shape():
    w = torch.ones(3, 4)
    factor = torch.ones(1, 5)
    with pytest.raises(RuntimeError, match="smooth_factor shape should"):
        qu.apply_smooth_weight(factor, w)


# ---- check_scale_offset_shape ---------------------------------------------


def test_check_scale_offset_per_tensor_ok():
    qu.check_scale_offset_shape(torch.ones(8, 16), torch.ones(1))


def test_check_scale_offset_per_channel_ok():
    qu.check_scale_offset_shape(torch.ones(8, 16), torch.ones(8))


def test_check_scale_offset_invalid_per_channel_shape():
    with pytest.raises(RuntimeError, match="scale.shape should be equal to 1 or cout"):
        qu.check_scale_offset_shape(torch.ones(8, 16), torch.ones(3))


def test_check_scale_offset_per_group_ok():
    weight = torch.ones(8, 17)
    scale = torch.ones(8, math.ceil(17 / 4), 1)
    qu.check_scale_offset_shape(weight, scale, group_size=4)


def test_check_scale_offset_per_group_bad_shape():
    weight = torch.ones(8, 16)
    scale = torch.ones(8, 3, 1)  # ceil(16/4)=4, but we pass 3
    with pytest.raises(RuntimeError, match="scale.shape should be"):
        qu.check_scale_offset_shape(weight, scale, group_size=4)


def test_check_scale_offset_offset_must_match_scale_shape():
    # Offset-vs-scale shape check only fires on the per-group branch.
    weight = torch.ones(4, 8)
    scale = torch.ones(4, math.ceil(8 / 4), 1)
    offset = torch.ones(4, 1, 1)
    with pytest.raises(RuntimeError, match="offset_w.shape should be equal"):
        qu.check_scale_offset_shape(weight, scale, offset_w=offset, group_size=4)


# ---- apply_awq_quantize_weight --------------------------------------------


def test_apply_awq_quantize_weight_divides_by_scale():
    w = torch.tensor([[2.0, 4.0, 6.0]])
    awq = torch.tensor([[2.0, 2.0, 2.0]])
    out = qu.apply_awq_quantize_weight(w, awq, group_size=None)
    assert torch.equal(out, torch.tensor([[1.0, 2.0, 3.0]]))


def test_apply_awq_quantize_weight_rejects_wrong_shape():
    w = torch.ones(3, 4)
    bad_awq = torch.ones(1, 3)
    with pytest.raises(RuntimeError, match="AWQ params scale.shape should be"):
        qu.apply_awq_quantize_weight(w, bad_awq, group_size=None)


# ---- quant_tensor / quant_dequant_tensor (CPU-safe int paths) -------------


def test_quant_tensor_int8_per_channel():
    w = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    scale = torch.tensor([[2.0]])
    q, shared = qu.quant_tensor(w, INT8, scale=scale)
    assert q.dtype == torch.int8
    assert q.tolist() == [[0, 1, 2, 2]]
    assert shared is None


def test_quant_dequant_tensor_int8_round_trip_close_to_input():
    w = torch.linspace(-1.0, 1.0, steps=8).reshape(1, 8)
    scale = torch.tensor([[w.abs().max() / INT8_MAX]])
    out = qu.quant_dequant_tensor(w, INT8, scale=scale)
    assert out.shape == w.shape
    assert out.dtype == w.dtype
    assert (out - w).abs().max() < scale.item() * 1.5


def test_quant_dequant_tensor_int4_round_trip_close_to_input():
    w = torch.linspace(-1.0, 1.0, steps=8).reshape(1, 8)
    scale = torch.tensor([[w.abs().max() / INT4_MAX]])
    out = qu.quant_dequant_tensor(w, INT4, scale=scale)
    assert out.shape == w.shape
    # int4 is much coarser, but mean error should still be bounded.
    assert (out - w).abs().mean() < scale.item() * 1.5


def test_quant_weight_per_channel_int8_round_trips_through_quant_tensor():
    # quant_weight without group_size transposes, calls quant_tensor with per-row scale,
    # then transposes back. Output should be int8 with same shape.
    w = torch.randn(4, 8)
    scale = w.abs().amax(dim=1, keepdim=True) / INT8_MAX
    q = qu.quant_weight(w, INT8, scale=scale)
    assert q.shape == w.shape
    assert q.dtype == torch.int8


def test_quant_weight_with_group_size_int4():
    w = torch.randn(4, 16)
    group_size = 8
    scale = torch.randn(4, 16 // group_size, 1).abs() + 0.1
    q = qu.quant_weight(w, INT4, scale=scale, group_size=group_size)
    assert q.shape == w.shape
    assert q.dtype == torch.int32


def test_quant_tensor_mxfp4_e2m1():
    t = torch.randn(2, 64)
    shared = qu.cal_shared_exponent(t, block_size=32)
    q, se = qu.quant_tensor(t, MXFP4_E2M1, scale=shared)
    assert q.dtype == torch.float32
    assert se.shape == (2, 2)


def test_quant_dequant_tensor_mxfp4_e2m1_round_trip_shape():
    t = torch.randn(2, 64)
    out = qu.quant_dequant_tensor(t, MXFP4_E2M1)
    assert out.shape == t.shape
    assert out.dtype == t.dtype


def test_quant_tensor_int8_per_group():
    t = torch.randn(2, 16)
    group_size = 8
    scale = torch.randn(2, 16 // group_size, 1) + 0.1
    q, shared = qu.quant_tensor(t, INT8, scale=scale, group_size=group_size)
    assert q.dtype == torch.int8
    assert shared is None


def test_quant_dequant_tensor_int8_per_group_round_trip():
    t = torch.randn(2, 16)
    group_size = 8
    scale = torch.randn(2, 16 // group_size, 1).abs() + 0.1
    out = qu.quant_dequant_tensor(t, INT8, scale=scale, group_size=group_size)
    assert out.shape == t.shape
    assert out.dtype == t.dtype


def test_quant_tensor_int4_per_group():
    t = torch.randn(2, 16)
    group_size = 8
    scale = torch.randn(2, 16 // group_size, 1) + 0.1
    q, shared = qu.quant_tensor(t, INT4, scale=scale, group_size=group_size)
    assert q.dtype == torch.int32
    assert shared is None


def test_quant_dequant_tensor_float4_e2m1():
    t = torch.randn(2, 32)
    max_val = t.abs().amax(dim=-1, keepdim=True)
    scale = max_val / 6.0
    out = qu.quant_dequant_tensor(t, FLOAT4_E2M1, scale=scale)
    assert out.shape == t.shape
    assert out.dtype == t.dtype


def test_quant_weight_no_group_size_with_offset():
    w = torch.randn(4, 8).abs() + 0.1
    scale = w.abs().amax(dim=1, keepdim=True) / INT8_MAX
    offset = torch.randn(4, 1)
    q = qu.quant_weight(w, INT8, scale=scale, offset=offset, group_size=None)
    assert q.shape == w.shape
    assert q.dtype == torch.int8


# ---- NPU-dependent paths covered via mock -----------------------------------


def _make_fake_torch_npu():
    mod = types.ModuleType("torch_npu")
    mod.hifloat8 = "hifloat8_enum"
    mod.npu_quantize = lambda t, s, z, dtype: t.to(torch.float32)
    mod.npu_dynamic_mx_quant = lambda t, axis, round_mode, dst_type, block_size: (
        t.to(torch.float32),
        torch.zeros(t.shape[0], (t.shape[1] + block_size - 1) // block_size),
    )
    mod.npu_dtype_cast = lambda t, dtype, input_dtype=None: t
    return mod


def test_quant_tensor_hifloat8_uses_npu_quantize():
    from amct_pytorch.common.utils.vars import HIFLOAT8

    fake_npu = _make_fake_torch_npu()
    with patch.dict(sys.modules, {"torch_npu": fake_npu}):
        t = torch.randn(2, 4)
        scale = torch.ones(2, 1)
        saved = getattr(torch.Tensor, "npu", None)
        torch.Tensor.npu = lambda self: self
        try:
            q, shared = qu.quant_tensor(t, HIFLOAT8, scale=scale)
        finally:
            if saved is not None:
                torch.Tensor.npu = saved
            elif hasattr(torch.Tensor, "npu"):
                delattr(torch.Tensor, "npu")
    assert q.shape == t.shape
    assert shared is None


def test_quant_tensor_float8_e4m3fn_uses_npu_quantize():
    from amct_pytorch.common.utils.vars import FLOAT8_E4M3FN

    fake_npu = _make_fake_torch_npu()
    with patch.dict(sys.modules, {"torch_npu": fake_npu}):
        t = torch.randn(2, 4)
        scale = torch.ones(2, 1)
        saved = getattr(torch.Tensor, "npu", None)
        torch.Tensor.npu = lambda self: self
        try:
            q, shared = qu.quant_tensor(t, FLOAT8_E4M3FN, scale=scale)
        finally:
            if saved is not None:
                torch.Tensor.npu = saved
            elif hasattr(torch.Tensor, "npu"):
                delattr(torch.Tensor, "npu")
    assert q.shape == t.shape
    assert shared is None


def test_quant_tensor_mxfp8_uses_npu_dynamic_mx_quant():
    from amct_pytorch.common.utils.vars import MXFP8_E4M3FN

    fake_npu = _make_fake_torch_npu()
    with patch.dict(sys.modules, {"torch_npu": fake_npu}):
        t = torch.randn(2, 64)
        saved = getattr(torch.Tensor, "npu", None)
        torch.Tensor.npu = lambda self: self
        try:
            q, shared = qu.quant_tensor(t, MXFP8_E4M3FN)
        finally:
            if saved is not None:
                torch.Tensor.npu = saved
            elif hasattr(torch.Tensor, "npu"):
                delattr(torch.Tensor, "npu")
    assert shared is not None


def test_quant_dequant_tensor_mxfp8_round_trip_shape():
    from amct_pytorch.common.utils.vars import MXFP8_E4M3FN

    fake_npu = _make_fake_torch_npu()
    with patch.dict(sys.modules, {"torch_npu": fake_npu}):
        t = torch.randn(2, 64)
        saved = getattr(torch.Tensor, "npu", None)
        torch.Tensor.npu = lambda self: self
        try:
            out = qu.quant_dequant_tensor(t, MXFP8_E4M3FN)
        finally:
            if saved is not None:
                torch.Tensor.npu = saved
            elif hasattr(torch.Tensor, "npu"):
                delattr(torch.Tensor, "npu")
    assert out.shape == t.shape
