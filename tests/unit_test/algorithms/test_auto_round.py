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

from amct_pytorch.algorithms.quant.auto_round import (
    AutoRound,
    _get_scale_shape,
    _reshape_pad_tensor_by_group_size,
    _revert_tensor_by_pad,
)


def _auto_round_args(w_size=(4, 8), quant_dtype="int"):
    return SimpleNamespace(w_size=w_size, quant_dtype=quant_dtype)


def test_reshape_pad_tensor_by_group_size_no_pad():
    weight = torch.randn(4, 8)
    reshaped, orig, pad = _reshape_pad_tensor_by_group_size(weight, -1)
    assert reshaped.shape == (4, 8)
    assert pad == 0


def test_reshape_pad_tensor_by_group_size_with_pad():
    weight = torch.randn(2, 10)
    reshaped, orig, pad = _reshape_pad_tensor_by_group_size(weight, 4)
    assert pad == 2
    assert reshaped.shape[-1] == 4


def test_reshape_pad_tensor_by_group_size_raises_on_non_2d():
    with pytest.raises(ValueError, match="2D"):
        _reshape_pad_tensor_by_group_size(torch.randn(2, 3, 4), 4)


def test_revert_tensor_by_pad_no_pad():
    tensor = torch.randn(4, 8)
    reverted = _revert_tensor_by_pad(tensor, (4, 8), 0)
    assert reverted.shape == (4, 8)
    assert torch.equal(reverted, tensor)


def test_revert_tensor_by_pad_with_pad():
    tensor = torch.randn(4, 8)
    reverted = _revert_tensor_by_pad(tensor, (4, 6), 2)
    assert reverted.shape == (4, 6)


def test_revert_tensor_by_pad_raises_on_non_2d():
    with pytest.raises(ValueError, match="2D"):
        _revert_tensor_by_pad(torch.randn(4, 8), (2, 3, 4), 0)


def test_get_scale_shape_per_row():
    assert _get_scale_shape((4, 8), -1) == 4


def test_get_scale_shape_per_tensor():
    assert _get_scale_shape((4, 8), 0) == 1


def test_get_scale_shape_grouped():
    assert _get_scale_shape((4, 32), 16) == 8


def test_auto_round_init_int_dtype():
    args = _auto_round_args(w_size=(4, 8), quant_dtype="int")
    ar = AutoRound(args, w_bits=8)
    assert ar.bits == 8
    assert ar.group_size == -1


def test_auto_round_init_mxfp_dtype():
    args = _auto_round_args(w_size=(4, 32), quant_dtype="mxfp")
    ar = AutoRound(args, w_bits=4)
    assert ar.bits == 4
    assert ar.group_size == 32


def test_auto_round_trainable_params():
    args = _auto_round_args()
    ar = AutoRound(args, w_bits=8)
    params = ar.trainable_params()
    assert len(params) == 3


def test_auto_round_export_ptq_params():
    args = _auto_round_args()
    ar = AutoRound(args, w_bits=8)
    exported = ar.export_ptq_params()
    assert "value" in exported
    assert "min_scale" in exported
    assert "max_scale" in exported


def test_auto_round_forward_passthrough():
    args = _auto_round_args()
    ar = AutoRound(args, w_bits=8)
    weight = torch.randn(4, 8)
    out = ar(weight)
    assert torch.equal(out, weight)


def test_get_group_size_unsupported_raises():
    args = _auto_round_args(w_size=(4, 8), quant_dtype="fp8")
    with pytest.raises(ValueError, match="hifx"):
        AutoRound._get_group_size(args)


def test_auto_round_load_ptq_params():
    args = _auto_round_args(w_size=(4, 32), quant_dtype="int")
    ar = AutoRound(args, w_bits=8)
    value = torch.randn(4, 32)
    min_scale = torch.randn(4)
    max_scale = torch.randn(4)
    ar.load_ptq_params({"value": value, "min_scale": min_scale, "max_scale": max_scale})
    assert torch.equal(ar.value.data, value.to(dtype=ar.value.dtype, device=ar.value.device))


def test_auto_round_compute_clip_range():
    args = _auto_round_args(w_size=(2, 8), quant_dtype="int")
    ar = AutoRound(args, w_bits=8)
    weight = torch.randn(2, 8)
    clip_min, clip_max = ar._compute_clip_range(weight)
    assert clip_min.shape == (2, 1)
    assert clip_max.shape == (2, 1)
    assert (clip_min <= clip_max).all()


def test_auto_round_quantize():
    args = _auto_round_args(w_size=(2, 8), quant_dtype="int")
    ar = AutoRound(args, w_bits=8)

    def mock_quant_obj(weight, v=None):
        return weight

    weight = torch.randn(2, 8)
    result = ar.quantize(weight, mock_quant_obj)
    assert result.shape == weight.shape


def test_auto_round_quantize_with_group_size():
    args = _auto_round_args(w_size=(2, 12), quant_dtype="int")
    ar = AutoRound(args, w_bits=8)

    def mock_quant_obj(weight, v=None):
        return weight

    weight = torch.randn(2, 12)
    result = ar.quantize(weight, mock_quant_obj)
    assert result.shape == weight.shape
