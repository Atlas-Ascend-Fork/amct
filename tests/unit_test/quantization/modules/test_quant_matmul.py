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

from amct_pytorch.quantization.dtypes import register_dtype
from amct_pytorch.quantization.modules.quant_matmul import QuantizedMatmul

register_dtype()


def _args(algos=(), quant_target=(), quant_dtype="int", w_bits=8):
    return SimpleNamespace(
        algos=list(algos),
        quant_dtype=quant_dtype,
        w_bits=w_bits,
        quant_target=list(quant_target),
    )


def test_init_default_disabled_attn_cache_skips_transform_setup():
    qm = QuantizedMatmul(_args())
    assert qm.enable_attn_cache is False
    assert qm.left_transform is None
    assert qm.right_transform is None
    assert qm.eval_mode is False


def test_forward_passthrough_when_attn_cache_disabled_is_l_at_r_t():
    qm = QuantizedMatmul(_args())
    left = torch.randn(2, 3, 4)
    right = torch.randn(2, 5, 4)  # shape allows transpose to (4, 5)
    out = qm(left, right)
    expected = torch.matmul(left, right.transpose(-2, -1))
    assert torch.allclose(out, expected, atol=1e-6)


def test_forward_can_skip_right_transpose_for_value_matmul():
    qm = QuantizedMatmul(_args(), transpose_right=False)
    left = torch.randn(2, 3, 5)
    right = torch.randn(2, 5, 4)
    out = qm(left, right)
    expected = torch.matmul(left, right)
    assert torch.allclose(out, expected, atol=1e-6)


def test_forward_quantizes_when_attn_cache_enabled_and_quantizers_active():
    qm = QuantizedMatmul(_args(quant_target=["attn-cache"]))
    qm.l_node.enable = True
    qm.r_node.enable = True
    left = torch.randn(1, 2, 4)
    right = torch.randn(1, 3, 4)
    out = qm(left, right)
    expected_shape = (1, 2, 3)
    assert out.shape == expected_shape


def test_forward_runs_transforms_when_set():
    qm = QuantizedMatmul(_args(quant_target=["attn-cache"]))
    seen = []

    def left_t(x):
        seen.append("L")
        return x

    def right_t(x):
        seen.append("R")
        return x
    qm.left_transform = left_t
    qm.right_transform = right_t
    left = torch.randn(1, 2, 4)
    right = torch.randn(1, 3, 4)
    qm(left, right)
    assert seen == ["L", "R"]


def test_init_constructs_per_side_quantizers_with_specified_bits():
    qm = QuantizedMatmul(_args(), l_bits=4, r_bits=8)
    assert qm.l_node.bits == 4
    assert qm.r_node.bits == 8
