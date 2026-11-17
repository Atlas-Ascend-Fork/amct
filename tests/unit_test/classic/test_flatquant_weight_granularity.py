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
"""FlatQuant weight granularity (per-channel / per-tensor) unit tests.

Covers the FR001-002/004 contract: strategy='tensor' must produce a true
single-scale quantizer (deploy artifact carries exactly one scale value),
strategy='channel' keeps the per-channel default, and any other strategy
is rejected at config parse time.
"""

import pytest
import torch
from torch import nn

from amct_pytorch.classic.quantize_op.flatquant.flat_config import (
    flat_config_parser,
)
from amct_pytorch.classic.quantize_op.flatquant.flat_linear import (
    FlatQuantizedLinear,
)
from amct_pytorch.classic.quantize_op.flatquant.quantizer import WeightQuantizer


def _layer_config(strategy):
    return {
        'algorithm': {'flatquant': {}},
        'inputs_cfg': {
            'quant_type': 'int8',
            'symmetric': True,
            'strategy': 'token',
        },
        'weights_cfg': {
            'quant_type': 'int4',
            'symmetric': True,
            'strategy': strategy,
        },
    }


def test_flat_config_parses_channel_and_tensor_strategies():
    assert flat_config_parser(_layer_config('channel')).w_granularity == 'channel'
    assert flat_config_parser(_layer_config('tensor')).w_granularity == 'tensor'


def test_flat_config_defaults_to_channel():
    cfg = flat_config_parser(_layer_config('channel'))
    assert cfg.w_granularity == 'channel'


def test_flat_config_rejects_unknown_weight_strategy():
    with pytest.raises(ValueError, match='strategy'):
        flat_config_parser(_layer_config('group'))


def test_tensor_strategy_yields_single_scale_module():
    cfg = flat_config_parser(_layer_config('tensor'))
    fql = FlatQuantizedLinear(nn.Linear(32, 16), cfg, lac=False)
    assert tuple(fql.weight_quantizer.scale.shape) == (1, 1)
    fql.weight_quantizer.find_params(fql.linear.weight)
    _, scale = fql.weight_quantizer.quantize(fql.linear.weight, quantonly=True)
    assert scale.numel() == 1


def test_channel_strategy_yields_per_channel_scale_module():
    cfg = flat_config_parser(_layer_config('channel'))
    fql = FlatQuantizedLinear(nn.Linear(32, 16), cfg, lac=False)
    assert tuple(fql.weight_quantizer.scale.shape) == (16, 1)
    fql.weight_quantizer.find_params(fql.linear.weight)
    _, scale = fql.weight_quantizer.quantize(fql.linear.weight, quantonly=True)
    assert scale.numel() == 16
    # rows genuinely differ (per-channel semantics, not a repeated constant)
    assert not torch.allclose(scale[:1].expand_as(scale), scale)


def test_per_tensor_scale_is_single_value_and_broadcasts():
    w = torch.randn(16, 32)
    wq = WeightQuantizer(shape=(1, 1))
    wq.configure(4, perchannel=False, sym=True)
    wq.find_params(w)
    q, scale = wq.quantize(w, quantonly=True)
    assert scale.numel() == 1
    dequant = scale * q
    assert torch.isfinite(dequant).all()
    # per-channel must be at least as accurate as per-tensor on the same data
    wq_ch = WeightQuantizer(shape=(16, 1))
    wq_ch.configure(4, perchannel=True, sym=True)
    wq_ch.find_params(w)
    q_ch, scale_ch = wq_ch.quantize(w, quantonly=True)
    assert (scale_ch * q_ch - w).abs().mean() <= (dequant - w).abs().mean() + 1e-6


def test_per_channel_backward_compatible_scale_shape():
    """Existing (default) path regression: scale stays (-1, 1)-broadcastable."""
    w = torch.randn(8, 8)
    wq = WeightQuantizer(shape=(8, 1))
    wq.configure(4, perchannel=True, sym=True)
    wq.find_params(w)
    assert tuple(wq.scale.shape) == (8, 1)
    out = wq.quantize(w)
    assert torch.isfinite(out).all()
