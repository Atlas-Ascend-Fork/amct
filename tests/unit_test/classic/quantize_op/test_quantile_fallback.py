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
import torch

from amct_pytorch.common.utils import quant_util as qu


def _make_quantile_dynamic_act():
    """构造一个只跑 fake_quant_forward dynamic 分支的最小 QuantileQuant 替身。"""
    from amct_pytorch.classic.quantize_op import quantile_module as qm

    inst = qm.QuantileQuant.__new__(qm.QuantileQuant)
    inst.weight_compress_only = False
    inst.dynamic = True
    inst.bias = None
    inst.cached_dq_w = torch.eye(8)
    inst.fake_quant_cache_ready = True
    return inst


def test_quantile_dynamic_fallback_pertoken_scale(monkeypatch):
    qu._hifloat8_supported.cache_clear()
    monkeypatch.setattr(qu, "_hifloat8_supported", lambda: False)

    captured = {}

    def fake_fake_quant(t):
        captured["scaled"] = t.clone()
        return t  # 透传

    monkeypatch.setattr(qu, "_hifloat8_fake_quant", fake_fake_quant)

    inst = _make_quantile_dynamic_act()
    # weight 为单位阵，bias=None：F.linear(dq_x, I) == dq_x
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]])
    out = inst.fake_quant_forward(x)

    amax = x.abs().amax(dim=-1, keepdim=True)
    expected_scale = amax / 15
    # 透传 fake_quant：dq_x = (x/scale)*scale = x
    assert torch.allclose(captured["scaled"], x / expected_scale)
    assert torch.allclose(out, x, atol=1e-4)
