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

from amct_pytorch.common.models.llm.deepseek.deepseek_v3_2.deepseekv3_2 import (
    DeepseekV32,
)
from amct_pytorch.common.models.llm.deepseek.deepseek_v4.deepseekv4 import DeepseekV4
from amct_pytorch.common.models.llm.glm.glm5.glm5 import GLM5
from amct_pytorch.common.models.llm.glm.glm5_2.glm5_2 import GLM5_2


def _new_v32():
    adapter = DeepseekV32.__new__(DeepseekV32)
    adapter.args = SimpleNamespace(device="cpu")
    adapter.model = SimpleNamespace(
        norm=nn.LayerNorm(4),
        head=nn.Linear(4, 8, bias=False),
    )
    return adapter


def test_v32_do_head_forward_generator():
    adapter = _new_v32()
    outputs = adapter.do_head_forward([torch.randn(1, 4, 4)])

    assert iter(outputs) is outputs
    logits = list(outputs)

    assert logits[0].shape == (1, 3, 8)
    assert logits[0].device.type == "cpu"


def test_glm_head_generator_inheritance():
    assert GLM5.do_head_forward is DeepseekV32.do_head_forward

    adapter = GLM5_2.__new__(GLM5_2)
    adapter.args = SimpleNamespace(device="cpu")
    adapter.model = SimpleNamespace(
        model=SimpleNamespace(norm=nn.LayerNorm(4)),
        lm_head=nn.Linear(4, 8, bias=False),
    )
    outputs = adapter.do_head_forward([torch.randn(1, 4, 4)])

    assert iter(outputs) is outputs
    assert list(outputs)[0].shape == (1, 3, 8)


class _FakeV4Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(8, 4))

    def hc_head(self, inp, fn, scale, base):
        return inp.mean(dim=2) * scale + 2 * base + 3 * fn


def _new_v4(meta=False):
    adapter = DeepseekV4.__new__(DeepseekV4)
    adapter.args = SimpleNamespace(device="cpu")
    param_device = "meta" if meta else "cpu"
    adapter.model = SimpleNamespace(
        head=_FakeV4Head(),
        norm=nn.LayerNorm(4),
        hc_head_fn=nn.Parameter(torch.full((1,), 0.25, device=param_device)),
        hc_head_scale=nn.Parameter(torch.full((1,), 1.5, device=param_device)),
        hc_head_base=nn.Parameter(torch.full((1,), -0.75, device=param_device)),
    )
    return adapter


def test_v4_do_head_forward_preserves_hc_head_math():
    adapter = _new_v4()
    inp = torch.randn(1, 4, 2, 4)

    outputs = adapter.do_head_forward([inp])
    assert iter(outputs) is outputs
    logits = list(outputs)

    x = adapter.model.head.hc_head(
        inp,
        adapter.model.hc_head_fn,
        adapter.model.hc_head_scale,
        adapter.model.hc_head_base,
    )
    expected = F.linear(adapter.model.norm(x).float(), adapter.model.head.weight)
    assert torch.allclose(logits[0], expected[:, :-1, :].contiguous())
    assert logits[0].device.type == "cpu"


def test_v4_do_head_forward_rejects_meta_top_level_param():
    adapter = _new_v4(meta=True)
    with pytest.raises(RuntimeError, match="still on meta"):
        next(adapter.do_head_forward([torch.randn(1, 4, 2, 4)]))
