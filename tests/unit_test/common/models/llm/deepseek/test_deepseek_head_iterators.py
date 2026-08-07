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

from amct_pytorch.common.models.llm.common.base import BaseModel
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


def test_v32_iterator_and_inherited_cpu_wrapper():
    adapter = _new_v32()
    inps = [torch.randn(1, 4, 4)]

    logits = list(adapter.iter_head_logits(inps))
    preds = adapter.do_head_forward(inps)

    assert logits[0].shape == (1, 3, 8)
    assert logits[0].device.type == "cpu"
    assert torch.allclose(preds[0], logits[0])


def test_glm_head_iterator_inheritance():
    assert GLM5.iter_head_logits is DeepseekV32.iter_head_logits
    assert GLM5_2.iter_head_logits is BaseModel.iter_head_logits


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


def test_v4_iterator_preserves_hc_head_math():
    adapter = _new_v4()
    inp = torch.randn(1, 4, 2, 4)

    logits = list(adapter.iter_head_logits([inp]))
    preds = adapter.do_head_forward([inp])

    x = adapter.model.head.hc_head(
        inp,
        adapter.model.hc_head_fn,
        adapter.model.hc_head_scale,
        adapter.model.hc_head_base,
    )
    expected = F.linear(adapter.model.norm(x).float(), adapter.model.head.weight)
    assert torch.allclose(logits[0], expected[:, :-1, :].contiguous())
    assert preds[0].device.type == "cpu"
    assert torch.allclose(preds[0], logits[0])


def test_v4_iterator_rejects_meta_top_level_param():
    adapter = _new_v4(meta=True)
    with pytest.raises(RuntimeError, match="still on meta"):
        next(adapter.iter_head_logits([torch.randn(1, 4, 2, 4)]))
