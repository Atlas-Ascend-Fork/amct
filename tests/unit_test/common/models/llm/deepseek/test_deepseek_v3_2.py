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
from torch import nn

from amct_pytorch.common.models.llm.deepseek.deepseek_v3_2.deepseekv3_2 import (
    DeepseekV32,
)


def _build_fake_v32_model():
    model = nn.Module()
    model.embed = nn.Embedding(8, 4)
    model.norm = nn.LayerNorm(4)
    model.head = nn.Linear(4, 8, bias=False)
    return model


def _build_adapter(tie_word_embeddings=None):
    adapter = object.__new__(DeepseekV32)
    if tie_word_embeddings is None:
        adapter.config = SimpleNamespace()
    else:
        adapter.config = SimpleNamespace(tie_word_embeddings=tie_word_embeddings)
    adapter.model = _build_fake_v32_model()
    return adapter


def test_get_embed_load_specs_uses_lm_head_prefix_when_untied():
    adapter = _build_adapter(tie_word_embeddings=False)

    specs = adapter.get_embed_load_specs()

    assert [prefix for _, prefix in specs] == [
        "model.embed_tokens.",
        "model.norm.",
        "lm_head.",
    ]
    assert specs[0][0] is adapter.model.embed
    assert specs[1][0] is adapter.model.norm
    assert specs[2][0] is adapter.model.head


def test_get_embed_load_specs_uses_embed_prefix_when_tied():
    adapter = _build_adapter(tie_word_embeddings=True)

    specs = adapter.get_embed_load_specs()

    assert [prefix for _, prefix in specs] == [
        "model.embed_tokens.",
        "model.norm.",
        "model.embed_tokens.",
    ]


def test_get_embed_load_specs_falls_back_to_untied_when_attr_missing():
    adapter = _build_adapter()

    specs = adapter.get_embed_load_specs()

    assert specs[2][1] == "lm_head."


def test_init_requires_trust_remote_code():
    args = SimpleNamespace(
        model_name="deepseek_v3_2",
        trust_remote_code=False,
    )

    with pytest.raises(
        ValueError,
        match="deepseek_v3_2 requires --trust_remote_code",
    ):
        DeepseekV32(args)
