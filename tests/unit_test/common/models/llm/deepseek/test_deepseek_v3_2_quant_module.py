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
"""Tests for DeepSeek V3.2 quant modules with tiny CPU-only fakes."""

from types import SimpleNamespace

import torch
import torch.nn as nn

from amct_pytorch.common.models.llm.deepseek.deepseek_v3_2.quant_module import (
    QuantIndexer,
)
from amct_pytorch.quantization.bit_policy import BitPolicy
from amct_pytorch.quantization.dtypes import register_dtype


register_dtype()


def _quant_args(quant_target=()):
    return SimpleNamespace(
        algos=[],
        quant_dtype="int",
        w_bits=8,
        a_bits=8,
        quant_target=list(quant_target),
        bit_policy=BitPolicy(
            {
                "attn-linear": {
                    "wq_b": {"w_bits": 8, "a_bits": 8},
                },
                "attn-cache": {"q": 8, "k": 8, "p": 8, "v": 8},
            }
        ),
    )


class _FakeIndexer(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_lora_rank = 8
        self.n_heads = 2
        self.head_dim = 4
        self.rope_head_dim = 2
        self.index_topk = 1
        self.softmax_scale = 1.0
        self.wq_b = nn.Linear(self.q_lora_rank, self.n_heads * self.head_dim)
        self.wk = nn.Linear(6, self.head_dim)
        self.k_norm = nn.LayerNorm(self.head_dim)
        self.weights_proj = nn.Linear(6, self.n_heads)


def _run_indexer_forward(quant_indexer):
    x = torch.randn(1, 2, 6)
    qr = torch.randn(1, 2, 8)
    freqs_cis = torch.ones(2, 1, dtype=torch.complex64)

    return quant_indexer(x, qr, 0, freqs_cis, mask=None)


def test_quant_indexer_passthrough_forward_returns_topk_indices(monkeypatch):
    quant_indexer = QuantIndexer(_quant_args(), _FakeIndexer())
    monkeypatch.setattr(
        "amct_pytorch.common.models.llm.deepseek.deepseek_v3_2.quant_module.rotate_activation",
        lambda x: x,
    )

    topk_indices = _run_indexer_forward(quant_indexer)

    assert topk_indices.shape == (1, 2, 1)
    assert torch.all(topk_indices < 2)


def test_quant_indexer_attn_linear_forward_returns_topk_indices(monkeypatch):
    quant_indexer = QuantIndexer(_quant_args(["attn-linear"]), _FakeIndexer())
    monkeypatch.setattr(
        "amct_pytorch.common.models.llm.deepseek.deepseek_v3_2.quant_module.rotate_activation",
        lambda x: x,
    )

    topk_indices = _run_indexer_forward(quant_indexer)

    assert topk_indices.shape == (1, 2, 1)
    assert torch.all(topk_indices < 2)
