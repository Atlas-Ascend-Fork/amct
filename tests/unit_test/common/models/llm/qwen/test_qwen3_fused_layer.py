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
from torch import nn

from amct_pytorch.classic.deploy_op.shared_quantize import SharedDynamicQuantize
import importlib.util
from pathlib import Path as _Path

_spec = importlib.util.spec_from_file_location(
    'amct_qwen3_fused_layer',
    _Path(__file__).resolve().parents[6]
    / 'amct_pytorch/common/models/llm/qwen/qwen3/fused_layer.py',
)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
Qwen3FusedDecoderLayer = _module.Qwen3FusedDecoderLayer


class Linear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x, pertoken_scale=None):
        return torch.nn.functional.linear(x, self.weight, self.bias)


class Attention(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.head_dim = hidden
        self.q_proj = Linear(hidden, hidden)
        self.k_proj = Linear(hidden, hidden)
        self.v_proj = Linear(hidden, hidden)

    def forward(self, x):
        return self.q_proj(x) + self.k_proj(x) + self.v_proj(x)


class MLP(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.up_proj = Linear(hidden, hidden)
        self.gate_proj = Linear(hidden, hidden)

    def forward(self, x):
        return self.up_proj(x) * self.gate_proj(x)


class Layer(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.input_layernorm = nn.LayerNorm(hidden)
        self.self_attn = Attention(hidden)
        self.post_attention_layernorm = nn.LayerNorm(hidden)
        self.mlp = MLP(hidden)

    def forward(self, x):
        residual = x
        h = self.self_attn(self.input_layernorm(x))
        x = residual + h
        return x + self.mlp(self.post_attention_layernorm(x))


def test_qwen3_fused_layer_one_dynamic_quant_for_qkv_and_mlp():
    hidden = 6
    layer = Layer(hidden)
    fused = Qwen3FusedDecoderLayer(layer, SharedDynamicQuantize())
    calls = []

    class CountingQuant(SharedDynamicQuantize):
        def forward(self, x):
            calls.append(x)
            return x, torch.ones(x.shape[:-1], dtype=torch.float32).reshape(-1)

    fused.shared_quantize = CountingQuant()
    x = torch.randn(2, 3, hidden)
    rope = (torch.ones(2, 3, hidden), torch.zeros(2, 3, hidden))
    output = fused(x, position_embeddings=rope)
    assert output.shape == x.shape
    assert len(calls) == 2
    assert calls[0] is not None
    assert calls[1] is not None
    assert fused.input_layernorm is layer.input_layernorm
    assert fused.self_attn is layer.self_attn
    assert fused.post_attention_layernorm is layer.post_attention_layernorm


def test_shared_dynamic_quantize_returns_two_values():
    quantize = SharedDynamicQuantize()
    assert hasattr(quantize, 'forward')
