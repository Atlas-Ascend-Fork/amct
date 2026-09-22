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


class Linear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x, pertoken_scale=None):
        return nn.functional.linear(x, self.weight, self.bias)


class Attention(nn.Module):
    def __init__(self, hidden=8):
        super().__init__()
        self.q_proj = Linear(hidden, hidden)
        self.k_proj = Linear(hidden, hidden)
        self.v_proj = Linear(hidden, hidden)
        self.o_proj = Linear(hidden, hidden)
        self.num_heads = 2
        self.head_dim = hidden // self.num_heads
        self.scaling = self.head_dim**-0.5


class MLP(nn.Module):
    def __init__(self, hidden=8, intermediate=16):
        super().__init__()
        self.fc1 = Linear(hidden, intermediate)
        self.fc2 = Linear(intermediate, hidden)

    @staticmethod
    def act(x):
        return torch.nn.functional.gelu(x)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class ViTLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = Attention()
        self.layernorm_before = nn.LayerNorm(8)
        self.layernorm_after = nn.LayerNorm(8)
        self.mlp = MLP()

    def forward(self, hidden_states, attention_mask=None, **kwargs):
        residual = hidden_states
        hidden_states = self.layernorm_before(hidden_states)
        hidden_states = self.attention_forward(hidden_states, attention_mask)
        hidden_states = hidden_states + residual

        residual = hidden_states
        hidden_states = self.layernorm_after(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return hidden_states + residual

    def attention_forward(self, hidden_states, attention_mask):
        attn = self.attention
        shape = (*hidden_states.shape[:-1], -1, attn.head_dim)
        q = attn.q_proj(hidden_states).view(shape).transpose(1, 2)
        k = attn.k_proj(hidden_states).view(shape).transpose(1, 2)
        v = attn.v_proj(hidden_states).view(shape).transpose(1, 2)
        weight = torch.matmul(q, k.transpose(2, 3)) * attn.scaling
        weight = torch.softmax(weight, dim=-1, dtype=torch.float32).to(q.dtype)
        output = torch.matmul(weight, v).transpose(1, 2).contiguous()
        return attn.o_proj(output.reshape(*hidden_states.shape[:-1], -1))


def test_vit_fused_layer_one_dynamic_quant_for_qkv_and_mlp():
    from amct_pytorch.common.models.vit.fused_layer import ViTFusedLayer

    layer = ViTLayer()
    fused = ViTFusedLayer(layer, SharedDynamicQuantize())
    calls = []

    class CountingQuant(SharedDynamicQuantize):
        def forward(self, x):
            calls.append(x)
            return x, torch.ones(x.shape[:-1], dtype=torch.float32).reshape(-1)

    fused.shared_quantize = CountingQuant()
    x = torch.randn(2, 5, 8)
    output = fused(x)
    assert output.shape == x.shape
    assert len(calls) == 2
