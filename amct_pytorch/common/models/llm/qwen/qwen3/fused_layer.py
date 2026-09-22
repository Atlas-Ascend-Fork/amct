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
'''Qwen3 decoder layer with shared dynamic activation quantization.'''

import torch


class Qwen3FusedDecoderLayer:
    '''Move a Qwen3 decoder layer into a shared-quantization wrapper.'''

    def __new__(cls, layer, shared_quantize):
        # Qwen3 layers are expected to expose input_layernorm, self_attn,
        # post_attention_layernorm, and mlp. Move the original object into a
        # wrapper without copying parameters.
        wrapper = _Qwen3FusedDecoderLayer(shared_quantize)
        wrapper.input_layernorm = layer.input_layernorm
        wrapper.self_attn = layer.self_attn
        wrapper.post_attention_layernorm = layer.post_attention_layernorm
        wrapper.mlp = layer.mlp
        return wrapper


class _Qwen3FusedDecoderLayer(torch.nn.Module):
    def __init__(self, shared_quantize):
        super().__init__()
        self.shared_quantize = shared_quantize

    def forward(self, x, *args, **kwargs):
        residual = x
        hidden, pertoken_scale = self.shared_quantize(self.input_layernorm(x))
        hidden = self.self_attn.q_proj(hidden)
        hidden = hidden + self.self_attn.k_proj(hidden)
        hidden = hidden + self.self_attn.v_proj(hidden)
        x = residual + hidden
        residual = x
        hidden, pertoken_scale = self.shared_quantize(self.post_attention_layernorm(x))
        hidden = self.mlp.up_proj(hidden)
        gate = self.mlp.gate_proj(hidden)
        return residual + hidden * gate
