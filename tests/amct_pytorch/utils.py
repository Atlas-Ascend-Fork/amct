# ----------------------------------------------------------------------------
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------
import torch.nn as nn
from transformers.models.deepseek_v3.modeling_deepseek_v3 import DeepseekV3Attention as deepseek_module
from transformers.models.longcat_flash.modular_longcat_flash import LongcatFlashMLA as longcat_module
import torch


class TestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(64, 64)
        self.linear2 = nn.Linear(64, 32, bias=True)
        self.linear3 = nn.Linear(32, 1)

    def forward(self, inputs):
        x = self.linear1(inputs)
        x = self.linear2(x)
        x = self.linear3(x)
        return x


class TestModelBias(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(64, 64)
        self.linear2 = nn.Linear(64, 32, bias=False)
        self.linear3 = nn.Linear(32, 1)

    def forward(self, inputs):
        x = self.linear1(inputs)
        x = self.linear2(x)
        x = self.linear3(x)
        return x


class TestModelConv2d(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d1 = nn.Conv2d(32, 32, kernel_size=3, padding_mode='zeros')
        self.conv2d2 = nn.Conv2d(32, 64, kernel_size=3, padding_mode='zeros')
        self.conv2d3 = nn.Conv2d(64, 64, kernel_size=6, padding_mode='zeros')

    def forward(self, inputs):
        x = self.conv2d1(inputs)
        x = self.conv2d2(x)
        x = self.conv2d3(x)
        return x


class TestModelDeepseekV3Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.attn = deepseek_module(config, layer_idx=0)
        self.register_buffer("cos", torch.ones(4096, config.qk_rope_head_dim))
        self.register_buffer("sin", torch.zeros(4096, config.qk_rope_head_dim))

    def forward(self, hidden_states, attention_mask=None, past_key_values=None, **kwargs):
        batch, seq_len, _ = hidden_states.shape
        cos_sin = (self.cos[:seq_len].unsqueeze(0), self.sin[:seq_len].unsqueeze(0))
        
        attn_out, attn_weight = self.attn(
            hidden_states=hidden_states,
            position_embeddings=cos_sin,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            **kwargs
        )
        return attn_out

    
class TestModelLongcatFlashMLA(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.attn = longcat_module(config, layer_idx=0)
        self.register_buffer("cos", torch.ones(4096, config.qk_rope_head_dim))
        self.register_buffer("sin", torch.zeros(4096, config.qk_rope_head_dim))

    def forward(self, hidden_states, attention_mask=None, past_key_values=None, **kwargs):
        batch, seq_len, _ = hidden_states.shape
        cos_sin = (self.cos[:seq_len].unsqueeze(0), self.sin[:seq_len].unsqueeze(0))
        
        attn_out, attn_weight = self.attn(
            hidden_states=hidden_states,
            position_embeddings=cos_sin,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            **kwargs
        )
        return attn_out