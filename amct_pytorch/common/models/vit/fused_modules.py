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
"""ViT attention and MLP adapters for shared dynamic quantization."""

from amct_pytorch.classic.deploy_op.local_dynamic import make_local_adapter


def vit_attention(attention, act_type='int8'):
    return make_local_adapter('vit_attention', attention, act_type)


def _attention(self, query, key, value, mask, dropout, **kwargs):
    from transformers.models.vit.modeling_vit import (
        ALL_ATTENTION_FUNCTIONS,
        eager_attention_forward,
    )

    interface = ALL_ATTENTION_FUNCTIONS.get_interface(
        self.config._attn_implementation, eager_attention_forward
    )
    return interface(
        self,
        query,
        key,
        value,
        mask,
        dropout=dropout if self.training else 0.0,
        scaling=self.scaling,
        **kwargs,
    )


def _attention_forward(self, hidden_states, attention_mask=None, **kwargs):
    input_shape = hidden_states.shape[:-1]
    shape = (*input_shape, -1, self.head_dim)
    quantized, scale = self.shared_quantize(hidden_states)
    query = self.q_proj(quantized, scale).view(shape).transpose(1, 2)
    key = self.k_proj(quantized, scale).view(shape).transpose(1, 2)
    value = self.v_proj(quantized, scale).view(shape).transpose(1, 2)
    output, weights = _attention(
        self, query, key, value, attention_mask, self.attention_dropout, **kwargs
    )
    output = output.reshape(*input_shape, -1).contiguous()
    return self.o_proj(output), weights


def vit_mlp(mlp, act_type='int8'):
    return make_local_adapter('vit_mlp', mlp, act_type)


def _mlp_forward(self, hidden_states):
    quantized, scale = self.shared_quantize(hidden_states)
    return self.fc2(self.activation_fn(self.fc1(quantized, scale)))
