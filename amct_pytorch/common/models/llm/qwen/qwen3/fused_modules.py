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
"""Qwen3 attention and MLP adapters for shared dynamic quantization."""

from amct_pytorch.classic.deploy_op.local_dynamic import make_local_adapter


def qwen3_attention(attention, act_type='int8'):
    return make_local_adapter('qwen3_attention', attention, act_type)


def _attention_forward(
    self,
    hidden_states,
    position_embeddings=None,
    attention_mask=None,
    past_key_values=None,
    **kwargs,
):
    from transformers.models.qwen3.modeling_qwen3 import (
        ALL_ATTENTION_FUNCTIONS,
        apply_rotary_pos_emb,
        eager_attention_forward,
    )

    if position_embeddings is None:
        raise ValueError('position_embeddings is required by fused attention')
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)
    quantized, scale = self.shared_quantize(hidden_states)
    query = self.q_norm(self.q_proj(quantized, scale).view(hidden_shape)).transpose(
        1, 2
    )
    key = self.k_norm(self.k_proj(quantized, scale).view(hidden_shape)).transpose(1, 2)
    value = self.v_proj(quantized, scale).view(hidden_shape).transpose(1, 2)
    cos, sin = position_embeddings
    query, key = apply_rotary_pos_emb(query, key, cos, sin)
    if past_key_values is not None:
        key, value = past_key_values.update(key, value, self.layer_idx)
    interface = ALL_ATTENTION_FUNCTIONS.get_interface(
        self.config._attn_implementation, eager_attention_forward
    )
    output, weights = interface(
        self,
        query,
        key,
        value,
        attention_mask,
        dropout=self.attention_dropout if self.training else 0.0,
        scaling=self.scaling,
        sliding_window=self.sliding_window,
        **kwargs,
    )
    output = output.reshape(*input_shape, -1).contiguous()
    return self.o_proj(output), weights


def qwen3_mlp(mlp, act_type='int8'):
    return make_local_adapter('qwen3_mlp', mlp, act_type)


def _mlp_forward(self, hidden_states):
    quantized, scale = self.shared_quantize(hidden_states)
    gate = self.act_fn(self.gate_proj(quantized, scale))
    up = self.up_proj(quantized, scale)
    return self.down_proj(gate * up)
