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

    @staticmethod
    def _repeat_kv(hidden_states, num_rep):
        if num_rep == 1:
            return hidden_states
        batch, heads, seq_len, head_dim = hidden_states.shape
        hidden_states = hidden_states[:, :, None, :, :].expand(
            batch, heads, num_rep, seq_len, head_dim
        )
        return hidden_states.reshape(batch, heads * num_rep, seq_len, head_dim)

    def _project(self, module, hidden, pertoken_scale):
        try:
            return module(hidden, pertoken_scale)
        except TypeError:
            return module(hidden)

    def _attention_forward(
        self, hidden_states, attention_mask, position_embeddings, past_key_values
    ):
        attn = self.self_attn
        input_shape = hidden_states.shape[:-1]
        head_dim = getattr(attn, 'head_dim', None)
        if head_dim is None:
            head_dim = attn.q_proj.weight.shape[-1] // getattr(attn, 'num_heads', 1)
        hidden_shape = (*input_shape, -1, head_dim)

        q, scale = self.shared_quantize(self.input_layernorm(hidden_states))
        query = self._project(attn.q_proj, q, scale)
        key = self._project(attn.k_proj, q, scale)
        value = self._project(attn.v_proj, q, scale)

        query = self._norm_and_transpose(attn, 'q_norm', query, hidden_shape)
        key = self._norm_and_transpose(attn, 'k_norm', key, hidden_shape)
        value = value.view(hidden_shape).transpose(1, 2)

        if position_embeddings is not None:
            cos, sin = position_embeddings
            query, key = _apply_rope(query, key, cos, sin)
        if past_key_values is not None:
            key, value = past_key_values.update(key, value, attn.layer_idx)

        groups = getattr(attn, 'num_key_value_groups', 1)
        key = self._repeat_kv(key, groups)
        value = self._repeat_kv(value, groups)
        scaling = getattr(attn, 'scaling', head_dim**-0.5)
        attn_weight = torch.matmul(query, key.transpose(2, 3)) * scaling
        if attention_mask is not None:
            if attention_mask.dtype == torch.bool:
                additive_mask = torch.zeros_like(
                    attn_weight[..., : attention_mask.shape[-1]],
                    dtype=attn_weight.dtype,
                )
                additive_mask.masked_fill_(attention_mask.logical_not(), float('-inf'))
                attn_weight = attn_weight + additive_mask
            else:
                attn_weight = attn_weight + attention_mask
        attn_weight = torch.softmax(attn_weight, dim=-1, dtype=torch.float32).to(
            query.dtype
        )
        attn_output = torch.matmul(attn_weight, value)
        attn_output = attn_output.transpose(1, 2).contiguous()
        output = attn_output.reshape(*input_shape, -1)
        o_proj = getattr(attn, 'o_proj', None)
        if o_proj is not None:
            return self._project(o_proj, output, None)
        return output

    @staticmethod
    def _norm_and_transpose(attn, norm_name, tensor, hidden_shape):
        norm = getattr(attn, norm_name, None)
        if norm is not None:
            tensor = norm(tensor.view(hidden_shape))
        else:
            tensor = tensor.view(hidden_shape)
        return tensor.transpose(1, 2)

    def _mlp_forward(self, hidden_states):
        mlp = self.mlp
        if not hasattr(mlp, 'up_proj') or not hasattr(mlp, 'gate_proj'):
            raise ValueError('Qwen3 MLP must expose up_proj and gate_proj')
        q, scale = self.shared_quantize(self.post_attention_layernorm(hidden_states))
        up = self._project(mlp.up_proj, q, scale)
        gate = self._project(mlp.gate_proj, q, scale)
        if hasattr(mlp, 'act_fn'):
            gate = mlp.act_fn(gate)
        hidden = gate * up
        down_proj = getattr(mlp, 'down_proj', None)
        if down_proj is not None:
            return self._project(down_proj, hidden, None)
        return hidden

    def forward(self, hidden_states, *args, **kwargs):
        residual = hidden_states
        hidden_states = self._attention_forward(
            hidden_states,
            kwargs.get('attention_mask'),
            kwargs.get('position_embeddings'),
            kwargs.get('past_key_values'),
        )
        hidden_states = residual + hidden_states
        return hidden_states + self._mlp_forward(hidden_states)


def _rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rope(q, k, cos, sin):
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return (
        (q * cos) + (_rotate_half(q) * sin),
        (k * cos) + (_rotate_half(k) * sin),
    )
