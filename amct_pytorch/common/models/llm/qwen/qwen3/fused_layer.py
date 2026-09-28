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


def _make_fused_layer_class(layer_type, mixin):
    """Keep a concrete original layer type in the fused wrapper's MRO.

    Transformers output capture and gradient checkpointing are enabled by
    isinstance checks against the registered layer class. Test stubs derived
    directly from nn.Module keep the plain mixin because inheriting the same
    base twice would create an inconsistent MRO.
    """
    bases = (layer_type, mixin) if layer_type is not torch.nn.Module else (mixin,)
    return type(
        f'Fused{layer_type.__name__}',
        bases,
        {'__init__': mixin.__init__, 'forward': mixin.forward},
    )


class Qwen3FusedDecoderLayer:
    '''Move a Qwen3 decoder layer into a shared-quantization wrapper.

    The wrapper owns the layer forward, so per-pre-op shared quantizers can
    receive both (qx, pertoken_scale) values inside our own forward. All
    original submodules are moved in by reference; other layer attributes are
    carried over so external references keep working.
    '''

    def __new__(cls, layer, shared_quantize):
        shared_map = {'_default': shared_quantize}
        return Qwen3FusedDecoderLayer.from_shared_map(layer, shared_map, None)

    @classmethod
    def from_shared_map(cls, layer, shared_by_pre_op, layer_ref):
        '''Build a wrapper with one shared quantizer per pre_op reference.

        shared_by_pre_op maps the absolute pre_op module path (e.g.
        "model.layers.0.input_layernorm") to its SharedDynamicQuantize. The
        attention pre_op and the mlp pre_op of the same layer each get their
        own quantizer; missing entries fall back to no shared quantization
        for that block (members keep their own input quantization).
        '''
        return _make_fused_layer_class(type(layer), _Qwen3FusedDecoderLayer)(
            layer, shared_by_pre_op, layer_ref
        )


class _Qwen3FusedDecoderLayer(torch.nn.Module):
    _LAYER_ATTRS = (
        'input_layernorm',
        'self_attn',
        'post_attention_layernorm',
        'mlp',
    )

    def __init__(self, layer, shared_by_pre_op, layer_ref):
        super().__init__()
        self._layer_ref = layer_ref
        self._shared_by_pre_op = torch.nn.ModuleDict(
            {ref.rsplit('.', 1)[-1]: quant for ref, quant in shared_by_pre_op.items()}
        )
        self.training = layer.training
        # Keep the single-quantizer attribute name for layers with one group
        # (and as the injection point used by tests): the first shared
        # quantizer of the layer.
        self.shared_quantize = next(iter(self._shared_by_pre_op.values()), None)
        # Move the original submodules in by reference (no copies).
        for attr in self._LAYER_ATTRS:
            setattr(self, attr, getattr(layer, attr))
        # Carry over remaining plain attributes so external code that touches
        # layer metadata keeps working (self_attn/mlp internals are untouched).
        for key, value in vars(layer).items():
            if key.startswith('_') or key in self._LAYER_ATTRS:
                continue
            if isinstance(value, torch.nn.Module):
                setattr(self, key, value)
            elif not callable(value) and not hasattr(self, key):
                try:
                    setattr(self, key, value)
                except (TypeError, AttributeError):
                    pass

    def _shared_for(self, pre_op):
        if pre_op in self._shared_by_pre_op:
            if len(self._shared_by_pre_op) == 1:
                return self.shared_quantize
            return self._shared_by_pre_op[pre_op]
        if '_default' in self._shared_by_pre_op:
            return self.shared_quantize
        return None

    @staticmethod
    def _repeat_kv(hidden_states, num_rep):
        if num_rep == 1:
            return hidden_states
        batch, heads, seq_len, head_dim = hidden_states.shape
        hidden_states = hidden_states[:, :, None, :, :].expand(
            batch, heads, num_rep, seq_len, head_dim
        )
        return hidden_states.reshape(batch, heads * num_rep, seq_len, head_dim)

    @staticmethod
    def _project(module, hidden, pertoken_scale):
        # Quantized deploy members accept (x, pertoken_scale); the None case
        # covers non-quantized projections and the o_proj/down_proj tails.
        if pertoken_scale is not None:
            return module(hidden, pertoken_scale)
        return module(hidden)

    def _attention_forward(
        self,
        hidden_states,
        attention_mask,
        position_embeddings,
        past_key_values,
        **kwargs,
    ):
        attn = self.self_attn
        input_shape = hidden_states.shape[:-1]
        head_dim = getattr(attn, 'head_dim', None)
        if head_dim is None:
            raise ValueError('cannot resolve head_dim for fused attention')
        hidden_shape = (*input_shape, -1, head_dim)

        shared = self._shared_for('input_layernorm')
        h = self.input_layernorm(hidden_states)
        if shared is not None:
            q, scale = shared(h)
            query = self._project(attn.q_proj, q, scale)
            key = self._project(attn.k_proj, q, scale)
            value = self._project(attn.v_proj, q, scale)
        else:
            query = self._project(attn.q_proj, h, None)
            key = self._project(attn.k_proj, h, None)
            value = self._project(attn.v_proj, h, None)

        query = self._norm_and_transpose(attn, 'q_norm', query, hidden_shape)
        key = self._norm_and_transpose(attn, 'k_norm', key, hidden_shape)
        value = value.view(hidden_shape).transpose(1, 2)

        if position_embeddings is None:
            raise ValueError(
                'position_embeddings is required by Qwen3FusedDecoderLayer'
            )
        cos, sin = position_embeddings
        query, key = _apply_rope(query, key, cos, sin)
        if past_key_values is not None:
            key, value = past_key_values.update(key, value, attn.layer_idx)

        if hasattr(attn, 'config'):
            from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
            from transformers.models.qwen3.modeling_qwen3 import eager_attention_forward

            interface = ALL_ATTENTION_FUNCTIONS.get_interface(
                attn.config._attn_implementation, eager_attention_forward
            )
            attn_output, _ = interface(
                attn,
                query,
                key,
                value,
                attention_mask,
                dropout=attn.attention_dropout if attn.training else 0.0,
                scaling=attn.scaling,
                sliding_window=attn.sliding_window,
                **kwargs,
            )
        else:
            groups = getattr(attn, 'num_key_value_groups', 1)
            key = self._repeat_kv(key, groups)
            value = self._repeat_kv(value, groups)
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=attention_mask,
                is_causal=attention_mask is None and query.shape[-2] > 1,
                scale=getattr(attn, 'scaling', head_dim**-0.5),
            ).transpose(1, 2)
        output = attn_output.reshape(*input_shape, -1).contiguous()
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
        shared = self._shared_for('post_attention_layernorm')
        h = self.post_attention_layernorm(hidden_states)
        if shared is not None:
            q, scale = shared(h)
            up = self._project(mlp.up_proj, q, scale)
            gate = self._project(mlp.gate_proj, q, scale)
        else:
            up = self._project(mlp.up_proj, h, None)
            gate = self._project(mlp.gate_proj, h, None)
        if hasattr(mlp, 'act_fn'):
            gate = mlp.act_fn(gate)
        hidden = gate * up
        down_proj = getattr(mlp, 'down_proj', None)
        if down_proj is not None:
            return self._project(down_proj, hidden, None)
        return hidden

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        use_cache=False,
        position_embeddings=None,
        **kwargs,
    ):
        residual = hidden_states
        hidden_states = self._attention_forward(
            hidden_states,
            attention_mask,
            position_embeddings,
            past_key_values,
            position_ids=position_ids,
            use_cache=use_cache,
            **kwargs,
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
