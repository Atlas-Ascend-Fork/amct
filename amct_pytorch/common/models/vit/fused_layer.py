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
'''ViT encoder layer with shared dynamic activation quantization.'''

import torch


class ViTFusedLayer:
    '''Move a ViT encoder layer into a shared-quantization wrapper.'''

    def __new__(cls, layer, shared_quantize):
        shared_map = {'_default': shared_quantize}
        return ViTFusedLayer.from_shared_map(layer, shared_map, None)

    @classmethod
    def from_shared_map(cls, layer, shared_by_pre_op, layer_ref):
        '''Build a wrapper with one shared quantizer per pre_op reference.'''
        return _ViTFusedLayer(layer, shared_by_pre_op, layer_ref)


class _ViTFusedLayer(torch.nn.Module):
    _LAYER_ATTRS = ('attention', 'layernorm_before', 'layernorm_after', 'mlp')

    def __init__(self, layer, shared_by_pre_op, layer_ref):
        super().__init__()
        self._layer_ref = layer_ref
        self._shared_by_pre_op = torch.nn.ModuleDict(
            {ref.rsplit('.', 1)[-1]: quant for ref, quant in shared_by_pre_op.items()}
        )
        self.training = layer.training
        self.dropout = getattr(layer, 'dropout', torch.nn.Identity())
        self.shared_quantize = next(iter(self._shared_by_pre_op.values()), None)
        for attr in self._LAYER_ATTRS:
            setattr(self, attr, getattr(layer, attr))
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
    def _project(module, hidden, pertoken_scale):
        if pertoken_scale is not None:
            return module(hidden, pertoken_scale)
        return module(hidden)

    def _attention_forward(self, hidden_states, attention_mask, **kwargs):
        attn = self.attention
        input_shape = hidden_states.shape[:-1]
        head_dim = getattr(attn, 'head_dim', None)
        if head_dim is None:
            head_dim = attn.q_proj.weight.shape[-1] // getattr(attn, 'num_heads', 1)
        hidden_shape = (*input_shape, -1, head_dim)

        shared = self._shared_for('layernorm_before')
        h = self.layernorm_before(hidden_states)
        if shared is not None:
            q, scale = shared(h)
            query = self._project(attn.q_proj, q, scale)
            key = self._project(attn.k_proj, q, scale)
            value = self._project(attn.v_proj, q, scale)
        else:
            query = self._project(attn.q_proj, h, None)
            key = self._project(attn.k_proj, h, None)
            value = self._project(attn.v_proj, h, None)

        query = query.view(hidden_shape).transpose(1, 2)
        key = key.view(hidden_shape).transpose(1, 2)
        value = value.view(hidden_shape).transpose(1, 2)

        if hasattr(attn, 'config'):
            from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
            from transformers.models.vit.modeling_vit import eager_attention_forward

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
                **kwargs,
            )
        else:
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=attention_mask,
                scale=getattr(attn, 'scaling', head_dim**-0.5),
            ).transpose(1, 2)
        output = attn_output.reshape(*input_shape, -1).contiguous()
        o_proj = getattr(attn, 'o_proj', None)
        if o_proj is not None:
            return self._project(o_proj, output, None)
        return output

    def _mlp_forward(self, hidden_states):
        mlp = self.mlp
        shared = self._shared_for('layernorm_after')
        h = self.layernorm_after(hidden_states)
        if shared is not None:
            q, scale = shared(h)
            intermediate = self._project(mlp.fc1, q, scale)
        else:
            intermediate = self._project(mlp.fc1, h, None)
        if hasattr(mlp, 'activation_fn'):
            intermediate = mlp.activation_fn(intermediate)
        elif hasattr(mlp, 'act_fn'):
            intermediate = mlp.act_fn(intermediate)
        down = getattr(mlp, 'fc2', None)
        if down is not None:
            return self._project(down, intermediate, None)
        return intermediate

    def forward(self, hidden_states, attention_mask=None, **kwargs):
        residual = hidden_states
        hidden_states = self._attention_forward(hidden_states, attention_mask, **kwargs)
        hidden_states = residual + self.dropout(hidden_states)
        return hidden_states + self.dropout(self._mlp_forward(hidden_states))
