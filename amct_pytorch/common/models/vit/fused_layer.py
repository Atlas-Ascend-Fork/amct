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


class ViTFusedLayer:
    '''Move a ViT encoder layer into a shared-quantization wrapper.'''

    def __new__(cls, layer, shared_quantize):
        shared_map = {'_default': shared_quantize}
        return ViTFusedLayer.from_shared_map(layer, shared_map, None)

    @classmethod
    def from_shared_map(cls, layer, shared_by_pre_op, layer_ref):
        '''Build a wrapper with one shared quantizer per pre_op reference.'''
        return _make_fused_layer_class(type(layer), _ViTFusedLayer)(
            layer, shared_by_pre_op, layer_ref
        )


class _ViTFusedLayer(torch.nn.Module):
    def __init__(self, layer, shared_by_pre_op, layer_ref):
        super().__init__()
        self._layer_ref = layer_ref
        self._shared_by_pre_op = torch.nn.ModuleDict(
            {ref.rsplit('.', 1)[-1]: quant for ref, quant in shared_by_pre_op.items()}
        )
        self.training = layer.training
        self.dropout = getattr(layer, 'dropout', torch.nn.Identity())
        for key, value in layer._modules.items():  # pylint: disable=protected-access
            setattr(self, key, value)
        for key, value in vars(layer).items():
            if key.startswith('_') or key in self._modules:
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
            return self._shared_by_pre_op[pre_op]
        if '_default' in self._shared_by_pre_op:
            return self._shared_by_pre_op['_default']
        return None

    @staticmethod
    def _module_by_path(root, path):
        module = root
        for attr in path:
            module = getattr(module, attr)
        return module

    @classmethod
    def _projection(cls, attn, paths):
        for path in paths:
            try:
                return cls._module_by_path(attn, path)
            except AttributeError:
                continue
        raise AttributeError(f'cannot resolve projection paths: {paths}')

    @classmethod
    def _qkv_projection(cls, attn):
        return (
            cls._projection(attn, (('q_proj',), ('attention', 'query'))),
            cls._projection(attn, (('k_proj',), ('attention', 'key'))),
            cls._projection(attn, (('v_proj',), ('attention', 'value'))),
        )

    @classmethod
    def _output_projection(cls, attn):
        return cls._projection(
            attn,
            (
                ('o_proj',),
                ('output', 'dense'),
                ('attention', 'output', 'dense'),
            ),
        )

    @classmethod
    def _mlp_modules(cls, layer):
        if hasattr(layer, 'mlp'):
            mlp = layer.mlp
            up = getattr(mlp, 'fc1', None)
            down = getattr(mlp, 'fc2', None)
            activation = getattr(mlp, 'activation_fn', None)
        else:
            up = layer.intermediate.dense
            down = layer.output.dense
            activation = layer.intermediate.intermediate_act_fn
            dropout = getattr(layer.output, 'dropout', None)
            return up, down, activation, dropout
        return up, down, activation, None

    @staticmethod
    def _project(module, hidden, pertoken_scale):
        if pertoken_scale is not None:
            return module(hidden, pertoken_scale)
        return module(hidden)

    def _attention_forward(self, hidden_states, attention_mask, **kwargs):
        attn = self.attention
        input_shape = hidden_states.shape[:-1]
        projection = getattr(attn, 'attention', attn)
        head_dim = (
            getattr(attn, 'head_dim', None)
            or getattr(projection, 'head_dim', None)
            or getattr(projection, 'attention_head_size', None)
        )
        if head_dim is None:
            raise ValueError('cannot resolve head_dim for fused attention')
        hidden_shape = (*input_shape, -1, head_dim)
        q_proj, k_proj, v_proj = self._qkv_projection(attn)

        shared = self._shared_for('layernorm_before')
        h = self.layernorm_before(hidden_states)
        if shared is not None:
            q, scale = shared(h)
            query = self._project(q_proj, q, scale)
            key = self._project(k_proj, q, scale)
            value = self._project(v_proj, q, scale)
        else:
            query = self._project(q_proj, h, None)
            key = self._project(k_proj, h, None)
            value = self._project(v_proj, h, None)

        query = query.view(hidden_shape).transpose(1, 2)
        key = key.view(hidden_shape).transpose(1, 2)
        value = value.view(hidden_shape).transpose(1, 2)

        attention_impl = attn if hasattr(attn, 'config') else projection
        if hasattr(attention_impl, 'config'):
            from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
            from transformers.models.vit.modeling_vit import eager_attention_forward

            interface = ALL_ATTENTION_FUNCTIONS.get_interface(
                attention_impl.config._attn_implementation, eager_attention_forward
            )
            attn_output, _ = interface(
                attention_impl,
                query,
                key,
                value,
                attention_mask,
                dropout=(
                    (
                        getattr(attention_impl, 'attention_dropout', None)
                        or getattr(attention_impl, 'dropout_prob', 0.0)
                    )
                    if attention_impl.training
                    else 0.0
                ),
                scaling=attention_impl.scaling,
                **kwargs,
            )
        else:
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=attention_mask,
                scale=getattr(attention_impl, 'scaling', head_dim**-0.5),
            ).transpose(1, 2)
        output = attn_output.reshape(*input_shape, -1).contiguous()
        output = self._project(self._output_projection(attn), output, None)
        output_dropout = getattr(self._output_parent(attn), 'dropout', None)
        if output_dropout is not None:
            output = output_dropout(output)
        return output

    @staticmethod
    def _output_parent(attn):
        output = getattr(attn, 'output', None)
        return output if output is not None else attn

    def _mlp_forward(self, hidden_states):
        mlp = getattr(self, 'mlp', None)
        output_dropout = None
        if mlp is not None:
            up, down, activation, _ = self._mlp_modules(self)
        else:
            up, down, activation, output_dropout = self._mlp_modules(self)
        shared = self._shared_for('layernorm_after')
        h = self.layernorm_after(hidden_states)
        if shared is not None:
            q, scale = shared(h)
            intermediate = self._project(up, q, scale)
        else:
            intermediate = self._project(up, h, None)
        if activation is not None:
            intermediate = activation(intermediate)
        output = self._project(down, intermediate, None)
        if output_dropout is not None:
            output = output_dropout(output)
        return output

    def forward(self, hidden_states, attention_mask=None, **kwargs):
        residual = hidden_states
        attention_output = self._attention_forward(
            hidden_states, attention_mask, **kwargs
        )
        if hasattr(self, 'mlp'):
            hidden_states = residual + self.dropout(attention_output)
            return hidden_states + self.dropout(self._mlp_forward(hidden_states))
        hidden_states = residual + attention_output
        return self._mlp_forward(hidden_states) + hidden_states
