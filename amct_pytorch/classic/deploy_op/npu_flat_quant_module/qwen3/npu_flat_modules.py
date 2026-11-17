# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
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
"""Qwen3 deploy modules for FlatQuant (W4A4).

from_quant_module packs each reparameterized projection into the operator
format; forward orchestrates npu_kronecker_quant + npu_quant_matmul per
the original Qwen3Attention structure (q/k/v via the fused kron path,
o_proj via the single-matrix + dynamic-quant path when use_o_quant)."""

from typing import Callable

import torch

from ..npu_flat_base import NpuFlatQuantDeployMixin
from ..npu_flat_utils import (
    pack_projection,
    pad_to_kronecker_multiple,
    w4a4_dynamic_linear,
    w4a4_flatquant_linear,
)


def _qwen3_attention_helpers():
    """Lazy import of the transformers Qwen3 attention stack; see
    flatquant/qwen3/flat_attention.py for the module-level-import caveat."""
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    from transformers.models.qwen3.modeling_qwen3 import (
        apply_rotary_pos_emb,
        eager_attention_forward,
    )

    return ALL_ATTENTION_FUNCTIONS, apply_rotary_pos_emb, eager_attention_forward


class NpuFlatQuantQwen3Attention(NpuFlatQuantDeployMixin):
    """W4A4 deploy module for FlatQuantQwen3Attention."""

    _quant_module_prefix = 'FlatQuantQwen3Attention'

    def _from_quant_module(self, q):
        self.config = q.module_config
        self.layer_idx = q.layer_idx
        self.head_dim = q.head_dim
        self.num_key_value_groups = q.num_key_value_groups
        self.scaling = q.scaling
        self.attention_dropout = q.attention_dropout
        self.is_causal = True

        self.q_norm = q.q_norm
        self.k_norm = q.k_norm
        if hasattr(q, 'sliding_window'):
            self.sliding_window = q.sliding_window

        for name in ('q', 'k', 'v'):
            proj = getattr(q, f'{name}_proj')
            packed = pack_projection(proj, q.ln_trans)
            (
                wt,
                scale,
                left,
                right,
                clip,
            ) = packed
            setattr(self, f'wt_{name}_packed', self._pack_param(wt))
            setattr(self, f'scale_{name}_proj', self._pack_param(scale))
            setattr(self, f'{name}_lac', clip)
        self.left_trans = self._pack_param(q.ln_trans.matrix_left.T.contiguous())
        self.right_trans = self._pack_param(q.ln_trans.matrix_right.contiguous())

        if q.flat_config.use_o_quant:
            self.o_trans = self._pack_param(q.o_trans.get_matrix().T.contiguous())
            q.o_proj.weight_quantizer.find_params(q.o_proj.linear.weight)
            wt_o, scale_o = q.o_proj.weight_quantizer.quantize(
                q.o_proj.linear.weight, quantonly=True
            )
            from ..npu_flat_utils import pack_int4_weights

            self.wt_o_packed = self._pack_param(pack_int4_weights(wt_o.data))
            self.scale_o_proj = self._pack_param(scale_o)
        elif q.vcache_trans is not None:
            # v-cache-only variant: o side carries v^-T on the fly
            self.v_trans = self._pack_param(
                q.vcache_trans.get_matrix(inv_t=True).T.contiguous()
            )
            q.o_proj.weight_quantizer.find_params(q.o_proj.linear.weight)
            wt_o, scale_o = q.o_proj.weight_quantizer.quantize(
                q.o_proj.linear.weight, quantonly=True
            )
            from ..npu_flat_utils import pack_int4_weights

            self.wt_o_packed = self._pack_param(pack_int4_weights(wt_o.data))
            self.scale_o_proj = self._pack_param(scale_o)
        else:
            self.o_proj = q.o_proj.linear

    def forward(self, hidden_states, **kwargs):
        bsz, q_len, _ = hidden_states.size()

        # npu_kronecker_quant requires bsz*q_len % 16 == 0
        hidden_states, pad_after = pad_to_kronecker_multiple(hidden_states)

        query_states = w4a4_flatquant_linear(
            self.left_trans,
            self.right_trans,
            self.wt_q_packed,
            self.scale_q_proj,
            hidden_states,
            self.q_lac,
        )
        key_states = w4a4_flatquant_linear(
            self.left_trans,
            self.right_trans,
            self.wt_k_packed,
            self.scale_k_proj,
            hidden_states,
            self.k_lac,
        )
        value_states = w4a4_flatquant_linear(
            self.left_trans,
            self.right_trans,
            self.wt_v_packed,
            self.scale_v_proj,
            hidden_states,
            self.v_lac,
        )
        if pad_after:
            query_states = query_states[:, :q_len]
            key_states = key_states[:, :q_len]
            value_states = value_states[:, :q_len]
        return self.forward_qkv(
            query_states, key_states, value_states, bsz, q_len, **kwargs
        )

    def forward_qkv(self, query_states, key_states, value_states, bsz, q_len, **kwargs):
        all_attention_functions, apply_rotary_pos_emb, eager_attention_forward = (
            _qwen3_attention_helpers()
        )
        hidden_shape = (bsz, q_len, -1, self.head_dim)

        if self.q_norm is None or self.k_norm is None:
            query_states = query_states.view(hidden_shape).transpose(1, 2)
            key_states = key_states.view(hidden_shape).transpose(1, 2)
        else:
            query_states = self.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
            key_states = self.k_norm(key_states.view(hidden_shape)).transpose(1, 2)
        value_states = value_states.view(hidden_shape).transpose(1, 2)

        cos, sin = kwargs.pop('position_embeddings')
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

        past_key_values = None
        if kwargs.get('past_key_value') is not None:
            past_key_values = kwargs.pop('past_key_value')
        if kwargs.get('past_key_values') is not None:
            past_key_values = kwargs.pop('past_key_values')
        if past_key_values is not None:
            cache_kwargs = {
                'sin': sin,
                'cos': cos,
                'cache_position': kwargs.pop('cache_position', None),
            }
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        attention_interface: Callable = eager_attention_forward
        attn_implementation = getattr(self.config, '_attn_implementation')
        if attn_implementation != 'eager':
            attention_interface = all_attention_functions[attn_implementation]

        if hasattr(self, 'sliding_window'):
            kwargs['sliding_window'] = self.sliding_window
        attention_mask = kwargs.pop('attention_mask')
        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        # o_proj path: composite baked at convert; single matrix
        # on the runtime side, dynamic per-token INT4 quant
        attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
        if hasattr(self, 'wt_o_packed'):
            init_shape = attn_output.shape
            attn_shape = [-1, self.config.num_attention_heads, self.head_dim]
            attn_output = attn_output.reshape(attn_shape)
            if hasattr(self, 'o_trans'):
                # v->o composite: left-multiply O^T (matches the calibration
                # side's composite bake)
                attn_output = torch.matmul(
                    self.o_trans.to(attn_output), attn_output
                ).reshape(init_shape)
            else:
                # v-cache-only variant: fold v^-T on the fly with a
                # right-multiply — the same orientation as the calibration
                # side (attn @ (P^-1)^T); v_trans is (head_dim, head_dim)
                # so it broadcasts over the (tokens, heads, head_dim) input.
                attn_output = torch.matmul(
                    attn_output, self.v_trans.to(attn_output)
                ).reshape(init_shape)
            attn_output = w4a4_dynamic_linear(
                attn_output, self.wt_o_packed, self.scale_o_proj
            )
        else:
            attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights


class NpuFlatQuantQwen3MLP(NpuFlatQuantDeployMixin):
    """W4A4 deploy module for FlatQuantQwen3MLP."""

    _quant_module_prefix = 'FlatQuantQwen3MLP'

    def _from_quant_module(self, q):
        self.act_fn = q.act_fn

        for name in ('up', 'gate'):
            proj = getattr(q, f'{name}_proj')
            wt, scale, left, right, clip = pack_projection(proj, q.up_gate_trans)
            setattr(self, f'wt_{name}_packed', self._pack_param(wt))
            setattr(self, f'scale_{name}_proj', self._pack_param(scale))
            setattr(self, f'{name}_lac', clip)
        self.ug_left_trans = self._pack_param(
            q.up_gate_trans.matrix_left.T.contiguous()
        )
        self.ug_right_trans = self._pack_param(
            q.up_gate_trans.matrix_right.contiguous()
        )

        if q.flat_config.use_down_quant:
            wt, scale, left, right, clip = pack_projection(q.down_proj, q.down_trans)
            self.wt_down_packed = self._pack_param(wt)
            self.scale_down_proj = self._pack_param(scale)
            self.down_left_trans = self._pack_param(left)
            self.down_right_trans = self._pack_param(right)
            self.down_lac = clip
        else:
            self.down_proj = q.down_proj.linear

    def forward(self, x):
        bsz, q_len, _ = x.size()

        # same 16-alignment requirement as the attention part
        x, pad_after = pad_to_kronecker_multiple(x)

        up_states = w4a4_flatquant_linear(
            self.ug_left_trans,
            self.ug_right_trans,
            self.wt_up_packed,
            self.scale_up_proj,
            x,
            self.up_lac,
        )
        gate_states = w4a4_flatquant_linear(
            self.ug_left_trans,
            self.ug_right_trans,
            self.wt_gate_packed,
            self.scale_gate_proj,
            x,
            self.gate_lac,
        )
        x_act_fn = self.act_fn(gate_states) * up_states
        if hasattr(self, 'wt_down_packed'):
            down_states = w4a4_flatquant_linear(
                self.down_left_trans,
                self.down_right_trans,
                self.wt_down_packed,
                self.scale_down_proj,
                x_act_fn,
                self.down_lac,
            )
        else:
            down_states = self.down_proj(x_act_fn)

        if pad_after:
            down_states = down_states[:, :q_len]
        return down_states
