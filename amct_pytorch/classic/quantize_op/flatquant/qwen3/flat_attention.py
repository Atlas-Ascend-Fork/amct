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
#
# The code snippet comes from FlatQuant (https://github.com/ruikangliu/FlatQuant).
#
# Copyright (c) 2024 ruikangliu. All rights reserved.
#
# This file is part of FlatQuant, which is released under the MIT License.
# See the LICENSE file in the root directory of this source tree
# or at https://opensource.org/licenses/MIT for details.
#
# --------------------------------------------------------------------------------
"""Qwen3 attention reconstruction for FlatQuant.

Each nn.Linear becomes a FlatQuantizedLinear (weight by reference), plus:
- ln_trans: SVDDecomposeTransMatrix shared by q/k/v input side
- o_trans / vcache_trans: SVDSingleTransMatrix pair forming the v->o
  composite when use_o_quant is on"""

from typing import Callable

import torch
import torch.nn as nn

from ..flat_base import FlatQuantModuleBase
from ..flat_config import flat_config_parser
from ..flat_linear import FlatQuantizedLinear
from ..quantizer import ActivationQuantizer
from ..trans_utils import (
    SVDDecomposeTransMatrix,
    SVDSingleTransMatrix,
    get_decompose_dim,
    get_init_scale,
)


def _qwen3_attention_helpers():
    """Lazy import of the transformers Qwen3 attention stack.

    transformers.modeling_utils pulls the accelerate integration, which
    probes torch_npu via find_spec and raises on partially mocked modules
    (unit-test subprocess isolation). Importing here keeps the failure out
    of amct_pytorch's import graph."""
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    from transformers.models.qwen3.modeling_qwen3 import (
        apply_rotary_pos_emb,
        eager_attention_forward,
    )

    return ALL_ATTENTION_FUNCTIONS, apply_rotary_pos_emb, eager_attention_forward


class FlatQuantQwen3Attention(FlatQuantModuleBase):
    """Reconstructed Qwen3Attention; forward signature stays identical."""

    def __init__(self, module: nn.Module, layer_name: str = '', layer_config=None):
        super().__init__(module, layer_name, layer_config)

        self.flat_config = flat_config_parser(layer_config)

        # keep original module's structural attributes
        self.module_config = module.config
        self.layer_idx = module.layer_idx
        self.head_dim = module.head_dim
        self.num_key_value_groups = module.num_key_value_groups
        self.scaling = module.scaling
        self.attention_dropout = module.attention_dropout
        self.is_causal = True

        self.q_norm = module.q_norm if hasattr(module, 'q_norm') else None
        self.k_norm = module.k_norm if hasattr(module, 'k_norm') else None
        if hasattr(module, 'sliding_window'):
            self.sliding_window = module.sliding_window

        # every Linear -> FlatQuantizedLinear (weights by reference)
        self.q_proj = FlatQuantizedLinear(
            module.q_proj, self.flat_config, self.flat_config.lac
        )
        self.k_proj = FlatQuantizedLinear(
            module.k_proj, self.flat_config, self.flat_config.lac
        )
        self.v_proj = FlatQuantizedLinear(
            module.v_proj, self.flat_config, self.flat_config.lac
        )
        self.o_proj = FlatQuantizedLinear(module.o_proj, self.flat_config, False)
        self.add_fq_trans()

        if self.flat_config.use_kcache_quant and self.flat_config.k_bits < 16:
            self.k_cache_quantizer = ActivationQuantizer(
                bits=self.flat_config.k_bits,
                sym=self.flat_config.k_sym,
                lac=self.flat_config.lac,
            )
        if self.flat_config.use_vcache_quant and self.flat_config.v_bits < 16:
            self.v_cache_quantizer = ActivationQuantizer(
                bits=self.flat_config.v_bits,
                sym=self.flat_config.v_sym,
                lac=self.flat_config.lac,
            )

        # float passthrough by default
        self._ori_mode = True
        self._eval_mode = False
        # running max of LN output stats, consumed by init_diag_scale
        self.ln_smax = (
            torch.ones_like(self.q_proj.linear.weight.abs().max(dim=0)[0]) * 1e-5
        )

    def add_fq_trans(self):
        """Attach the learnable transforms."""
        self.ln_trans, self.o_trans = None, None
        self.vcache_trans, self.kcache_trans = None, None
        if self.flat_config.a_bits < 16 or self.flat_config.w_bits < 16:
            ln_dim_left, ln_dim_right = get_decompose_dim(
                self.q_proj.linear.weight.shape[1]
            )
            self.ln_trans = SVDDecomposeTransMatrix(
                ln_dim_left, ln_dim_right, add_diag=self.flat_config.add_diag
            )
        if self.flat_config.use_o_quant and (
            self.flat_config.a_bits < 16 or self.flat_config.w_bits < 16
        ):
            self.o_trans = SVDSingleTransMatrix(self.module_config.num_attention_heads)
            # vcache_trans carries the v-side factor of the v->o composite
            self.vcache_trans = SVDSingleTransMatrix(self.head_dim)

        if self.flat_config.use_kcache_quant and self.flat_config.k_bits < 16:
            self.kcache_trans = SVDSingleTransMatrix(self.head_dim)
        if (
            self.flat_config.use_vcache_quant
            and self.flat_config.v_bits < 16
            and self.vcache_trans is None
        ):
            self.vcache_trans = SVDSingleTransMatrix(self.head_dim)

    # ------------------------------------------------------------------
    # float forward: used for input capture and fp reference collection
    # ------------------------------------------------------------------
    def _ori_forward_after_ln(self, hidden_states):
        if getattr(self, 'ln_smax', None) is not None:
            self.ln_smax = torch.maximum(
                self.ln_smax,
                hidden_states.reshape(-1, hidden_states.shape[-1])
                .abs()
                .max(0)[0]
                .clone()
                .detach(),
            )
        query_states = self.q_proj._ori_forward(hidden_states)
        key_states = self.k_proj._ori_forward(hidden_states)
        value_states = self.v_proj._ori_forward(hidden_states)
        return query_states, key_states, value_states

    # ------------------------------------------------------------------
    # fake-quant training forward
    # ------------------------------------------------------------------
    def _trans_forward_after_ln(self, hidden_states):
        if self.ln_trans is not None:
            hidden_states = self.ln_trans(hidden_states)
        query_states = self.q_proj(hidden_states, qa_trans=self.ln_trans)
        key_states = self.k_proj(hidden_states, qa_trans=self.ln_trans)
        value_states = self.v_proj(
            hidden_states, qa_trans=self.ln_trans, out_trans=self.vcache_trans
        )
        return query_states, key_states, value_states

    def quant_vcache(self, value_states):
        if self.flat_config.v_bits < 16:
            value_states = self.v_cache_quantizer(value_states)
        return value_states

    def quant_kcache(self, q, k):
        if not self.flat_config.k_bits < 16:
            return q, k
        if self.kcache_trans is not None:
            q = self.kcache_trans(q, inv_t=True)
            k = self.kcache_trans(k)
        k = self.k_cache_quantizer(k).to(q)
        return q, k

    def forward_qkv(self, query_states, key_states, value_states, bsz, q_len, **kwargs):
        _, apply_rotary_pos_emb, eager_attention_forward = _qwen3_attention_helpers()
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

        # kv-cache fake quant happens only during training
        if not self._ori_mode:
            if self.flat_config.use_kcache_quant:
                query_states, key_states = self.quant_kcache(query_states, key_states)
            if self.flat_config.use_vcache_quant:
                value_states = self.quant_vcache(value_states)

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
        attn_implementation = getattr(self.module_config, '_attn_implementation')
        if attn_implementation != 'eager':
            attention_interface = _qwen3_attention_helpers()[0][attn_implementation]

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

        attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()

        # o_proj path: float / composite fake-quant / eval
        if self._ori_mode or (self.o_trans is None and self.vcache_trans is None):
            attn_output = self.o_proj._ori_forward(attn_output)
        else:
            init_shape = attn_output.shape
            attn_shape = [-1, self.module_config.num_attention_heads, self.head_dim]
            attn_output = attn_output.reshape(attn_shape)
            if self.o_trans is None and self.vcache_trans is not None:
                # v-cache-only transform: fold v^-T on the fly, plain quant o_proj
                attn_output = torch.matmul(
                    attn_output,
                    self.vcache_trans.get_matrix(inv_t=True).T.to(attn_output),
                ).reshape(init_shape)
                attn_output = self.o_proj(attn_output)
            else:
                # v->o composite: O^T @ attn, then o_proj with kron(O^-T, V^-T)
                attn_output = torch.matmul(
                    self.o_trans.get_matrix().T.to(attn_output), attn_output
                ).reshape(init_shape)
                if not self._eval_mode:
                    attn_o_og_it = self.o_trans.get_matrix(inv_t=True)
                    attn_v_og_it = self.vcache_trans.get_matrix(inv_t=True)
                    attn_output = self.o_proj(
                        attn_output, qa_trans=[attn_o_og_it, attn_v_og_it]
                    )
                else:
                    attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights

    def forward(self, hidden_states, **kwargs):
        bsz, q_len, _ = hidden_states.size()
        if self._ori_mode:
            query_states, key_states, value_states = self._ori_forward_after_ln(
                hidden_states
            )
        else:
            # both the training forward and the deploy forward route through the
            # transform: y = Q(x*M) @ Q(W*M^-T) keeps its activation half
            # after convert (only the Linear internals change)
            query_states, key_states, value_states = self._trans_forward_after_ln(
                hidden_states
            )
        return self.forward_qkv(
            query_states, key_states, value_states, bsz, q_len, **kwargs
        )

    # ------------------------------------------------------------------
    # calibration / convert hooks
    # ------------------------------------------------------------------
    def set_fake_quant(self, enabled):
        """Enable/disable fake-quant training; calibrator calls this."""
        self._ori_mode = not enabled
        for proj in (self.q_proj, self.k_proj, self.v_proj, self.o_proj):
            proj.set_fake_quant(enabled)

    def init_diag_scale(self, alpha=0.5):
        """Init diag scale from weight/LN-output stats; speeds up convergence."""
        assert hasattr(self, 'ln_smax')
        qkvw_smax = (
            torch.cat(
                [
                    self.q_proj.linear.weight,
                    self.k_proj.linear.weight,
                    self.v_proj.linear.weight,
                ],
                dim=0,
            )
            .abs()
            .max(dim=0)[0]
        )
        if self.ln_trans is not None:
            self.ln_trans.diag_scale.data = get_init_scale(
                qkvw_smax, self.ln_smax, alpha
            )
        self.ln_smax = None

    def reparameterize(self):
        """Convert-time: freeze transforms into matrices, bake W/d @ M^-T."""
        if self._eval_mode:
            return
        for trans in (
            self.ln_trans,
            self.kcache_trans,
            self.vcache_trans,
            self.o_trans,
        ):
            if trans is not None:
                trans.to_eval_mode()
        # bake the weights while use_diag is still True so the weight side
        # carries the diag inverse; the Pass then folds the diag into the
        # preceding Norm (gamma <- gamma * c), keeping the deploy identity:
        #   (W/d @ M^-T) @ (M @ (x*c)) == W @ x
        self.q_proj.reparameterize(qa_trans=self.ln_trans)
        self.k_proj.reparameterize(qa_trans=self.ln_trans)
        self.v_proj.reparameterize(qa_trans=self.ln_trans, out_trans=self.vcache_trans)
        if self.o_trans is not None and self.vcache_trans is not None:
            attn_o_og_it = self.o_trans.get_matrix(inv_t=True)
            attn_v_og_it = self.vcache_trans.get_matrix(inv_t=True)
            self.o_proj.reparameterize(qa_trans=[attn_o_og_it, attn_v_og_it])
        if self.ln_trans is not None:
            self.ln_trans.use_diag = False
        self._eval_mode = True

    def get_external_reparam(self, model=None, name=''):
        """Declare out-of-module side effects for the convert Pass.

        Returns [(host_module, fuse_fn), ...]; ln diag folds into the
        preceding input_layernorm's weight: gamma <- gamma * c."""
        if (
            self.ln_trans is None
            or not self.ln_trans.add_diag
            or not self.ln_trans.use_diag
        ):
            return []
        return [('input_layernorm', self.ln_trans)]
