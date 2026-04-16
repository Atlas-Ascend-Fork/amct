# -*- coding: UTF-8 -*-
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
import math
import torch
from torch import nn
import torch.nn.functional as F
import transfermer
from amct_pytorch.utils.vars import HIFLOAT8
from amct_pytorch.utils.log import LOGGER


class DeepseekV3AttentionQuant(BaseQuantizeModule):
    """
    Function: Customized torch.nn.Module of the GPTQuant class.
    APIs: forward.
    """
    def __init__(self,
                 ori_module,
                 layer_name,
                 quant_config):
        """
        Function: init objective.
        Args:
        ori_module: torch module. Linear.
        layer_name: ori_module's name.
        quant_config: quantization parameters.
        """
        super().__init__(ori_module, layer_name, quant_config)
        self.config = ori_module.config
        self.layer_idx = ori_module.layer_idx
        self.num_key_value_groups = ori_module.num_key_value_groups
        self.attention_dropout = ori_module.attention_dropout
        self.num_heads = ori_module.num_heads

        self.q_lora_rank = ori_module.q_lora_rank
        self.qk_rope_head_dim = ori_module.qk_rope_head_dim
        self.kv_lora_rank = ori_module.kv_lora_rank
        self.v_head_dim = ori_module.v_head_dim
        self.qk_nope_head_dim = ori_module.qk_nope_head_dim
        self.qk_head_dim = ori_module.qk_head_dim

        self.is_causal = ori_module.is_causal

        self.q_proj = ori_module.q_proj if ori_module.q_proj else None
        self.q_a_proj = ori_module.q_a_proj if ori_module.q_a_proj else None
        self.q_a_layernorm = ori_module.q_a_layernorm if ori_module.q_a_layernorm else None
        self.q_b_proj = ori_module.q_b_proj if ori_module.q_b_proj else None

        self.kv_a_proj_with_mqa = ori_module.kv_a_proj_with_mqa
        self.kv_a_layernorm = ori_module.kv_a_layernorm
        self.kv_b_proj = ori_module.kv_b_proj
        
        self.o_proj = ori_module.o_proj

        self.scaling = ori_module.scaling

    @torch.no_grad()
    def forward(
            self,
            hidden_states,
            position_embeddings,
            attention_mask,
            past_key_values,
            cu_seq_lens_q,
            cu_seq_lens_k,
            max_length_q,
            max_length_k,
        ):
        batch_size, seq_length = hidden_states.shape[:-1]
        query_shape = (batch_size, seq_length, -1, self.qk_head_dim)
        key_shape = (batch_size, seq_length, -1, self.qk_nope_head_dim + self.v_head_dim)

        if self.q_lora_rank is None:
            q_states = self.q_proj(hidden_states)
        else:
            q_states = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
        q_states = q_states.view(query_shape).transpose(1, 2)
        q_pass, q_rot = torch.split(q_states, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        k_pass, k_rot = torch.split(compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)

        k_pass = self.kv_b_proj(self.kv_a_layernorm(k_pass)).view(key_shape).transpose(1, 2)
        k_pass, value_states = torch.split(k_pass, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        k_rot = k_rot.view(batch_size, 1, seq_length, self.qk_rope_head_dim)

        cos, sin = position_embeddings
        if self.config.rope_interleave:  # support using interleaved weights for efficiency
            q_rot, k_rot = apply_rotary_pos_emb_interleave(q_rot, k_rot, cos, sin)
        else:
            q_rot, k_rot = apply_rotary_pos_emb(q_rot, k_rot, cos, sin)
        k_rot = k_rot.expand(*k_pass.shape[:-1], -1)

        query_states = torch.cat((q_pass, q_rot), dim=-1)
        key_states = torch.cat((k_pass, k_rot), dim=-1)

        if past_key_values is not None:
            quantized_key = 
            quantized_value = 
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        if is_flash_attention_requested(self.config) and self.qk_head_dim != self.v_head_dim:
            value_states = F.pad(value_states, [0, self.qk_head_dim - self.v_head_dim])

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )

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

        if is_flash_attention_requested(self.config) and self.qk_head_dim != self.v_head_dim:
            attn_output = attn_output[:, :, :, : self.v_head_dim]

        attn_output = attn_output.reshape(batch_size, seq_length, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights