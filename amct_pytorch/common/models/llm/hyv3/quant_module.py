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

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.models.hy_v3.modeling_hy_v3 import (
    HYV3Attention,
    HYV3MLP,
    HYV3MoE,
    apply_rotary_pos_emb,
)

from amct_pytorch.common.models.llm.common.attention_forward import scaled_dot_product_attention
from amct_pytorch.common.models.llm.common.quant_apply import QuantGatedMLP, PlainLinear, build_no_algo_args
from amct_pytorch.common.models.llm.common.moe_unpack import GatedExpertView
from amct_pytorch.common.models.llm.qwen.moe_common import QuantGatedExperts
from amct_pytorch.algorithms.quant import AlgoBuildContext
from amct_pytorch.quantization.modules.quant_base import ActivationQuantizer, build_algorithms_by_target
from amct_pytorch.quantization.modules.quant_linear import QuantLinear
from amct_pytorch.quantization.modules.quant_matmul import QuantizedMatmul
from amct_pytorch.quantization.bit_policy import ensure_bit_policy


class QuantHYV3MLP(QuantGatedMLP):
    pass


class QuantHYV3Attn(nn.Module):
    def __init__(self, quant_args, attn_module: HYV3Attention):
        super().__init__()
        self.quant_args = quant_args
        ensure_bit_policy(quant_args)
        self.enable_attn_linear = "attn-linear" in self.quant_args.quant_target
        self.enable_attn_cache = "attn-cache" in self.quant_args.quant_target
        
        self.config = attn_module.config
        self.layer_idx = attn_module.layer_idx
        self.head_dim = attn_module.head_dim
        self.num_key_value_groups = attn_module.num_key_value_groups
        self.scaling = attn_module.scaling
        self.attention_dropout = attn_module.attention_dropout
        self.is_causal = attn_module.is_causal
        
        bits = quant_args.bit_policy["attn-linear"]
        q_bits = bits["q_proj"]
        k_bits = bits["k_proj"]
        v_bits = bits["v_proj"]
        o_bits = bits["o_proj"]
        
        self.q_proj = QuantLinear(
            quant_args, attn_module.q_proj, w_bits=q_bits.w, name="q_proj"
        ) if self.enable_attn_linear else PlainLinear(attn_module.q_proj)
        
        self.k_proj = QuantLinear(
            quant_args, attn_module.k_proj, w_bits=k_bits.w, name="k_proj"
        ) if self.enable_attn_linear else PlainLinear(attn_module.k_proj)
        
        self.v_proj = QuantLinear(
            quant_args, attn_module.v_proj, w_bits=v_bits.w, name="v_proj"
        ) if self.enable_attn_linear else PlainLinear(attn_module.v_proj)
        
        self.o_proj = QuantLinear(
            quant_args, attn_module.o_proj, w_bits=o_bits.w, name="o_proj"
        ) if self.enable_attn_linear else PlainLinear(attn_module.o_proj)
        
        self.q_norm = attn_module.q_norm
        self.k_norm = attn_module.k_norm
        
        self.qk_matmul = QuantizedMatmul(
            self.quant_args,
            l_bits=quant_args.bit_policy.cache_bits("q"),
            r_bits=quant_args.bit_policy.cache_bits("k"),
            left_dim=self.head_dim,
            right_dim=self.head_dim,
        )
        
        self.pv_matmul = QuantizedMatmul(
            self.quant_args,
            l_bits=quant_args.bit_policy.cache_bits("p"),
            r_bits=quant_args.bit_policy.cache_bits("v"),
            right_dim=self.head_dim,
        )
        
        self.input_transform = None
        self.out_transform = None
        
        if self.enable_attn_linear:
            self._init_structure_transforms()
            self.inp_afq = ActivationQuantizer(quant_args, q_bits.a)
            self.o_proj_afq = ActivationQuantizer(quant_args, o_bits.a)
        else:
            self.inp_afq = nn.Identity()
            self.o_proj_afq = nn.Identity()

    def _init_structure_transforms(self):
        ctx = AlgoBuildContext(matrix_size=128, dim_size=self.head_dim)
        self.input_transform = build_algorithms_by_target(self.quant_args, "structure", ctx)
        self.out_transform = build_algorithms_by_target(self.quant_args, "structure", ctx)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        if self.input_transform is not None:
            hidden_states = self.input_transform(hidden_states)
        
        hidden_states = self.inp_afq(hidden_states)
        
        query_states = self.q_norm(
            self.q_proj(hidden_states, structure_transform=self.input_transform).view(hidden_shape)
        ).transpose(1, 2)
        key_states = self.k_norm(
            self.k_proj(hidden_states, structure_transform=self.input_transform).view(hidden_shape)
        ).transpose(1, 2)
        value_states = self.v_proj(
            hidden_states, structure_transform=self.input_transform
        ).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        attn_output = scaled_dot_product_attention(
            self,
            query_states,
            key_states,
            value_states,
            attn_mask=None,
            is_causal=True,
            dropout_p=self.attention_dropout if self.training else 0.0,
            scale=self.scaling,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        if self.out_transform is not None:
            attn_output = self.out_transform(attn_output)
        attn_output = self.o_proj_afq(attn_output)
        attn_output = self.o_proj(attn_output, structure_transform=self.out_transform)

        return attn_output, None


class QuantHYV3Experts(nn.Module):
    def __init__(self, args, experts_module, group: str = "moe.routed"):
        super().__init__()
        self.args = args
        self.packed_experts = experts_module
        self.num_experts = experts_module.num_experts
        self.group = group
        self.expert_modules = nn.ModuleList([
            QuantHYV3MLP(
                args,
                GatedExpertView(
                    self.packed_experts,
                    expert_idx,
                    hidden_attr="hidden_dim",
                    intermediate_attr="intermediate_dim",
                    materialize=False
                ),
                group=group
            )
            for expert_idx in range(self.num_experts)
        ])
        self.act_fn = experts_module.act_fn

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        final_hidden_states = torch.zeros_like(hidden_states)
        if top_k_index.numel() == 0:
            return final_hidden_states

        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero(as_tuple=False)

        for expert_idx_tensor in expert_hit:
            expert_idx = int(expert_idx_tensor.reshape(-1)[0].item())
            if expert_idx == self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]
            current_hidden_states = self.expert_modules[expert_idx](current_state)
            current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))

        return final_hidden_states


class QuantHYV3MoE(nn.Module):
    def __init__(self, args, moe_module: HYV3MoE):
        super().__init__()
        self.args = args
        self.top_k = moe_module.top_k
        self.gate = moe_module.gate
        self.experts = QuantGatedExperts(args, moe_module.experts, group="moe.routed")
        self.register_buffer("e_score_correction_bias", moe_module.e_score_correction_bias)
        self.enable_moe_fp32_combine = moe_module.enable_moe_fp32_combine
        shared_args = build_no_algo_args(args)
        self.shared_experts = QuantHYV3MLP(shared_args, moe_module.shared_experts, group="moe.shared")

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)

        _, top_k_weights, top_k_index = self.gate(hidden_states, self.e_score_correction_bias)
        routed_output = self.experts(hidden_states, top_k_index, top_k_weights)

        if self.enable_moe_fp32_combine:
            hidden_states = (routed_output.float() + self.shared_experts(hidden_states).float()).to(
                hidden_states.dtype
            )
        else:
            hidden_states = routed_output + self.shared_experts(hidden_states)

        return hidden_states.reshape(batch_size, seq_len, hidden_dim)