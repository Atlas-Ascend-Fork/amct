# -*- coding: UTF-8 -*-
# Copyright 2025 Meituan and the HuggingFace Inc. team. All rights reserved.
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
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
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.longcat_flash.modeling_longcat_flash import (
    LongcatFlashDecoderLayer,
    LongcatFlashMLA,
    LongcatFlashMLP,
    apply_rotary_pos_emb_interleave,
)
from amct_pytorch.algorithms.quant import AlgoBuildContext
from amct_pytorch.common.models.llm.common.attention_forward import scaled_dot_product_attention
from amct_pytorch.common.models.llm.common.quant_apply import (
    PlainLinear,
    QuantGatedMLP,
)
from amct_pytorch.quantization.bit_policy import ensure_bit_policy
from amct_pytorch.quantization.modules.quant_base import ActivationQuantizer, build_algorithms_by_target
from amct_pytorch.quantization.modules.quant_linear import QuantLinear
from amct_pytorch.quantization.modules.quant_matmul import QuantizedMatmul


class QuantLongcatMLP(QuantGatedMLP):
    pass


class LongcatPackedExpertLinearView:
    def __init__(self, experts_module, expert_idx: int, weight_name: str,
                 start: int | None = None, end: int | None = None):
        super().__init__()
        self.experts_module = experts_module
        self.expert_idx = expert_idx
        self.weight_name = weight_name
        self.start = start
        self.end = end
        self.bias = None

    @property
    def weight(self):
        weight_tensor = getattr(self.experts_module, self.weight_name)[self.expert_idx]
        if self.start is None and self.end is None:
            return weight_tensor
        return weight_tensor[self.start:self.end]


class LongcatPackedExpertView(nn.Module):
    def __init__(self, experts_module, expert_idx: int):
        super().__init__()
        self.hidden_size = experts_module.hidden_size
        self.intermediate_size = experts_module.intermediate_size
        self.act_fn = experts_module.act_fn
        self.gate_proj = LongcatPackedExpertLinearView(
            experts_module,
            expert_idx,
            "gate_up_proj",
            0,
            self.intermediate_size,
        )
        self.up_proj = LongcatPackedExpertLinearView(
            experts_module,
            expert_idx,
            "gate_up_proj",
            self.intermediate_size,
            None,
        )
        self.down_proj = LongcatPackedExpertLinearView(
            experts_module,
            expert_idx,
            "down_proj",
        )


class LongcatUnpackedExperts(nn.Module):
    def __init__(self, args, experts_module):
        super().__init__()
        self.packed_experts = experts_module
        self.num_routed_experts = experts_module.num_routed_experts
        self.total_experts = experts_module.total_experts
        expert_modules = [
            QuantLongcatMLP(args, LongcatPackedExpertView(experts_module, expert_idx))
            for expert_idx in range(self.num_routed_experts)
        ]
        expert_modules.extend(nn.Identity() for _ in range(self.total_experts - self.num_routed_experts))
        self.expert_modules = nn.ModuleList(expert_modules)

    def forward(self, hidden_states: torch.Tensor, top_k_index: torch.Tensor,
                top_k_weights: torch.Tensor) -> torch.Tensor:
        final_hidden_states = torch.zeros_like(hidden_states)
        if top_k_index.numel() == 0:
            return final_hidden_states

        expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.total_experts).permute(2, 1, 0)
        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero(as_tuple=False)

        for expert_idx_tensor in expert_hit:
            expert_idx = int(expert_idx_tensor.reshape(-1)[0].item())
            selection_idx, token_idx = torch.where(expert_mask[expert_idx])
            if token_idx.numel() == 0:
                continue

            current_state = hidden_states[token_idx]
            current_hidden_states = self.expert_modules[expert_idx](current_state)
            current_hidden_states = current_hidden_states * top_k_weights[token_idx, selection_idx, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))

        return final_hidden_states


class QuantLongcatMLA(LongcatFlashMLA):
    def __init__(self, quant_args, attn_module: LongcatFlashMLA):
        super().__init__(attn_module.config, attn_module.layer_idx)
        self.quant_args = quant_args
        ensure_bit_policy(quant_args)
        self.enable_attn_linear = "attn-linear" in self.quant_args.quant_target
        self.hidden_size = attn_module.config.hidden_size
        self.value_dim = attn_module.num_heads * attn_module.v_head_dim

        bits = quant_args.bit_policy["attn-linear"]
        kv_a, kv_b, o = bits["kv_a_proj_with_mqa"], bits["kv_b_proj"], bits["o_proj"]
        if hasattr(attn_module, "q_proj"):
            q = bits["q_proj"]
            self.q_proj = QuantLinear(
    quant_args,
    attn_module.q_proj,
    w_bits=q.w,
    name="q_proj") if self.enable_attn_linear else PlainLinear(
        attn_module.q_proj)
        else:
            q_a, q_b = bits["q_a_proj"], bits["q_b_proj"]
            self.q_a_proj = QuantLinear(
    quant_args,
    attn_module.q_a_proj,
    w_bits=q_a.w,
    name="q_a_proj") if self.enable_attn_linear else PlainLinear(
        attn_module.q_a_proj)
            self.q_a_layernorm = attn_module.q_a_layernorm
            self.q_b_proj = QuantLinear(
    quant_args,
    attn_module.q_b_proj,
    w_bits=q_b.w,
    name="q_b_proj") if self.enable_attn_linear else PlainLinear(
        attn_module.q_b_proj)

        self.kv_a_proj_with_mqa = (
            QuantLinear(quant_args, attn_module.kv_a_proj_with_mqa, w_bits=kv_a.w, name="kv_a_proj_with_mqa")
            if self.enable_attn_linear else PlainLinear(attn_module.kv_a_proj_with_mqa)
        )
        self.kv_a_layernorm = attn_module.kv_a_layernorm
        self.kv_b_proj = QuantLinear(
    quant_args,
    attn_module.kv_b_proj,
    w_bits=kv_b.w,
    name="kv_b_proj") if self.enable_attn_linear else PlainLinear(
        attn_module.kv_b_proj)
        self.o_proj = QuantLinear(
    quant_args,
    attn_module.o_proj,
    w_bits=o.w,
    name="o_proj") if self.enable_attn_linear else PlainLinear(
        attn_module.o_proj)
        self.qk_matmul = QuantizedMatmul(
            self.quant_args,
            l_bits=quant_args.bit_policy.cache_bits("q"),
            r_bits=quant_args.bit_policy.cache_bits("k"),
            left_dim=self.qk_head_dim,
            right_dim=self.qk_head_dim,
        )
        self.pv_matmul = QuantizedMatmul(
            self.quant_args,
            l_bits=quant_args.bit_policy.cache_bits("p"),
            r_bits=quant_args.bit_policy.cache_bits("v"),
            right_dim=self.v_head_dim,
        )
        self.input_transform = None
        self.out_transform = None
        if self.enable_attn_linear:
            self._init_structure_transforms()
            self.inp_afq = ActivationQuantizer(quant_args, kv_a.a)
            self.o_proj_afq = ActivationQuantizer(quant_args, o.a)
        else:
            self.inp_afq = nn.Identity()
            self.o_proj_afq = nn.Identity()

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values=None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        batch_size, seq_length = hidden_states.shape[:-1]
        query_shape = (batch_size, seq_length, -1, self.qk_head_dim)
        key_shape = (batch_size, seq_length, -1, self.qk_nope_head_dim + self.v_head_dim)
        if self.input_transform is not None:
            hidden_states = self.input_transform(hidden_states)
        hidden_states = self.inp_afq(hidden_states)

        if hasattr(self, "q_proj"):
            q_states = self.q_proj(
    hidden_states,
    structure_transform=self.input_transform).view(query_shape).transpose(
        1,
         2)
        else:
            q_states = self.q_b_proj(
    self.q_a_layernorm(
        self.q_a_proj(
            hidden_states,
             structure_transform=self.input_transform)))
            q_states = q_states.view(query_shape).transpose(1, 2)

        q_pass, q_rot = torch.split(q_states, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        compressed_kv = self.kv_a_proj_with_mqa(hidden_states, structure_transform=self.input_transform)
        k_pass, k_rot = torch.split(compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        k_pass = self.kv_a_layernorm(k_pass)

        q_pass = q_pass * self.mla_scale_q_lora
        q_rot = q_rot * self.mla_scale_q_lora
        k_pass = k_pass * self.mla_scale_kv_lora

        k_pass = self.kv_b_proj(k_pass).view(key_shape).transpose(1, 2)
        k_pass, value_states = torch.split(k_pass, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        k_rot = k_rot.view(batch_size, 1, seq_length, self.qk_rope_head_dim)
        cos, sin = position_embeddings
        q_rot, k_rot = apply_rotary_pos_emb_interleave(q_rot, k_rot, cos, sin)
        k_rot = k_rot.expand(*k_pass.shape[:-1], -1)

        query_states = torch.cat((q_pass, q_rot), dim=-1)
        key_states = torch.cat((k_pass, k_rot), dim=-1)

        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attn_output = scaled_dot_product_attention(
            self,
            query_states,
            key_states,
            value_states,
            attn_mask=None,
            is_causal=True,
            dropout_p=0.0 if not self.training else self.attention_dropout,
            scale=self.scaling,
        )

        attn_output = attn_output.reshape(batch_size, seq_length, -1).contiguous()
        if self.out_transform is not None:
            attn_output = self.out_transform(attn_output)
        attn_output = self.o_proj_afq(attn_output)
        attn_output = self.o_proj(attn_output, structure_transform=self.out_transform)
        return attn_output, None

    def _init_structure_transforms(self):
        ctx = AlgoBuildContext(matrix_size=128, dim_size=self.hidden_size)
        self.input_transform = build_algorithms_by_target(self.quant_args, "structure", ctx)
        ctx = AlgoBuildContext(matrix_size=128, dim_size=self.value_dim)
        self.out_transform = build_algorithms_by_target(self.quant_args, "structure", ctx)
