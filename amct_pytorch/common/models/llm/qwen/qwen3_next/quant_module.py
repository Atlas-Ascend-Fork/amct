# -*- coding: UTF-8 -*-
# Copyright 2025 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
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
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.qwen3_next.modeling_qwen3_next import (
    Qwen3NextAttention,
    Qwen3NextGatedDeltaNet,
    Qwen3NextMLP,
    apply_mask_to_padding_states,
    apply_rotary_pos_emb,
)
from amct_pytorch.common.models.llm.common.quant_apply import QuantGatedMLP, PlainLinear
from amct_pytorch.common.models.llm.common.attention_forward import scaled_dot_product_attention
from amct_pytorch.algorithms.quant import AlgoBuildContext
from amct_pytorch.quantization.modules.quant_linear import QuantLinear
from amct_pytorch.quantization.modules.quant_matmul import QuantizedMatmul
from amct_pytorch.quantization.bit_policy import ensure_bit_policy
from amct_pytorch.quantization.modules.quant_base import ActivationQuantizer, build_algorithms_by_target


class QuantQwen3NextMLP(QuantGatedMLP):
    pass


class QuantQwen3NextLinearAttn(Qwen3NextGatedDeltaNet):
    def __init__(self, quant_args, attn_module: Qwen3NextGatedDeltaNet):
        super().__init__(attn_module.config, attn_module.layer_idx)
        self.quant_args = quant_args
        ensure_bit_policy(quant_args)
        self.enable_attn_linear = "attn-linear" in self.quant_args.quant_target
        self.hidden_size = attn_module.hidden_size
        self.value_dim = attn_module.value_dim

        self.conv1d = attn_module.conv1d
        self.dt_bias = attn_module.dt_bias
        self.A_log = attn_module.A_log
        self.norm = attn_module.norm
        bits = quant_args.bit_policy["attn-linear"]
        out, qkvz, ba = bits["out_proj"], bits["in_proj_qkvz"], bits["in_proj_ba"]
        self.out_proj = QuantLinear(
            quant_args,
            attn_module.out_proj,
            w_bits=out.w,
            name="out_proj") if self.enable_attn_linear else PlainLinear(
            attn_module.out_proj)
        self.in_proj_qkvz = QuantLinear(
            quant_args,
            attn_module.in_proj_qkvz,
            w_bits=qkvz.w,
            name="in_proj_qkvz") if self.enable_attn_linear else PlainLinear(
            attn_module.in_proj_qkvz)
        self.in_proj_ba = QuantLinear(
            quant_args,
            attn_module.in_proj_ba,
            w_bits=ba.w,
            name="in_proj_ba") if self.enable_attn_linear else PlainLinear(
            attn_module.in_proj_ba)
        self.input_transform = None
        self.out_transform = None
        if self.enable_attn_linear:
            self._init_structure_transforms()
            self.inp_afq = ActivationQuantizer(quant_args, qkvz.a)
            self.o_proj_afq = ActivationQuantizer(quant_args, out.a)
        else:
            self.inp_afq = nn.Identity()
            self.o_proj_afq = nn.Identity()

    def forward(
        self,
        hidden_states: torch.Tensor,
        cache_params=None,
        **kwargs,
    ):
        batch_size, seq_len, _ = hidden_states.shape
        if self.input_transform is not None:
            hidden_states = self.input_transform(hidden_states)
        hidden_states = self.inp_afq(hidden_states)

        projected_states_qkvz = self.in_proj_qkvz(hidden_states, structure_transform=self.input_transform)
        projected_states_ba = self.in_proj_ba(hidden_states, structure_transform=self.input_transform)
        query, key, value, z, b, a = self.fix_query_key_value_ordering(projected_states_qkvz, projected_states_ba)
        query, key, value = (x.reshape(x.shape[0], x.shape[1], -1) for x in (query, key, value))

        mixed_qkv = torch.cat((query, key, value), dim=-1).transpose(1, 2)
        mixed_qkv = F.silu(self.conv1d(mixed_qkv)[:, :, :seq_len])

        mixed_qkv = mixed_qkv.transpose(1, 2)
        query, key, value = torch.split(
            mixed_qkv,
            [self.key_dim, self.key_dim, self.value_dim],
            dim=-1,
        )
        query = query.reshape(query.shape[0], query.shape[1], -1, self.head_k_dim)
        key = key.reshape(key.shape[0], key.shape[1], -1, self.head_k_dim)
        value = value.reshape(value.shape[0], value.shape[1], -1, self.head_v_dim)

        beta = b.sigmoid()
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
        if self.num_v_heads // self.num_k_heads > 1:
            query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
            key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)

        core_attn_out, last_recurrent_state = self.chunk_gated_delta_rule(
            query,
            key,
            value,
            g=g,
            beta=beta,
            initial_state=None,
            output_final_state=cache_params is not None,
            use_qk_l2norm_in_kernel=True,
        )

        z_shape_og = z.shape
        core_attn_out = core_attn_out.reshape(-1, core_attn_out.shape[-1])
        z = z.reshape(-1, z.shape[-1])
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(z_shape_og)
        core_attn_out = core_attn_out.reshape(core_attn_out.shape[0], core_attn_out.shape[1], -1)
        if self.out_transform is not None:
            core_attn_out = self.out_transform(core_attn_out)
        core_attn_out = self.o_proj_afq(core_attn_out)

        output = self.out_proj(core_attn_out, structure_transform=self.out_transform)
        return output

    def _init_structure_transforms(self):
        ctx = AlgoBuildContext(matrix_size=128, dim_size=self.hidden_size)
        self.input_transform = build_algorithms_by_target(self.quant_args, "structure", ctx)
        ctx = AlgoBuildContext(matrix_size=128, dim_size=self.value_dim)
        self.out_transform = build_algorithms_by_target(self.quant_args, "structure", ctx)


class QuantQwen3NextAttn(Qwen3NextAttention):
    def __init__(self, quant_args, attn_module: Qwen3NextAttention):
        super().__init__(attn_module.config, attn_module.layer_idx)
        self.quant_args = quant_args
        ensure_bit_policy(quant_args)
        self.enable_attn_linear = "attn-linear" in self.quant_args.quant_target
        self.hidden_size = attn_module.config.hidden_size
        self.value_dim = attn_module.config.num_attention_heads * self.head_dim
        bits = quant_args.bit_policy["attn-linear"]
        q, k, v, o = bits["q_proj"], bits["k_proj"], bits["v_proj"], bits["o_proj"]
        self.q_proj = QuantLinear(
    quant_args,
    attn_module.q_proj,
    w_bits=q.w,
    name="q_proj") if self.enable_attn_linear else PlainLinear(
        attn_module.q_proj)
        self.k_proj = QuantLinear(
    quant_args,
    attn_module.k_proj,
    w_bits=k.w,
    name="k_proj") if self.enable_attn_linear else PlainLinear(
        attn_module.k_proj)
        self.v_proj = QuantLinear(
    quant_args,
    attn_module.v_proj,
    w_bits=v.w,
    name="v_proj") if self.enable_attn_linear else PlainLinear(
        attn_module.v_proj)
        self.o_proj = QuantLinear(
    quant_args,
    attn_module.o_proj,
    w_bits=o.w,
    name="o_proj") if self.enable_attn_linear else PlainLinear(
        attn_module.o_proj)
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
            self.inp_afq = ActivationQuantizer(quant_args, q.a)
            self.o_proj_afq = ActivationQuantizer(quant_args, o.a)
        else:
            self.inp_afq = nn.Identity()
            self.o_proj_afq = nn.Identity()

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        if self.input_transform is not None:
            hidden_states = self.input_transform(hidden_states)
        hidden_states = self.inp_afq(hidden_states)

        query_states, gate = torch.chunk(
    self.q_proj(
        hidden_states, structure_transform=self.input_transform).view(
            *input_shape, -1, self.head_dim * 2), 2, dim=-1)
        gate = gate.reshape(*input_shape, -1)

        query_states = self.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(
    self.k_proj(
        hidden_states,
        structure_transform=self.input_transform).view(hidden_shape)).transpose(
            1,
             2)
        value_states = self.v_proj(
    hidden_states,
    structure_transform=self.input_transform).view(hidden_shape).transpose(
        1,
         2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        attn_output = scaled_dot_product_attention(
            self,
            query_states,
            key_states,
            value_states,
            attn_mask=None,
            is_causal=True,
            dropout_p=0.0,
            scale=self.scaling,
        )
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = attn_output * torch.sigmoid(gate)
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
