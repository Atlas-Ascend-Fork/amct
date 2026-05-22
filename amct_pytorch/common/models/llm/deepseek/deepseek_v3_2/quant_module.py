# coding=utf-8
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
#
# Adapted from
# https://huggingface.co/deepseek-ai/DeepSeek-V3.2/blob/main/inference/model.py
#
# Copyright (c) 2023 DeepSeek.
#
# This file is part of [OtherProject], which is released under the MIT License.
# See the LICENSE file in the root directory of this source tree
# or at https://opensource.org/licenses/MIT for details.
from typing import Tuple, Optional, Literal
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear
from torch.nn import LayerNorm
from einops import rearrange
from amct_pytorch.algorithms.quant import AlgoBuildContext
from amct_pytorch.quantization.modules.quant_base import ActivationQuantizer, build_algorithms_by_target
from amct_pytorch.quantization.modules.quant_linear import QuantLinear
from amct_pytorch.quantization.bit_policy import ensure_bit_policy
from amct_pytorch.quantization.modules.quant_matmul import QuantizedMatmul
from amct_pytorch.common.models.llm.common.quant_apply import QuantGatedMLP, PlainLinear
from amct_pytorch.common.models.llm.deepseek.deepseek_v3_2.modeling.modeling_deepseek_v3_2 import (
    rotate_activation, fp32_index)
from amct_pytorch.common.models.llm.deepseek.deepseek_v3_2.modeling.modeling_deepseek_v3_2 import (
    DeepseekV3Attention, DeepseekV3MLP, Indexer, apply_rotary_emb,
)


def _linear_with_structure(hidden_states, linear, structure_transform=None, name=None):
    weight = linear.weight
    if structure_transform is not None:
        weight = structure_transform(weight, inv_t=True, name=name)
    return F.linear(hidden_states, weight, linear.bias)


class QuantIndexer(torch.nn.Module):
    def __init__(self, quant_args, module: Indexer):
        super().__init__()
        bits = quant_args.bit_policy["attn-linear"]
        wq_b = bits["wq_b"]
        self.enable_attn_linear = "attn-linear" in quant_args.quant_target
        self.enable_attn_cache = "attn-cache" in quant_args.quant_target
        self.q_lora_rank = module.q_lora_rank
        self.wq_b = QuantLinear(
            quant_args,
            module.wq_b,
            w_bits=wq_b.w,
            name="wq_b") if self.enable_attn_linear else PlainLinear(
            module.wq_b)

        self.wk = module.wq_b
        self.k_norm = module.k_norm
        self.weights_proj = module.weights_proj
        if self.enable_attn_linear:
            self._init_structure_transforms()
            self.wq_b_afq = ActivationQuantizer(quant_args, q_b_proj.a)
        else:
            self.wq_b_afq = nn.Identity()

    def forward(self, x: torch.Tensor, qr: torch.Tensor, start_pos: int, 
                freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]):
        bsz, seqlen, _ = x.size()
        end_pos = start_pos + seqlen
        if self.input_transform is not None:
            qr = self.input_transform(qr)
        qr = self.q_b_proj_afq(qr)
        q = self.wq_b(qr)
        q = q.view(bsz, seqlen, self.n_heads, self.head_dim)
        q_pe, q_nope = torch.split(q, [self.rope_head_dim, self.head_dim - self.rope_head_dim], dim=-1)
        # rope in indexer is not interleaved
        q_pe = apply_rotary_emb(q_pe, freqs_cis, False)
        q = torch.cat([q_pe, q_nope], dim=-1)
        k = self.wk(x)
        k = self.k_norm(k)
        k_pe, k_nope = torch.split(k, [self.rope_head_dim, self.head_dim - self.rope_head_dim], dim=-1)
        # rope in indexer is not interleaved
        k_pe = apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis, False).squeeze(2)
        k = torch.cat([k_pe, k_nope], dim=-1)
        q = rotate_activation(q)
        k = rotate_activation(k)
        weights = self.weights_proj(x.float()) * self.n_heads ** -0.5
        weights = weights.unsqueeze(-1) * self.softmax_scale
        index_score = fp32_index(q, k.unsqueeze(2), weights)
        if mask is not None:
            index_score += mask
        topk_indices = index_score.topk(min(self.index_topk, end_pos), dim=-1)[1]
        topk_indices_ = topk_indices.clone()
        assert torch.all(topk_indices == topk_indices_), f"{topk_indices=} {topk_indices_=}"
        return topk_indices

    def _init_structure_transforms(self):
        ctx = AlgoBuildContext(matrix_size=128, dim_size=self.q_lora_rank)
        self.input_transform = build_algorithms_by_target(self.quant_args, "structure", ctx)


class QuantDeepseekV3Attention(nn.Module):
    def __init__(self, quant_args, module: DeepseekV3Attention):
        super().__init__()
        self.quant_args = quant_args
        ensure_bit_policy(quant_args)
        bits = quant_args.bit_policy["attn-linear"]
        q_b_proj, o_proj = bits["q_b_proj"], bits["o_proj"]
        self.enable_attn_linear = "attn-linear" in quant_args.quant_target
        self.enable_attn_cache = "attn-cache" in quant_args.quant_target
        self.q_lora_rank = module.q_lora_rank
        self.value_dim = module.n_heads * module.v_head_dim
        self.q_a_proj = module.q_a_proj
        self.q_a_layernorm = module.q_a_layernorm
        self.q_b_proj = QuantLinear(
            quant_args,
            module.q_b_proj,
            w_bits=q_b_proj.w,
            name="q_b_proj") if self.enable_attn_linear else PlainLinear(
            module.q_b_proj)
        self.kv_a_proj_with_mqa = module.kv_a_proj_with_mqa
        self.kv_a_layernorm = module.kv_a_layernorm
        self.kv_b_proj = module.kv_b_proj
        self.o_proj = QuantLinear(
            quant_args,
            module.o_proj,
            w_bits=o_proj.w,
            name="o_proj") if self.enable_attn_linear else PlainLinear(
            module.o_proj)
        self.freqs_cis = module.freqs_cis
        self.indexer = QuantIndexer(quant_args, module.indexer)

        if self.enable_attn_linear:
            self._init_structure_transforms()
            self.q_b_proj_afq = ActivationQuantizer(quant_args, q_b_proj.a)
            self.o_proj_afq = ActivationQuantizer(quant_args, o_proj.a)
        else:
            self.q_b_proj_afq = nn.Identity()
            self.o_proj_afq = nn.Identity()

    def forward(self, x: torch.Tensor, start_pos: int = 0):
        bsz, seqlen, _ = x.size()
        end_pos = start_pos + seqlen
        mask = torch.full((seqlen, seqlen), float("-inf"), device=x.device).triu_(1) if seqlen > 1 else None
        freqs_cis = self.freqs_cis[start_pos:end_pos]
        qr = self.q_a_layernorm(self.q_a_proj(x))
        if self.input_transform is not None:
            qr = self.input_transform(qr)
        qr = self.q_b_proj_afq(qr)
        q = self.q_b_proj(qr)
        q = q.view(bsz, seqlen, self.n_local_heads, self.qk_head_dim)
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        q_pe = apply_rotary_emb(q_pe, freqs_cis)
        kv = self.kv_a_proj_with_mqa(x)
        kv, k_pe = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        kv = self.kv_a_layernorm(kv)
        k_pe = apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis)
        q = torch.cat([q_nope, q_pe], dim=-1)
        kv = self.kv_b_proj(kv)
        kv = kv.view(bsz, seqlen, self.n_local_heads, self.qk_nope_head_dim + self.v_head_dim)
        k_nope, v = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        k = torch.cat([k_nope, k_pe.expand(-1, -1, self.n_local_heads, -1)], dim=-1)
        scores = torch.einsum("bshd,bthd->bsht", q, k).mul_(self.softmax_scale)

        # indexer
        topk_indices = self.indexer(x, qr, start_pos, freqs_cis, mask)
        index_mask = torch.full((bsz, seqlen, seqlen), float("-inf"), device=x.device).scatter_(-1, topk_indices, 0)
        index_mask += mask
        scores += index_mask.unsqueeze(2)

        scores = scores.softmax(dim=-1)
        x = torch.einsum("bsht,bthd->bshd", scores, v)
        x = x.flatten(2)
        if self.out_transform is not None:
            x = self.out_transform(x)
        x = self.o_proj_afq(x)
        x = self.o_proj(x)
        return x

    def _init_structure_transforms(self):
        ctx = AlgoBuildContext(matrix_size=128, dim_size=self.q_lora_rank)
        self.input_transform = build_algorithms_by_target(self.quant_args, "structure", ctx)
        ctx = AlgoBuildContext(matrix_size=128, dim_size=self.value_dim)
        self.out_transform = build_algorithms_by_target(self.quant_args, "structure", ctx)


class QuantDeepseekV3MLP(QuantGatedMLP):
    pass
