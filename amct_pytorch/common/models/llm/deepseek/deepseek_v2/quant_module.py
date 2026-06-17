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
# https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite/blob/main/modeling_deepseek.py
# Attention forward derived from DeepSeek-V2 MLA; Indexer removed for V2-Lite.
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from amct_pytorch.quantization.modules.quant_base import ActivationQuantizer, build_algorithms_by_target
from amct_pytorch.quantization.modules.quant_linear import QuantLinear
from amct_pytorch.quantization.bit_policy import ensure_bit_policy
from amct_pytorch.algorithms.quant import AlgoBuildContext
from amct_pytorch.common.models.llm.common.quant_apply import QuantGatedMLP, PlainLinear
from amct_pytorch.common.models.llm.deepseek.deepseek_v2.modeling.modeling_deepseek import (
    DeepseekV2Attention, apply_rotary_emb,
)


class QuantDeepseekV2Attention(nn.Module):
    """Quantized DeepSeek V2 Attention (No Indexer)."""

    def __init__(self, quant_args, module: DeepseekV2Attention):
        super().__init__()
        self.quant_args = quant_args
        ensure_bit_policy(quant_args)
        bits = quant_args.bit_policy["attn-linear"]
        self.enable_attn_linear = "attn-linear" in quant_args.quant_target
        self.enable_attn_cache = "attn-cache" in quant_args.quant_target

        self._copy_attn_dims(module)
        self._build_q_proj(quant_args, module, bits)

        # KV projection (MLA) kept float for now, same as DeepSeek-V3.2
        self.kv_a_proj_with_mqa = module.kv_a_proj_with_mqa
        self.kv_a_layernorm = module.kv_a_layernorm
        self.kv_b_proj = module.kv_b_proj

        self.o_proj = self._wrap_linear(quant_args, module.o_proj, bits["o_proj"].w, "o_proj")
        self.register_buffer("freqs_cis", module.freqs_cis, persistent=False)
        self.softmax_scale = module.softmax_scale

        # default None so forward stays valid when attn-linear is disabled
        self.input_transform = None
        self.out_transform = None
        if self.enable_attn_linear:
            self._init_structure_transforms()
        self._build_act_quantizers(quant_args, bits)

    def _wrap_linear(self, quant_args, linear, w_bits, name):
        if self.enable_attn_linear:
            return QuantLinear(quant_args, linear, w_bits=w_bits, name=name)
        return PlainLinear(linear)

    def _copy_attn_dims(self, module):
        self.q_lora_rank = module.q_lora_rank
        self.value_dim = module.n_heads * module.v_head_dim
        self.n_heads = module.n_heads
        self.qk_nope_head_dim = module.qk_nope_head_dim
        self.qk_rope_head_dim = module.qk_rope_head_dim
        self.qk_head_dim = module.qk_head_dim
        self.v_head_dim = module.v_head_dim
        self.kv_lora_rank = module.kv_lora_rank

    def _build_q_proj(self, quant_args, module, bits):
        # query projection: direct q_proj, or LoRA q_a_proj + q_b_proj when q_lora_rank set
        if self.q_lora_rank is None:
            self.q_proj = self._wrap_linear(quant_args, module.q_proj, bits["q_proj"].w, "q_proj")
        else:
            self.q_a_proj = module.q_a_proj
            self.q_a_layernorm = module.q_a_layernorm
            self.q_b_proj = self._wrap_linear(quant_args, module.q_b_proj, bits["q_b_proj"].w, "q_b_proj")

    def _build_act_quantizers(self, quant_args, bits):
        q_key = "q_proj" if self.q_lora_rank is None else "q_b_proj"
        for key, attr in ((q_key, q_key + "_afq"), ("o_proj", "o_proj_afq")):
            afq = ActivationQuantizer(quant_args, bits[key].a) if self.enable_attn_linear else nn.Identity()
            setattr(self, attr, afq)

    def forward(self, x: torch.Tensor, start_pos: int = 0):
        bsz, seqlen, _ = x.size()
        end_pos = start_pos + seqlen

        # Causal mask - shape (seqlen, seqlen), will be broadcast properly
        mask = None
        if seqlen > 1:
            mask = torch.full((seqlen, seqlen), float("-inf"), device=x.device, dtype=x.dtype).triu_(1)

        # Rotary embeddings
        freqs_cis = self.freqs_cis[start_pos:end_pos]

        # query projection
        if self.q_lora_rank is None:
            # Direct q_proj
            if self.input_transform is not None:
                x = self.input_transform(x)
            x = self.q_proj_afq(x)
            q = self.q_proj(x)
        else:
            # LoRA q_a_proj + q_b_proj
            qr = self.q_a_layernorm(self.q_a_proj(x))
            if self.input_transform is not None:
                qr = self.input_transform(qr)
            qr = self.q_b_proj_afq(qr)
            q = self.q_b_proj(qr)

        q = q.view(bsz, seqlen, self.n_heads, self.qk_head_dim)
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        q_pe = apply_rotary_emb(q_pe, freqs_cis)

        # KV (keep float)
        kv = self.kv_a_proj_with_mqa(x)
        kv, k_pe = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        kv = self.kv_a_layernorm(kv)
        k_pe = k_pe.unsqueeze(2)  # -> bsz, seqlen, 1, qk_rope_head_dim
        k_pe = apply_rotary_emb(k_pe, freqs_cis)  # -> bsz, seqlen, 1, qk_rope_head_dim
        k_pe = k_pe.expand(-1, -1, self.n_heads, -1)  # -> bsz, seqlen, n_heads, qk_rope_head_dim

        kv = self.kv_b_proj(kv)
        kv = kv.view(bsz, seqlen, self.n_heads, self.qk_nope_head_dim + self.v_head_dim)
        k_nope, v = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        # Assemble query and key
        q = torch.cat([q_nope, q_pe], dim=-1)  # -> bsz, seqlen, n_heads, qk_head_dim
        k = torch.cat([k_nope, k_pe], dim=-1)  # -> bsz, seqlen, n_heads, qk_head_dim

        # Attention scores: einsum produces (bsz, n_heads, seqlen, seqlen) for proper mask broadcast
        scores = torch.einsum("bshd,bthd->bhst", q, k).mul_(self.softmax_scale)
        if mask is not None:
            scores = scores + mask  # mask (seqlen, seqlen) broadcasts to (bsz, n_heads, seqlen, seqlen)

        scores = scores.softmax(dim=-1)
        x = torch.einsum("bhst,bthd->bshd", scores, v)
        x = x.flatten(2)

        # Output projection
        if self.out_transform is not None:
            x = self.out_transform(x)
        x = self.o_proj_afq(x)
        x = self.o_proj(x)

        return x

    def _init_structure_transforms(self):
        """Initialize structure transforms for input and output."""
        # Input transform
        ctx = AlgoBuildContext(matrix_size=128, dim_size=self.q_lora_rank if self.q_lora_rank else self.kv_lora_rank)
        self.input_transform = build_algorithms_by_target(self.quant_args, "structure", ctx)

        # Output transform
        ctx = AlgoBuildContext(matrix_size=128, dim_size=self.value_dim)
        self.out_transform = build_algorithms_by_target(self.quant_args, "structure", ctx)


class QuantDeepseekV2MLP(QuantGatedMLP):
    """Quantized DeepSeek V2 MLP (inherits from QuantGatedMLP)."""

    def __init__(self, quant_args, mlp_module, group: str = "mlp"):
        mlp_module.hidden_size = mlp_module.gate_proj.in_features
        mlp_module.intermediate_size = mlp_module.gate_proj.out_features
        super().__init__(quant_args, mlp_module, group=group)