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

from amct_pytorch.algorithms.quant import AlgoBuildContext
from amct_pytorch.common.models.llm.common.quant_apply import PlainLinear
from amct_pytorch.quantization.bit_policy import ensure_bit_policy
from amct_pytorch.quantization.modules.quant_base import (
    ActivationQuantizer,
    build_algorithms_by_target,
)
from amct_pytorch.quantization.modules.quant_linear import QuantLinear
from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import (
    apply_rotary_pos_emb,
    eager_attention_forward,
)
from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import (
    GlmMoeDsaRMSNorm,
    GlmMoeDsaIndexer,
)


class QuantGlmMoeDsaAttention(nn.Module):
    """Quantized attention wrapper for GLM-5.1 MoE DSA (DeepSeek Sparse Attention).

    Wraps GlmMoeDsaAttention with QuantLinear for weight quantization and
    adds activation quantizers. The forward path is simplified for blockwise
    evaluation -- no caching, no prev_topk_indices, no skip_topk logic.
    """

    def __init__(self, args, attn_module):
        super().__init__()
        self.quant_args = args
        ensure_bit_policy(args)
        self.enable_attn_linear = "attn-linear" in args.quant_target
        self.enable_attn_cache = "attn-cache" in args.quant_target
        use_quant = self.enable_attn_linear or self.enable_attn_cache

        # Dimensional attributes (copied from source module)
        self.hidden_size = attn_module.config.hidden_size
        self.num_heads = attn_module.num_heads
        self.q_lora_rank = attn_module.q_lora_rank
        self.kv_lora_rank = attn_module.kv_lora_rank
        self.qk_head_dim = attn_module.qk_head_dim
        self.qk_nope_head_dim = attn_module.qk_nope_head_dim
        self.qk_rope_head_dim = attn_module.qk_rope_head_dim
        self.v_head_dim = attn_module.v_head_dim
        self.scaling = attn_module.scaling
        self.is_causal = True

        # Norm layers (not quantized) -- kept as-is
        self.q_a_layernorm = attn_module.q_a_layernorm
        self.kv_a_layernorm = attn_module.kv_a_layernorm

        # ---------- MLA linear projections ----------
        bits = args.bit_policy.get("attn-linear", {})
        self.q_a_proj = self._make_linear(args, use_quant, attn_module.q_a_proj, bits, "q_a_proj")
        self.q_b_proj = self._make_linear(args, use_quant, attn_module.q_b_proj, bits, "q_b_proj")
        self.kv_a_proj_with_mqa = self._make_linear(args, use_quant, attn_module.kv_a_proj_with_mqa, bits, "kv_a_proj_with_mqa")
        self.kv_b_proj = self._make_linear(args, use_quant, attn_module.kv_b_proj, bits, "kv_b_proj")
        self.o_proj = self._make_linear(args, use_quant, attn_module.o_proj, bits, "o_proj")

        # ---------- DSA Indexer ----------
        self.indexer = attn_module.indexer
        self.indexer.wq_b = self._make_linear(args, use_quant, attn_module.indexer.wq_b, bits, "wq_b")
        self.indexer.wk = self._make_linear(args, use_quant, attn_module.indexer.wk, bits, "wk")
        self.indexer.weights_proj = self._make_linear(args, use_quant, attn_module.indexer.weights_proj, bits, "weights_proj")

        # ---------- Structure transforms ----------
        if self.enable_attn_linear:
            self._init_structure_transforms()
            o_bits = bits.get("o_proj", None)
            o_a = o_bits.a if o_bits is not None and hasattr(o_bits, "a") else None
            self.o_proj_afq = ActivationQuantizer(args, o_a)
        else:
            self.o_proj_afq = nn.Identity()

    @staticmethod
    def _make_linear(quant_args, use_quant, linear, bits, name):
        bit_entry = bits.get(name, None)
        w_bits = bit_entry.w if bit_entry is not None and hasattr(bit_entry, "w") else None
        if use_quant:
            return QuantLinear(quant_args, linear, w_bits=w_bits, name=name)
        return PlainLinear(linear)

    def _init_structure_transforms(self):
        ctx = AlgoBuildContext(matrix_size=128, dim_size=self.q_lora_rank)
        self.q_input_transform = build_algorithms_by_target(self.quant_args, "structure", ctx)
        ctx = AlgoBuildContext(matrix_size=128, dim_size=self.num_heads * self.v_head_dim)
        self.o_output_transform = build_algorithms_by_target(self.quant_args, "structure", ctx)

    def _dsa_indexer(self, hidden_states, q_resid, position_embeddings, attention_mask):
        """Simplified DSA indexer -- replica of GlmMoeDsaIndexer.forward without cache."""
        batch_size, seq_len = hidden_states.shape[:2]
        cos, sin = position_embeddings

        # Query path: wq_b(q_resid) -> split nope/rope -> RoPE -> recombine
        q = self.indexer.wq_b(q_resid, structure_transform=None)
        q = q.view(batch_size, seq_len, self.indexer.n_heads, self.indexer.head_dim)
        q_pe, q_nope = torch.split(q, [self.qk_rope_head_dim, self.indexer.head_dim - self.qk_rope_head_dim], dim=-1)
        q_pe = apply_rotary_pos_emb(q_pe, cos, sin, unsqueeze_dim=2)
        q = torch.cat([q_pe, q_nope], dim=-1)

        # Key path: wk(hidden_states) -> k_norm -> split nope/rope -> RoPE -> recombine
        k = self.indexer.k_norm(self.indexer.wk(hidden_states, structure_transform=None))
        k_pe, k_nope = torch.split(k, [self.qk_rope_head_dim, self.indexer.head_dim - self.qk_rope_head_dim], dim=-1)
        k_pe = apply_rotary_pos_emb(k_pe.unsqueeze(2), cos, sin, unsqueeze_dim=2).squeeze(2)
        k = torch.cat([k_pe, k_nope], dim=-1)

        # Scores: einsum -> relu -> weighted by weights_proj
        weights = self.indexer.weights_proj(
            hidden_states, structure_transform=None
        ).float() * (self.indexer.n_heads ** -0.5)
        scores = torch.einsum("bshd,btd->bsht", q.float(), k.float()) * self.indexer.softmax_scale
        scores = F.relu(scores)
        index_scores = torch.einsum("bsht,bsh->bst", scores, weights)

        # Apply causal / attention mask
        if attention_mask is not None:
            if attention_mask.dim() == 4:
                index_scores = index_scores + attention_mask[:, 0, :, :]
            else:
                index_scores = index_scores + attention_mask

        total_len = index_scores.shape[-1]
        topk = min(self.indexer.index_topk, total_len)
        return index_scores.topk(topk, dim=-1).indices

    def forward(self, hidden_states, position_embeddings, attention_mask=None):
        batch_size, seq_length = hidden_states.shape[:-1]
        cos, sin = position_embeddings

        # ===== Query path (MLA with LoRA) =====
        q_resid = self.q_a_layernorm(self.q_a_proj(hidden_states, structure_transform=None))
        if self.enable_attn_linear and self.q_input_transform is not None:
            q_resid = self.q_input_transform(q_resid)
        query_states = self.q_b_proj(q_resid, structure_transform=None)
        query_states = query_states.view(batch_size, seq_length, -1, self.qk_head_dim).transpose(1, 2)
        q_nope, q_pe = torch.split(query_states, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        q_pe = apply_rotary_pos_emb(q_pe, cos, sin, unsqueeze_dim=1)

        # ===== KV path (MLA compressed) =====
        compressed_kv = self.kv_a_proj_with_mqa(hidden_states, structure_transform=None)
        k_compressed, k_pe = torch.split(compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        k_compressed = self.kv_a_layernorm(k_compressed)
        kv_expanded = self.kv_b_proj(k_compressed, structure_transform=None)
        kv_expanded = kv_expanded.view(batch_size, seq_length, -1, self.qk_nope_head_dim + self.v_head_dim)
        k_nope, value_states = torch.split(kv_expanded, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        k_nope = k_nope.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        # RoPE on k_pe (single-head rope stream, broadcast to all heads)
        k_pe = k_pe.view(batch_size, 1, seq_length, self.qk_rope_head_dim)
        k_pe = apply_rotary_pos_emb(k_pe, cos, sin, unsqueeze_dim=1)
        k_pe = k_pe.expand(-1, k_nope.shape[1], -1, -1)

        # Assemble full Q and K
        query_states = torch.cat([q_nope, q_pe], dim=-1)
        key_states = torch.cat([k_nope, k_pe], dim=-1)

        # ===== DSA Indexer =====
        topk_indices = self._dsa_indexer(hidden_states, q_resid, position_embeddings, attention_mask)

        # Build combined DSA + causal mask
        total_len = key_states.shape[2]
        index_mask = torch.full(
            (batch_size, seq_length, total_len),
            float("-inf"),
            device=hidden_states.device,
            dtype=query_states.dtype,
        )
        index_mask.scatter_(-1, topk_indices, 0.0)
        index_mask = index_mask.unsqueeze(1)
        if attention_mask is not None and attention_mask.dim() == 4:
            combined_mask = index_mask + attention_mask[..., :total_len]
        elif attention_mask is not None:
            combined_mask = index_mask + attention_mask
        else:
            combined_mask = index_mask

        # ===== Eager attention =====
        attn_output, _ = eager_attention_forward(
            self, query_states, key_states, value_states,
            combined_mask, self.scaling, dropout=0.0,
        )
        attn_output = attn_output.reshape(batch_size, seq_length, -1).contiguous()

        # ===== Output projection =====
        if self.enable_attn_linear and self.o_output_transform is not None:
            attn_output = self.o_output_transform(attn_output)
        attn_output = self.o_proj_afq(attn_output)
        attn_output = self.o_proj(attn_output, structure_transform=None)
        return attn_output
