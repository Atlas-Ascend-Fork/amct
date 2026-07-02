# Copyright 2026 the HuggingFace Team. All rights reserved.
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
from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import (
    apply_rotary_pos_emb,
    apply_rotary_pos_emb_interleave,
    eager_attention_forward,
)

from amct_pytorch.algorithms.quant import AlgoBuildContext
from amct_pytorch.common.models.llm.common.quant_apply import PlainLinear
from amct_pytorch.quantization.bit_policy import ensure_bit_policy
from amct_pytorch.quantization.modules.quant_base import (
    ActivationQuantizer,
    build_algorithms_by_target,
)
from amct_pytorch.quantization.modules.quant_linear import QuantLinear

_W = "w"
_A = "a"


class QuantGlmMoeDsaAttention(nn.Module):
    """Quantized attention wrapper for GLM-5.2 MoE DSA (DeepSeek Sparse Attention).

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

        self._copy_attn_dimensions(attn_module)
        bits = args.bit_policy["attn-linear"]
        self._build_mla_projections(args, attn_module, bits, use_quant)
        self._build_indexer_projections(args, attn_module, bits, use_quant)
        self._build_activation_quantizers(args, bits, use_quant)
        self._build_cache_quantizers(args)

    @staticmethod
    def _get_bits(bit_policy_entry, key):
        """Extract w or a from a LayerBits entry, or None if absent."""
        if bit_policy_entry is not None and hasattr(bit_policy_entry, key):
            return getattr(bit_policy_entry, key)
        return None

    def forward(
        self,
        hidden_states,
        position_embeddings,
        attention_mask=None,
        prev_topk_indices=None,
        position_ids=None,
        **_kwargs,
    ):
        batch_size, seq_length = hidden_states.shape[:-1]
        cos, sin = position_embeddings
        hidden_states = self.inp_afq(hidden_states)

        # ===== Query path (MLA with LoRA) =====
        q_resid = self.q_a_layernorm(
            self.q_a_proj(hidden_states, structure_transform=None)
        )
        if self.enable_attn_linear and self.q_input_transform is not None:
            q_resid = self.q_input_transform(q_resid)
        q_resid = self.q_b_proj_afq(q_resid)
        query_states = self.q_b_proj(q_resid, structure_transform=None)
        query_states = query_states.view(
            batch_size, seq_length, -1, self.qk_head_dim
        ).transpose(1, 2)
        q_nope, q_pe = torch.split(
            query_states, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )

        # ===== KV path (MLA compressed) =====
        compressed_kv = self.kv_a_proj_with_mqa(hidden_states, structure_transform=None)
        k_compressed, k_pe = torch.split(
            compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )
        k_compressed = self.kv_a_layernorm(k_compressed)
        kv_expanded = self.kv_b_proj(k_compressed, structure_transform=None)
        kv_expanded = kv_expanded.view(
            batch_size, seq_length, -1, self.qk_nope_head_dim + self.v_head_dim
        )
        k_nope, value_states = torch.split(
            kv_expanded, [self.qk_nope_head_dim, self.v_head_dim], dim=-1
        )
        k_nope = k_nope.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        # RoPE on q_pe and k_pe together
        k_pe = k_pe.view(batch_size, 1, seq_length, self.qk_rope_head_dim)
        q_pe, k_pe = apply_rotary_pos_emb_interleave(q_pe, k_pe, cos, sin)
        k_pe = k_pe.expand(-1, k_nope.shape[1], -1, -1)

        # Assemble full Q and K
        query_states = torch.cat([q_nope, q_pe], dim=-1)
        key_states = torch.cat([k_nope, k_pe], dim=-1)

        query_states = self.q_cache_quantizer(query_states)
        key_states = self.k_cache_quantizer(key_states)

        # ===== DSA Indexer (IndexShare: shared layers reuse, full layers compute) =====
        if self.indexer is not None:
            indexer_mask = attention_mask[:, 0, :, :] if attention_mask is not None else None
            topk_indices = self.indexer(
                hidden_states, q_resid, position_embeddings, indexer_mask, position_ids
            )
        else:
            topk_indices = prev_topk_indices

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
            self,
            query_states,
            key_states,
            value_states,
            combined_mask,
            self.scaling,
            dropout=0.0,
        )
        attn_output = attn_output.reshape(batch_size, seq_length, -1).contiguous()

        # ===== Output projection =====
        if self.enable_attn_linear and self.o_output_transform is not None:
            attn_output = self.o_output_transform(attn_output)
        attn_output = self.o_proj_afq(attn_output)
        attn_output = self.o_proj(attn_output, structure_transform=None)
        return attn_output, None, topk_indices

    def _build_cache_quantizers(self, args):
        """Build KV cache activation quantizers for attn-cache target."""
        self.q_cache_quantizer = (
            ActivationQuantizer(args, bits=args.bit_policy.cache_bits("q"))
            if self.enable_attn_cache else nn.Identity()
        )
        self.k_cache_quantizer = (
            ActivationQuantizer(args, bits=args.bit_policy.cache_bits("k"))
            if self.enable_attn_cache else nn.Identity()
        )

    def _build_activation_quantizers(self, args, bits, use_quant):
        """Build activation quantizers and structure transforms for attn-linear path."""
        if not self.enable_attn_linear:
            self.inp_afq = nn.Identity()
            self.q_b_proj_afq = nn.Identity()
            self.o_proj_afq = nn.Identity()
            return

        self._init_structure_transforms()
        q_b_a = self._get_bits(bits["q_b_proj"], _A)
        kv_a_a = self._get_bits(bits["kv_a_proj_with_mqa"], _A)
        o_a = self._get_bits(bits["o_proj"], _A)
        self.inp_afq = (
            ActivationQuantizer(args, kv_a_a)
            if args.quant_dtype == "mxfp" else nn.Identity()
        )
        self.q_b_proj_afq = ActivationQuantizer(args, q_b_a)
        self.o_proj_afq = ActivationQuantizer(args, o_a)

    def _copy_attn_dimensions(self, attn_module):
        """Copy dimensional attributes, norm layers, and flags from source module."""
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
        self.skip_topk = attn_module.skip_topk
        self.num_key_value_groups = attn_module.num_key_value_groups
        self.q_a_layernorm = attn_module.q_a_layernorm
        self.kv_a_layernorm = attn_module.kv_a_layernorm

    def _build_mla_projections(self, args, attn_module, bits, use_quant):
        """Build MLA linear projections.

        Aligns with infer repo convert_model.py:
          q_a_proj / kv_a_proj_with_mqa: PlainLinear in INT, QuantLinear in mxfp
          q_b_proj / o_proj: always quantized; kv_b_proj: never quantized
        """
        q_a = bits["q_a_proj"]
        q_b = bits["q_b_proj"]
        kv_a = bits["kv_a_proj_with_mqa"]
        o = bits["o_proj"]
        quant_q_a = use_quant and args.quant_dtype == "mxfp"

        self.q_a_proj = (
            QuantLinear(args, attn_module.q_a_proj,
                        w_bits=self._get_bits(q_a, _W), name="q_a_proj")
            if quant_q_a else PlainLinear(attn_module.q_a_proj)
        )
        self.q_b_proj = (
            QuantLinear(args, attn_module.q_b_proj,
                        w_bits=self._get_bits(q_b, _W), name="q_b_proj")
            if use_quant else PlainLinear(attn_module.q_b_proj)
        )
        self.kv_a_proj_with_mqa = (
            QuantLinear(args, attn_module.kv_a_proj_with_mqa,
                        w_bits=self._get_bits(kv_a, _W), name="kv_a_proj_with_mqa")
            if quant_q_a else PlainLinear(attn_module.kv_a_proj_with_mqa)
        )
        self.kv_b_proj = PlainLinear(attn_module.kv_b_proj)
        self.o_proj = (
            QuantLinear(args, attn_module.o_proj,
                        w_bits=self._get_bits(o, _W), name="o_proj")
            if use_quant else PlainLinear(attn_module.o_proj)
        )

    def _build_indexer_projections(self, args, attn_module, bits, use_quant):
        """Build DSA Indexer wrapper.

        Indexer is None for shared layers in transformers >= 5.12.
        Delegates to QuantGlmIndexer which wraps the HF indexer and inserts
        activation/cache quantizers into the forward path.
        """
        if attn_module.indexer is None:
            self.indexer = None
            return
        self.indexer = QuantGlmIndexer(args, attn_module.indexer, bits, use_quant)

    def _init_structure_transforms(self):
        ctx = AlgoBuildContext(matrix_size=128, dim_size=self.q_lora_rank)
        self.q_input_transform = build_algorithms_by_target(
            self.quant_args, "structure", ctx
        )
        ctx = AlgoBuildContext(
            matrix_size=128, dim_size=self.num_heads * self.v_head_dim
        )
        self.o_output_transform = build_algorithms_by_target(
            self.quant_args, "structure", ctx
        )


class QuantGlmIndexer(nn.Module):
    """Quantized wrapper for GLM-5.2 DSA Indexer.

    Wraps GlmMoeDsaIndexer with QuantLinear/PlainLinear for weight quantization
    and inserts activation/cache quantizers into the forward path. The forward
    logic mirrors HF GlmMoeDsaIndexer.forward exactly, adding quantization at
    four points: before wq_b, before weights_proj, and on Q/K after RoPE.
    """

    def __init__(self, args, indexer, bits, use_quant):
        super().__init__()
        self.quant_args = args
        self.enable_attn_linear = "attn-linear" in args.quant_target
        self.enable_attn_cache = "attn-cache" in args.quant_target

        # Copy dimension attributes from HF indexer for forward computation.
        self.n_heads = indexer.n_heads
        self.head_dim = indexer.head_dim
        self.qk_rope_head_dim = indexer.qk_rope_head_dim
        self.index_topk = indexer.index_topk
        self.q_lora_rank = indexer.q_lora_rank
        self.softmax_scale = indexer.softmax_scale

        # Build projections: wq_b always quantized, wk only in mxfp mode.
        wq_b = bits["wq_b"]
        wk = bits["wk"]
        quant_wk = use_quant and args.quant_dtype == "mxfp"

        self.wq_b = (
            QuantLinear(args, indexer.wq_b,
                        w_bits=QuantGlmMoeDsaAttention._get_bits(wq_b, _W), name="wq_b")
            if use_quant else PlainLinear(indexer.wq_b)
        )
        self.wk = (
            QuantLinear(args, indexer.wk,
                        w_bits=QuantGlmMoeDsaAttention._get_bits(wk, _W), name="wk")
            if quant_wk else PlainLinear(indexer.wk)
        )
        # k_norm and weights_proj are kept as raw modules (no weight quant).
        self.k_norm = indexer.k_norm
        self.weights_proj = indexer.weights_proj

        # Activation quantizers for attn-linear: qr_afq before wq_b, x_afq before weights_proj.
        if self.enable_attn_linear:
            wq_b_a = QuantGlmMoeDsaAttention._get_bits(wq_b, _A)
            wp_a = QuantGlmMoeDsaAttention._get_bits(bits["weights_proj"], _A)
            self.qr_afq = ActivationQuantizer(args, wq_b_a)
            self.x_afq = ActivationQuantizer(args, wp_a)
        else:
            self.qr_afq = nn.Identity()
            self.x_afq = nn.Identity()

        # Cache quantizers for attn-cache: quantize Q/K after RoPE before scoring.
        self.q_cache_quantizer = (
            ActivationQuantizer(args, bits=args.bit_policy.cache_bits("q"))
            if self.enable_attn_cache else nn.Identity()
        )
        self.k_cache_quantizer = (
            ActivationQuantizer(args, bits=args.bit_policy.cache_bits("k"))
            if self.enable_attn_cache else nn.Identity()
        )

    def forward(self, hidden_states, q_resid, position_embeddings,
                attention_mask, position_ids):
        """Compute DSA top-k indices with quantization inserted.

        Mirrors HF GlmMoeDsaIndexer.forward exactly, adding activation/cache
        quantization at four points.
        """
        batch_size, seq_len, _ = hidden_states.shape
        cos, sin = position_embeddings

        qr_q = self.qr_afq(q_resid)  # [INSERT] activation quant before wq_b
        q = self.wq_b(qr_q)
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim)
        q_rot, q_pass = torch.split(
            q, [self.qk_rope_head_dim, self.head_dim - self.qk_rope_head_dim], dim=-1
        )

        k = self.k_norm(self.wk(hidden_states)).unsqueeze(2)
        k_rot, k_pass = torch.split(
            k, [self.qk_rope_head_dim, self.head_dim - self.qk_rope_head_dim], dim=-1
        )

        q_rot, k_rot = apply_rotary_pos_emb(q_rot, k_rot, cos, sin, unsqueeze_dim=2)
        q = torch.cat([q_rot, q_pass], dim=-1)
        k = torch.cat([k_rot, k_pass], dim=-1).squeeze(2)

        q = self.q_cache_quantizer(q)  # [INSERT] cache quant on Q after RoPE
        k = self.k_cache_quantizer(k)  # [INSERT] cache quant on K after RoPE

        scores = torch.matmul(
            q.float(), k.transpose(-1, -2).float().unsqueeze(1)
        ) * self.softmax_scale
        scores = torch.relu(scores)

        x_q = self.x_afq(hidden_states)  # [INSERT] activation quant before weights_proj
        weights = self.weights_proj(
            x_q.to(self.weights_proj.weight.dtype)
        ).float() * (self.n_heads ** -0.5)
        index_scores = torch.matmul(weights.unsqueeze(-2), scores).squeeze(-2)

        if attention_mask is not None:
            index_scores = index_scores + attention_mask
        else:
            key_positions = torch.arange(
                index_scores.shape[-1], device=index_scores.device
            )
            causal = key_positions[None, None, :] > position_ids[:, :, None]
            index_scores = index_scores.masked_fill(causal, float("-inf"))

        topk = min(self.index_topk, index_scores.shape[-1])
        return index_scores.topk(topk, dim=-1).indices.to(torch.int32)
