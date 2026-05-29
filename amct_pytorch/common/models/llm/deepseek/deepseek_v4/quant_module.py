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

"""v4 quant wrappers — prefill-only, blockwise PTQ path.

Wrappers compose around the upstream-style modules in `modeling_deepseek_v4.py`:
they reuse the wrapped module's parameters / sub-modules but override `forward`
to drop kv_cache / decode branches. They do NOT register kv_cache / kv_state /
score_state buffers themselves — those live on the wrapped Attention/Compressor/
Indexer instances and are simply unused under prefill.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from amct_pytorch.common.models.llm.common.quant_apply import QuantGatedMLP, PlainLinear
from amct_pytorch.quantization.modules.quant_base import ActivationQuantizer
from amct_pytorch.quantization.modules.quant_linear import QuantLinear
from amct_pytorch.quantization.bit_policy import ensure_bit_policy
from amct_pytorch.common.models.llm.deepseek.deepseek_v4.modeling.modeling_deepseek_v4 import (
    Attention,
    Compressor,
    Expert,
    Indexer,
    apply_rotary_emb,
    get_compress_topk_idxs,
    get_window_topk_idxs,
    rotate_activation,
)


class QuantV4Expert(QuantGatedMLP):
    """v4 Expert (w1=gate, w3=up, w2=down) reusing the framework's gated-MLP wrapper.

    Adapter shim: parent expects ``gate_proj/up_proj/down_proj/hidden_size/
    intermediate_size/act_fn``; v4 Expert provides ``w1/w3/w2`` and a numeric
    ``swiglu_limit``. Build a thin facade module that aliases the names, then
    let the parent wrap them in QuantLinear. ``forward`` is re-implemented to
    apply v4's clamp-then-silu semantics and the optional routing weight.
    """

    def __init__(self, quant_args, expert: Expert, group: str = "moe.routed"):
        facade = nn.Module()
        facade.gate_proj = expert.w1
        facade.up_proj = expert.w3
        facade.down_proj = expert.w2
        facade.hidden_size = expert.w1.in_features
        facade.intermediate_size = expert.w1.out_features
        facade.act_fn = nn.SiLU()
        super().__init__(quant_args, facade, group=group)
        self.swiglu_limit = float(getattr(expert, "swiglu_limit", 0.0))

    def forward(self, x, weights=None):
        dtype = x.dtype
        if self.input_transform is not None:
            x = self.input_transform(x)
        x_q = self.input_quant(x)
        gate = self.gate_proj(x_q, structure_transform=self.input_transform).float()
        up = self.up_proj(x_q, structure_transform=self.input_transform).float()
        if self.swiglu_limit > 0:
            up = torch.clamp(up, min=-self.swiglu_limit, max=self.swiglu_limit)
            gate = torch.clamp(gate, max=self.swiglu_limit)
        hidden = F.silu(gate) * up
        if self.hidden_transform is not None:
            hidden = self.hidden_transform(hidden)
        hidden_q = self.hidden_quant(hidden)
        out = self.down_proj(hidden_q, structure_transform=self.hidden_transform).to(dtype)
        if weights is not None:
            out = weights * out
        return out


class QuantV4Compressor(nn.Module):
    """Compressor wrapper. Quantizes ``wkv`` / ``wgate``. Prefill-only."""

    def __init__(self, quant_args, compressor: Compressor):
        super().__init__()
        self.quant_args = quant_args
        ensure_bit_policy(quant_args)
        self.enable_attn_linear = "attn-linear" in quant_args.quant_target
        self.compress_ratio = compressor.compress_ratio
        self.overlap = compressor.overlap
        self.rotate = compressor.rotate
        self.head_dim = compressor.head_dim
        self.rope_head_dim = compressor.rope_head_dim
        self.norm = compressor.norm
        self.ape = compressor.ape
        self.freqs_cis = compressor.freqs_cis
        if self.enable_attn_linear:
            bits = quant_args.bit_policy["attn-linear"]
            wkv, wgate = bits["comp_wkv"], bits["comp_wgate"]
            self.wkv = QuantLinear(quant_args, compressor.wkv, w_bits=wkv.w, name="comp_wkv")
            self.wgate = QuantLinear(quant_args, compressor.wgate, w_bits=wgate.w, name="comp_wgate")
            self.inp_afq = ActivationQuantizer(quant_args, wkv.a)
        else:
            self.wkv = PlainLinear(compressor.wkv)
            self.wgate = PlainLinear(compressor.wgate)
            self.inp_afq = nn.Identity()

    def overlap_transform(self, tensor: torch.Tensor, value=0):
        b, s, _, _ = tensor.size()
        ratio, d = self.compress_ratio, self.head_dim
        new_tensor = tensor.new_full((b, s, 2 * ratio, d), value)
        new_tensor[:, :, ratio:] = tensor[:, :, :, d:]
        new_tensor[:, 1:, :ratio] = tensor[:, :-1, :, :d]
        return new_tensor

    def forward(self, x: torch.Tensor, start_pos: int = 0):
        assert start_pos == 0, "QuantV4Compressor is prefill-only"
        bsz, seqlen, _ = x.size()
        ratio, overlap, rd = self.compress_ratio, self.overlap, self.rope_head_dim
        dtype = x.dtype
        x = x.float()
        x_q = self.inp_afq(x)
        kv = self.wkv(x_q)
        score = self.wgate(x_q)
        if seqlen < ratio:
            return None
        remainder = seqlen % ratio
        cutoff = seqlen - remainder
        if remainder > 0:
            kv = kv[:, :cutoff]
            score = score[:, :cutoff]
        kv = kv.unflatten(1, (-1, ratio))
        score = score.unflatten(1, (-1, ratio)) + self.ape
        if overlap:
            kv = self.overlap_transform(kv, 0)
            score = self.overlap_transform(score, float("-inf"))
        kv = (kv * score.softmax(dim=2)).sum(dim=2)
        kv = self.norm(kv.to(dtype))
        freqs_cis = self.freqs_cis[:cutoff:ratio]
        apply_rotary_emb(kv[..., -rd:], freqs_cis)
        if self.rotate:
            kv = rotate_activation(kv)
        return kv


class QuantV4Indexer(nn.Module):
    """Indexer wrapper. Quantizes ``wq_b`` / ``weights_proj`` and embeds a
    QuantV4Compressor for the indexer's own compressed-KV path.
    """

    def __init__(self, quant_args, indexer: Indexer):
        super().__init__()
        self.quant_args = quant_args
        ensure_bit_policy(quant_args)
        self.enable_attn_linear = "attn-linear" in quant_args.quant_target
        self.n_local_heads = indexer.n_local_heads
        self.head_dim = indexer.head_dim
        self.rope_head_dim = indexer.rope_head_dim
        self.compress_ratio = indexer.compress_ratio
        self.index_topk = indexer.index_topk
        self.softmax_scale = indexer.softmax_scale
        self.freqs_cis = indexer.freqs_cis
        if self.enable_attn_linear:
            bits = quant_args.bit_policy["attn-linear"]
            wq_b, weights_proj = bits["idx_wq_b"], bits["idx_weights_proj"]
            self.wq_b = QuantLinear(quant_args, indexer.wq_b, w_bits=wq_b.w, name="idx_wq_b")
            self.weights_proj = QuantLinear(
    quant_args,
    indexer.weights_proj,
    w_bits=weights_proj.w,
     name="idx_weights_proj")
            self.qr_afq = ActivationQuantizer(quant_args, wq_b.a)
            self.x_afq = ActivationQuantizer(quant_args, weights_proj.a)
        else:
            self.wq_b = PlainLinear(indexer.wq_b)
            self.weights_proj = PlainLinear(indexer.weights_proj)
            self.qr_afq = nn.Identity()
            self.x_afq = nn.Identity()
        self.compressor = QuantV4Compressor(quant_args, indexer.compressor)

    def forward(self, x, qr, start_pos: int = 0, offset: int = 0):
        assert start_pos == 0, "QuantV4Indexer is prefill-only"
        bsz, seqlen, _ = x.size()
        ratio, rd = self.compress_ratio, self.rope_head_dim
        freqs_cis = self.freqs_cis[:seqlen]
        qr_q = self.qr_afq(qr)
        q = self.wq_b(qr_q)
        q = q.unflatten(-1, (self.n_local_heads, self.head_dim))
        apply_rotary_emb(q[..., -rd:], freqs_cis)
        q = rotate_activation(q)
        compressed_kv = self.compressor(x, 0)  # [b, t, d] where t = seqlen // ratio
        x_q = self.x_afq(x)
        weights = self.weights_proj(x_q) * (self.softmax_scale * self.n_local_heads ** -0.5)
        end_pos = seqlen
        index_score = torch.einsum("bshd,btd->bsht", q, compressed_kv[:bsz, : end_pos // ratio])
        index_score = (index_score.relu_() * weights.unsqueeze(-1)).sum(dim=2)
        device = x.device
        causal_mask = torch.arange(seqlen // ratio, device=device).repeat(seqlen, 1) >= (
            torch.arange(1, seqlen + 1, device=device).unsqueeze(1) // ratio
        )
        index_score = index_score + torch.where(causal_mask, float("-inf"), 0.0)
        topk_idxs = index_score.topk(min(self.index_topk, end_pos // ratio), dim=-1)[1]
        oob_mask = topk_idxs >= (torch.arange(1, seqlen + 1, device=device).unsqueeze(1) // ratio)
        topk_idxs = torch.where(oob_mask, -1, topk_idxs + offset)
        return topk_idxs


class QuantV4Attention(nn.Module):
    """v4 MLA + sliding-window + optional KV compression. Prefill-only.

    Targets:
      - ``attn-linear``: quantizes ``wq_a / wq_b / wkv / wo_b``, plus the
        embedded Compressor/Indexer Linears. ``wo_a`` uses a grouped einsum
        (per-group block of a `[g*r, d_in]` weight) so we apply weight quant
        manually rather than via QuantLinear.forward.
      - ``attn-cache``: ActivationQuantizer on q (post-RMS+RoPE) and kv (post-
        norm+RoPE) — these are the inputs the original Attention writes into
        kv_cache during decode; under prefill they only flow into sparse_attn,
        but the quantizer placement still represents the cache-quant path.
    """

    def __init__(self, quant_args, module: Attention):
        super().__init__()
        self.quant_args = quant_args
        ensure_bit_policy(quant_args)
        self.enable_attn_linear = "attn-linear" in quant_args.quant_target
        self.enable_attn_cache = "attn-cache" in quant_args.quant_target

        self.layer_id = module.layer_id
        self.dim = module.dim
        self.n_heads = module.n_heads
        self.n_local_heads = module.n_local_heads
        self.q_lora_rank = module.q_lora_rank
        self.o_lora_rank = module.o_lora_rank
        self.head_dim = module.head_dim
        self.rope_head_dim = module.rope_head_dim
        self.n_local_groups = module.n_local_groups
        self.window_size = module.window_size
        self.compress_ratio = module.compress_ratio
        self.eps = module.eps
        self.softmax_scale = module.softmax_scale
        self.attn_sink = module.attn_sink
        self.q_norm = module.q_norm
        self.kv_norm = module.kv_norm
        self.freqs_cis = module.freqs_cis

        if self.enable_attn_linear:
            bits = quant_args.bit_policy["attn-linear"]
            wq_a, wq_b = bits["wq_a"], bits["wq_b"]
            wkv, wo_b, wo_a = bits["wkv"], bits["wo_b"], bits["wo_a"]
            self.wq_a = QuantLinear(quant_args, module.wq_a, w_bits=wq_a.w, name="wq_a")
            self.wq_b = QuantLinear(quant_args, module.wq_b, w_bits=wq_b.w, name="wq_b")
            self.wkv = QuantLinear(quant_args, module.wkv, w_bits=wkv.w, name="wkv")
            self.wo_b = QuantLinear(quant_args, module.wo_b, w_bits=wo_b.w, name="wo_b")
            from amct_pytorch.quantization.modules.quant_base import WeightQuantizer
            self.wo_a = module.wo_a
            self.quant_args.w_size = module.wo_a.weight.data.shape
            self.wo_a_name = "wo_a"
            self.wo_a_weight_quantizer = WeightQuantizer(self.quant_args, w_bits=wo_a.w)
            self.inp_afq = ActivationQuantizer(quant_args, wq_a.a)
            self.wq_b_afq = ActivationQuantizer(quant_args, wq_b.a)
            self.wo_b_afq = ActivationQuantizer(quant_args, wo_b.a)
        else:
            self.wq_a = PlainLinear(module.wq_a)
            self.wq_b = PlainLinear(module.wq_b)
            self.wkv = PlainLinear(module.wkv)
            self.wo_b = PlainLinear(module.wo_b)
            self.wo_a = module.wo_a
            self.wo_a_weight_quantizer = None
            self.inp_afq = nn.Identity()
            self.wq_b_afq = nn.Identity()
            self.wo_b_afq = nn.Identity()

        self.q_cache_quantizer = (
            ActivationQuantizer(quant_args, bits=quant_args.bit_policy.cache_bits("q"))
            if self.enable_attn_cache else nn.Identity()
        )
        self.k_cache_quantizer = (
            ActivationQuantizer(quant_args, bits=quant_args.bit_policy.cache_bits("k"))
            if self.enable_attn_cache else nn.Identity()
        )

        if self.compress_ratio:
            self.compressor = QuantV4Compressor(quant_args, module.compressor)
            self.indexer = (
                QuantV4Indexer(quant_args, module.indexer)
                if module.indexer is not None else None
            )
        else:
            self.compressor = None
            self.indexer = None

    def sparse_attn(self, query_states, kv_states, attn_sink, topk_idxs, softmax_scale):
        """Bit-for-bit copy of upstream Attention.sparse_attn — pure compute,
        no state, safe to inline.
        """
        query_states = query_states.transpose(1, 2)
        kv_states = kv_states.unsqueeze(1)
        attn_weights = torch.matmul(query_states, kv_states.transpose(2, 3)) * softmax_scale
        index_mask = torch.full(
            (query_states.shape[0], 1, query_states.shape[2], kv_states.shape[2] + 1),
            fill_value=torch.finfo(torch.float32).min,
            dtype=torch.float32,
            device=query_states.device,
        )
        valid_mask = topk_idxs >= 0
        q_idx, k_idx, m_idx = torch.where(valid_mask)
        index_mask[:, q_idx, k_idx, topk_idxs[q_idx, k_idx, m_idx]] = 0
        attn_weights = attn_weights + index_mask[..., :-1]
        sinks = attn_sink.reshape(1, -1, 1, 1).expand(query_states.shape[0], -1, query_states.shape[2], -1)
        combined_logits = torch.cat([attn_weights, sinks], dim=-1)
        combined_logits = combined_logits - combined_logits.max(dim=-1, keepdim=True).values
        probs = F.softmax(combined_logits, dim=-1, dtype=combined_logits.dtype)
        scores = probs[..., :-1]
        attn_output = torch.matmul(scores.to(query_states.dtype), kv_states)
        return attn_output.transpose(1, 2).contiguous()

    def forward(self, x: torch.Tensor, start_pos: int = 0):
        assert start_pos == 0, "QuantV4Attention is prefill-only"
        bsz, seqlen, _ = x.size()
        freqs_cis = self.freqs_cis[:seqlen]
        win = self.window_size
        ratio = self.compress_ratio
        rd = self.rope_head_dim

        if self.compress_ratio:
            if self.compressor.freqs_cis is None:
                self.compressor.freqs_cis = self.freqs_cis
            if self.indexer is not None:
                if self.indexer.freqs_cis is None:
                    self.indexer.freqs_cis = self.freqs_cis
                if self.indexer.compressor.freqs_cis is None:
                    self.indexer.compressor.freqs_cis = self.freqs_cis

        x_q = self.inp_afq(x)
        qr = q = self.q_norm(self.wq_a(x_q))
        qr_q = self.wq_b_afq(qr)
        q = self.wq_b(qr_q).unflatten(-1, (self.n_local_heads, self.head_dim))
        q = q * torch.rsqrt(q.square().mean(-1, keepdim=True) + self.eps)
        apply_rotary_emb(q[..., -rd:], freqs_cis)

        kv = self.wkv(x_q)
        kv = self.kv_norm(kv)
        apply_rotary_emb(kv[..., -rd:], freqs_cis)

        q = self.q_cache_quantizer(q)
        kv = self.k_cache_quantizer(kv)

        topk_idxs = get_window_topk_idxs(win, bsz, seqlen, 0)
        if self.compress_ratio:
            offset = kv.size(1)
            if self.indexer is not None:
                compress_topk_idxs = self.indexer(x, qr, 0, offset)
            else:
                compress_topk_idxs = get_compress_topk_idxs(ratio, bsz, seqlen, 0, offset)
            topk_idxs = torch.cat([topk_idxs, compress_topk_idxs], dim=-1)
        topk_idxs = topk_idxs.int()

        if self.compress_ratio:
            kv_compress = self.compressor(x, 0)
            if kv_compress is not None:
                kv = torch.cat([kv, kv_compress], dim=1)

        o = self.sparse_attn(q, kv, self.attn_sink.float(), topk_idxs.to(q.device), self.softmax_scale)
        apply_rotary_emb(o[..., -rd:], freqs_cis, True)

        o = o.view(bsz, seqlen, self.n_local_groups, -1)
        o = self._wo_a_apply(o)
        o_flat = o.flatten(2)
        o_q = self.wo_b_afq(o_flat)
        return self.wo_b(o_q)

    def _wo_a_apply(self, o):
        """Replicate the upstream einsum over wo_a's weight, with optional
        weight quant. ``o`` is shaped ``[b, s, g, d_in]`` where d_in = head_dim
        per group."""
        weight = self.wo_a.weight  # [g*r, d_in]
        if self.wo_a_weight_quantizer is not None:
            self.wo_a_weight_quantizer.observe_input(o.flatten(0, 2), weight)
            weight = self.wo_a_weight_quantizer(weight)
        weight = weight.view(self.n_local_groups, self.o_lora_rank, -1)  # [g, r, d_in]
        return torch.einsum("bsgd,grd->bsgr", o, weight)
