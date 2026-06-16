# coding=utf-8
# Copyright 2023 DeepSeek-AI and The HuggingFace Inc. team. All rights reserved.
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Adapted from
# https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite/blob/main/modeling_deepseek.py
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
import math
from typing import Optional
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers.activations import SiLUActivation
from amct_pytorch.common.models.llm.deepseek.deepseek_v2.modeling.configuration_deepseek import DeepseekV2Config


@dataclass
class ModelArgs:
    """ModelArgs for DeepSeek V2 compatibility layer."""
    max_batch_size: int = 4
    max_seq_len: int = 4096
    vocab_size: int = 102400
    dim: int = 2048
    inter_dim: int = 10944
    moe_inter_dim: int = 1408
    n_layers: int = 27
    n_heads: int = 16
    n_routed_experts: int = 64
    n_shared_experts: int = 2
    n_activated_experts: int = 6
    score_func: str = "softmax"
    route_scale: float = 1.0
    q_lora_rank: Optional[int] = None  # V2 can be None
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    norm_eps: float = 1e-6
    first_k_dense_replace: int = 1
    moe_layer_freq: int = 1
    original_seq_len: int = 4096
    rope_theta: float = 10000.0
    rope_factor: float = 40.0
    beta_fast: int = 32
    beta_slow: int = 1
    mscale: float = 0.707


def find_correction_dim(num_rotations, dim, base=10000, max_seq_len=4096):
    return dim * math.log(max_seq_len / (num_rotations * 2 * math.pi)) / (2 * math.log(base))


def find_correction_range(low_rot, high_rot, dim, base=10000, max_seq_len=4096):
    low = math.floor(find_correction_dim(low_rot, dim, base, max_seq_len))
    high = math.ceil(find_correction_dim(high_rot, dim, base, max_seq_len))
    return max(low, 0), min(high, dim - 1)


def linear_ramp_factor(min_val, max_val, dim):
    if min_val == max_val:
        max_val += 0.001
    linear_func = (torch.arange(dim, dtype=torch.float32) - min_val) / (max_val - min_val)
    ramp_func = torch.clamp(linear_func, 0, 1)
    return ramp_func


def precompute_freqs_cis(args: ModelArgs):
    """Precompute rotary embedding frequencies (YaRN scaled)."""
    dim = args.qk_rope_head_dim
    seqlen = args.max_seq_len
    base = args.rope_theta
    factor = args.rope_factor
    beta_fast = args.beta_fast
    beta_slow = args.beta_slow

    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if seqlen > args.original_seq_len:
        low, high = find_correction_range(beta_fast, beta_slow, dim, base, args.original_seq_len)
        smooth = 1 - linear_ramp_factor(low, high, dim // 2)
        freqs = freqs / factor * (1 - smooth) + freqs * smooth

    t = torch.arange(seqlen)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor, interleaved: bool = True) -> torch.Tensor:
    """Apply rotary positional embeddings.

    Args:
        x: Input tensor of shape (bsz, seqlen, n_heads, head_dim) or (bsz, seqlen, 1, head_dim)
        freqs_cis: Precomputed complex frequencies of shape (seqlen, head_dim // 2)
        interleaved: Whether the rotary embedding is interleaved

    Returns:
        Tensor with rotary embeddings applied, shape preserved
    """
    dtype = x.dtype
    shape = x.shape
    # x shape: (bsz, seqlen, n_heads_or_1, head_dim)
    # freqs_cis shape: (seqlen, head_dim // 2)

    if not interleaved:
        x = x.view(*shape[:-1], 2, -1).transpose(-1, -2).contiguous()
    x = torch.view_as_complex(x.float().view(*shape[:-1], -1, 2))
    # freqs_cis needs to broadcast to (bsz, seqlen, n_heads, head_dim // 2)
    freqs_cis = freqs_cis.view(1, x.size(1), 1, x.size(-1))
    y = torch.view_as_real(x * freqs_cis).flatten(3)
    if not interleaved:
        y = torch.cat([y[..., 0::2], y[..., 1::2]], dim=-1)
    return y.to(dtype)


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))

    def forward(self, x: torch.Tensor):
        dtype = x.dtype
        x = x.float()
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return (self.weight * x).to(dtype)


class DeepseekV2Attention(nn.Module):
    """Multi-Head Latent Attention for DeepSeek V2 (No Indexer)."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.dim = args.dim
        self.n_heads = args.n_heads
        self.q_lora_rank = args.q_lora_rank
        self.kv_lora_rank = args.kv_lora_rank
        self.qk_nope_head_dim = args.qk_nope_head_dim
        self.qk_rope_head_dim = args.qk_rope_head_dim
        self.qk_head_dim = args.qk_nope_head_dim + args.qk_rope_head_dim
        self.v_head_dim = args.v_head_dim

        # Query projection: can be direct (q_proj) or LoRA (q_a_proj + q_b_proj)
        if self.q_lora_rank is None:
            self.q_proj = nn.Linear(self.dim, self.n_heads * self.qk_head_dim, bias=False)
        else:
            self.q_a_proj = nn.Linear(self.dim, self.q_lora_rank, bias=False)
            self.q_a_layernorm = RMSNorm(self.q_lora_rank)
            self.q_b_proj = nn.Linear(self.q_lora_rank, self.n_heads * self.qk_head_dim, bias=False)

        # KV projection (MLA)
        self.kv_a_proj_with_mqa = nn.Linear(self.dim, self.kv_lora_rank + self.qk_rope_head_dim, bias=False)
        self.kv_a_layernorm = RMSNorm(self.kv_lora_rank)
        self.kv_b_proj = nn.Linear(self.kv_lora_rank, self.n_heads * (self.qk_nope_head_dim + self.v_head_dim), bias=False)

        # Output projection
        self.o_proj = nn.Linear(self.n_heads * self.v_head_dim, self.dim, bias=False)

        # Softmax scale
        self.softmax_scale = self.qk_head_dim ** -0.5
        if args.max_seq_len > args.original_seq_len:
            mscale = 0.1 * args.mscale * math.log(args.rope_factor) + 1.0
            self.softmax_scale = self.softmax_scale * mscale * mscale

        # Precompute rotary embeddings
        self.register_buffer("freqs_cis", precompute_freqs_cis(args), persistent=False)

    def forward(self, x: torch.Tensor, start_pos: int = 0):
        bsz, seqlen, _ = x.size()
        end_pos = start_pos + seqlen

        # Causal mask - shape (seqlen, seqlen), will be broadcast properly
        mask = None
        if seqlen > 1:
            mask = torch.full((seqlen, seqlen), float("-inf"), device=x.device, dtype=x.dtype).triu_(1)

        # Rotary embeddings
        freqs_cis = self.freqs_cis[start_pos:end_pos]

        # Query
        if self.q_lora_rank is None:
            q = self.q_proj(x)
        else:
            qr = self.q_a_layernorm(self.q_a_proj(x))
            q = self.q_b_proj(qr)

        q = q.view(bsz, seqlen, self.n_heads, self.qk_head_dim)
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        q_pe = apply_rotary_emb(q_pe, freqs_cis)

        # KV (MLA)
        kv = self.kv_a_proj_with_mqa(x)
        kv, k_pe = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        kv = self.kv_a_layernorm(kv)
        # k_pe: (bsz, seqlen, qk_rope_head_dim) -> apply RoPE after unsqueeze and expand
        k_pe = k_pe.unsqueeze(2)  # (bsz, seqlen, 1, qk_rope_head_dim)
        k_pe = apply_rotary_emb(k_pe, freqs_cis)  # (bsz, seqlen, 1, qk_rope_head_dim)
        k_pe = k_pe.expand(-1, -1, self.n_heads, -1)  # (bsz, seqlen, n_heads, qk_rope_head_dim)

        kv = self.kv_b_proj(kv)
        kv = kv.view(bsz, seqlen, self.n_heads, self.qk_nope_head_dim + self.v_head_dim)
        k_nope, v = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        # Assemble query and key
        q = torch.cat([q_nope, q_pe], dim=-1)  # (bsz, seqlen, n_heads, qk_head_dim)
        k = torch.cat([k_nope, k_pe], dim=-1)  # (bsz, seqlen, n_heads, qk_head_dim)

        # Attention scores: einsum produces (bsz, seqlen, n_heads, seqlen)
        # We need scores shape (bsz, n_heads, seqlen, seqlen) to match standard attention
        scores = torch.einsum("bshd,bthd->bhst", q, k).mul_(self.softmax_scale)
        if mask is not None:
            scores = scores + mask  # mask (seqlen, seqlen) broadcasts to (bsz, n_heads, seqlen, seqlen)

        scores = scores.softmax(dim=-1)
        # After softmax, scores is (bsz, n_heads, seqlen, seqlen), need to einsum back
        x = torch.einsum("bhst,bthd->bshd", scores, v)
        x = x.flatten(2)
        x = self.o_proj(x)

        return x


class DeepseekV2MLP(nn.Module):
    """Dense MLP for DeepSeek V2."""

    def __init__(self, dim: int, inter_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(dim, inter_dim, bias=False)
        self.up_proj = nn.Linear(dim, inter_dim, bias=False)
        self.down_proj = nn.Linear(inter_dim, dim, bias=False)
        self.act_fn = SiLUActivation()

    def forward(self, x: torch.Tensor):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class MoEGate(nn.Module):
    """MoE gating network."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.top_k = args.n_activated_experts
        self.n_routed_experts = args.n_routed_experts
        self.route_scale = args.route_scale
        self.scoring_func = args.score_func
        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, args.dim)))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, hidden_states: torch.Tensor):
        bsz, seq_len, h = hidden_states.shape
        hidden_states = hidden_states.view(-1, h)
        logits = F.linear(hidden_states.float(), self.weight.float())
        scores = logits.softmax(dim=-1) if self.scoring_func == "softmax" else logits
        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)
        topk_weight = topk_weight * self.route_scale
        return topk_idx, topk_weight


class DeepseekV2MoE(nn.Module):
    """Mixed Expert module with shared experts."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.num_experts_per_tok = args.n_activated_experts
        self.experts = nn.ModuleList([
            DeepseekV2MLP(args.dim, args.moe_inter_dim)
            for _ in range(args.n_routed_experts)
        ])
        self.gate = MoEGate(args)
        if args.n_shared_experts is not None:
            self.shared_experts = DeepseekV2MLP(args.dim, args.moe_inter_dim * args.n_shared_experts)

    def forward(self, x: torch.Tensor):
        identity = x
        orig_shape = x.shape
        topk_idx, topk_weight = self.gate(x)
        x = x.view(-1, x.shape[-1])

        # Process routed experts
        y = self.moe_infer(x, topk_idx, topk_weight).view(*orig_shape)

        # Add shared experts
        if hasattr(self, "shared_experts"):
            y = y + self.shared_experts(identity)

        return y

    @torch.no_grad()
    def moe_infer(self, x, topk_ids, topk_weight):
        cnts = topk_ids.new_zeros((topk_ids.shape[0], len(self.experts)))
        cnts.scatter_(1, topk_ids, 1)
        tokens_per_expert = cnts.sum(dim=0)
        idxs = topk_ids.view(-1).argsort()
        sorted_tokens = x[idxs // topk_ids.shape[1]]

        outputs = []
        start_idx = 0
        for i, num_tokens in enumerate(tokens_per_expert.cpu().numpy()):
            end_idx = start_idx + num_tokens
            if num_tokens == 0:
                continue
            expert = self.experts[i]
            tokens_for_this_expert = sorted_tokens[start_idx:end_idx]
            expert_out = expert(tokens_for_this_expert)
            outputs.append(expert_out)
            start_idx = end_idx

        outs = torch.cat(outputs, dim=0) if len(outputs) else sorted_tokens.new_empty(0)
        new_x = torch.empty_like(outs)
        new_x[idxs] = outs
        final_out = (
            new_x.view(*topk_ids.shape, -1)
            .type(topk_weight.dtype)
            .mul_(topk_weight.unsqueeze(dim=-1))
            .sum(dim=1)
            .type(new_x.dtype)
        )
        return final_out


class Block(nn.Module):
    """DeepSeek V2 Decoder Block."""

    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.self_attn = DeepseekV2Attention(args)
        self.mlp = (
            DeepseekV2MLP(args.dim, args.inter_dim)
            if layer_id < args.first_k_dense_replace
            else DeepseekV2MoE(args)
        )
        self.input_layernorm = RMSNorm(args.dim)
        self.post_attention_layernorm = RMSNorm(args.dim)

    def forward(self, x: torch.Tensor, start_pos: int = 0):
        residual = x
        x = self.input_layernorm(x)
        x = self.self_attn(x, start_pos)
        x = residual.to(x.device) + x
        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        x = residual.to(x.device) + x
        return x


class DeepseekV2PreTrainedModel(PreTrainedModel):
    config_class = DeepseekV2Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = False
    _no_split_modules = ["Block"]
    _supports_cache_class = False

    def _init_weights(self, module):
        # Weights are loaded layer-by-layer from checkpoint; skip random init
        return


class DeepseekV2ForCausalLM(DeepseekV2PreTrainedModel):
    """DeepSeek V2 model for CausalLM."""

    def __init__(self, config: DeepseekV2Config):
        super().__init__(config)
        # Map config to ModelArgs
        args = ModelArgs(
            max_batch_size=config.max_batch_size,
            max_seq_len=config.max_seq_len,
            vocab_size=config.vocab_size,
            dim=config.dim,
            inter_dim=config.inter_dim,
            moe_inter_dim=config.moe_inter_dim,
            n_layers=config.n_layers,
            n_heads=config.n_heads,
            n_routed_experts=config.n_routed_experts,
            n_shared_experts=config.n_shared_experts,
            n_activated_experts=config.n_activated_experts,
            score_func=config.score_func,
            route_scale=config.route_scale,
            q_lora_rank=config.q_lora_rank,
            kv_lora_rank=config.kv_lora_rank,
            qk_nope_head_dim=config.qk_nope_head_dim,
            qk_rope_head_dim=config.qk_rope_head_dim,
            v_head_dim=config.v_head_dim,
            norm_eps=config.norm_eps,
            first_k_dense_replace=config.first_k_dense_replace,
            moe_layer_freq=config.moe_layer_freq,
            original_seq_len=config.original_seq_len,
            rope_theta=config.rope_theta,
            rope_factor=config.rope_factor,
            beta_fast=config.beta_fast,
            beta_slow=config.beta_slow,
            mscale=config.mscale,
        )

        self.embed = nn.Embedding(config.vocab_size, config.dim)
        self.layers = torch.nn.ModuleList()
        for layer_id in range(config.n_layers):
            self.layers.append(Block(layer_id, args))
        self.norm = RMSNorm(config.dim, config.norm_eps)
        self.head = nn.Linear(config.dim, config.vocab_size, bias=False)
        self.post_init()

    @torch.inference_mode()
    def forward(self, tokens: torch.Tensor, start_pos: int = 0):
        h = self.embed(tokens)
        for layer in self.layers:
            h = layer(h, start_pos=start_pos)
        h, _ = self.norm(h)
        logits = self.head(h[:, -1].float())
        return logits