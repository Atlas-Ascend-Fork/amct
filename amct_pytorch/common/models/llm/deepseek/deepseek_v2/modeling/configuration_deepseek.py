# coding=utf-8
# Copyright 2023 DeepSeek-AI and The HuggingFace Inc. team. All rights reserved.
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
#
# Adapted from the DeepSeek-V2 HuggingFace implementation for AMCT quantization.
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
from transformers.configuration_utils import PretrainedConfig


class DeepseekV2Config(PretrainedConfig):
    """Bilingual config for DeepSeek V2: accepts both HuggingFace canonical
    field names (`hidden_size`, `num_hidden_layers`, ...) and ModelArgs-style
    names (`dim`, `n_layers`, ...) that Block / Attention / MoE actually read.
    HF aliases win when both are provided.
    """
    model_type = "deepseek_v2"

    def __init__(
        self,
        # ---- ModelArgs-style canonical fields (defaults) ----
        max_batch_size=4,
        max_seq_len=4096,
        vocab_size=102400,
        dim=2048,
        inter_dim=10944,
        moe_inter_dim=1408,
        n_layers=27,
        n_heads=16,
        n_routed_experts=64,
        n_shared_experts=2,
        n_activated_experts=6,
        score_func="softmax",
        route_scale=1.0,
        q_lora_rank=None,  # V2 can be None (uses q_proj instead of q_a/q_b)
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        norm_eps=1e-6,
        first_k_dense_replace=1,
        moe_layer_freq=1,
        original_seq_len=4096,
        rope_theta=10000.0,
        rope_factor=40,
        beta_fast=32,
        beta_slow=1,
        mscale=0.707,
        tie_word_embeddings=False,
        # ---- HF-style aliases (override when provided) ----
        hidden_size=None,
        num_hidden_layers=None,
        num_attention_heads=None,
        intermediate_size=None,
        moe_intermediate_size=None,
        num_experts_per_tok=None,
        qk_rope_head_dim_hf=None,
        rms_norm_eps=None,
        scoring_func=None,
        routed_scaling_factor=None,
        max_position_embeddings=None,
        # ---- Nested HF dicts to unpack ----
        rope_scaling=None,
        **kwargs,
    ):
        # 1) HF flat aliases -> ModelArgs names
        if hidden_size is not None:
            dim = hidden_size
        if num_hidden_layers is not None:
            n_layers = num_hidden_layers
        if num_attention_heads is not None:
            n_heads = num_attention_heads
        if intermediate_size is not None:
            inter_dim = intermediate_size
        if moe_intermediate_size is not None:
            moe_inter_dim = moe_intermediate_size
        if num_experts_per_tok is not None:
            n_activated_experts = num_experts_per_tok
        if qk_rope_head_dim_hf is not None:
            qk_rope_head_dim = qk_rope_head_dim_hf
        if rms_norm_eps is not None:
            norm_eps = rms_norm_eps
        if scoring_func is not None:
            score_func = scoring_func
        if routed_scaling_factor is not None:
            route_scale = routed_scaling_factor
        if max_position_embeddings is not None:
            max_seq_len = max_position_embeddings

        # 2) rope_scaling dict -> flat YaRN fields
        if rope_scaling is not None:
            if "factor" in rope_scaling:
                rope_factor = rope_scaling["factor"]
            if "original_max_position_embeddings" in rope_scaling:
                original_seq_len = rope_scaling["original_max_position_embeddings"]
            if "beta_fast" in rope_scaling:
                beta_fast = rope_scaling["beta_fast"]
            if "beta_slow" in rope_scaling:
                beta_slow = rope_scaling["beta_slow"]
            if "mscale" in rope_scaling:
                mscale = rope_scaling["mscale"]

        # 3) Bind ModelArgs-style fields (what Block / Attention / MoE read)
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.dim = dim
        self.inter_dim = inter_dim
        self.moe_inter_dim = moe_inter_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.n_activated_experts = n_activated_experts
        self.score_func = score_func
        self.route_scale = route_scale
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.norm_eps = norm_eps
        self.first_k_dense_replace = first_k_dense_replace
        self.moe_layer_freq = moe_layer_freq
        self.original_seq_len = original_seq_len
        self.rope_theta = rope_theta
        self.rope_factor = rope_factor
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        self.mscale = mscale

        # 4) Keep HF-style aliases on self for HF utilities
        self.hidden_size = dim
        self.num_hidden_layers = n_layers
        self.num_attention_heads = n_heads
        self.intermediate_size = inter_dim
        self.moe_intermediate_size = moe_inter_dim
        self.num_experts_per_tok = n_activated_experts
        self.qk_rope_head_dim = qk_rope_head_dim
        self.rms_norm_eps = norm_eps
        self.scoring_func = score_func
        self.routed_scaling_factor = route_scale
        self.max_position_embeddings = max_seq_len
        self.rope_scaling = rope_scaling

        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)