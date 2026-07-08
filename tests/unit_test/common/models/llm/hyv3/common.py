# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
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
# ----------------------------------------------------------------------------
"""Shared fixtures and helpers for hyv3 unit tests."""
from types import SimpleNamespace

import torch
from transformers.models.hy_v3.modeling_hy_v3 import HYV3Config

from amct_pytorch.quantization.bit_policy import BitPolicy
from amct_pytorch.quantization.dtypes import register_dtype

register_dtype()

W_BITS = "w_bits"
A_BITS = "a_bits"
ATTN_LINEAR = "attn-linear"
ATTN_CACHE = "attn-cache"
MLP = "mlp"


def quant_args(quant_target=(ATTN_LINEAR,)):
    """Build a minimal quant_args with a BitPolicy covering all hyv3 groups."""
    return SimpleNamespace(
        algos=[],
        quant_dtype="int",
        w_bits=8,
        a_bits=8,
        q_bits=8,
        k_bits=8,
        p_bits=8,
        v_bits=8,
        quant_target=list(quant_target),
        bit_policy=BitPolicy({
            ATTN_LINEAR: {n: {W_BITS: 8, A_BITS: 8} for n in ("q_proj", "k_proj", "v_proj", "o_proj")},
            ATTN_CACHE: {"q": 8, "k": 8, "p": 8, "v": 8},
            MLP: {n: {W_BITS: 8, A_BITS: 8} for n in ("gate_proj", "up_proj", "down_proj")},
            "moe": {
                "routed": {n: {W_BITS: 8, A_BITS: 8} for n in ("gate_proj", "up_proj", "down_proj")},
                "shared": {n: {W_BITS: 8, A_BITS: 8} for n in ("gate_proj", "up_proj", "down_proj")},
            },
        }),
    )


def tiny_config():
    return HYV3Config(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        vocab_size=32,
        max_position_embeddings=64,
        rms_norm_eps=1e-6,
        num_experts=2,
        num_experts_per_tok=2,
        num_shared_experts=1,
        moe_intermediate_size=32,
    )


def position_embeddings(bs, seq, head_dim):
    return torch.randn(bs, seq, head_dim), torch.randn(bs, seq, head_dim)
