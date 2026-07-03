#!/usr/bin/env python3
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
"""Shared fixtures for hyv3 unit tests.

``transformers.models.hy_v3`` is only available since transformers 5.12.1.
When the installed transformers is older, inject a mock that is functional
enough to exercise ``QuantHYV3Attn`` / ``QuantHYV3MoE`` init + forward, so
``test_hyv3_quant_module`` can still run and keep coverage without a real
hy_v3 module. ``TestHyV3Mocked`` (which drives ``HyV3.__init__`` via
``AutoModelForCausalLM.from_config``) cannot work with a mock config and is
skipped via ``pytest_collection_modifyitems``.
"""
import importlib
import os
import sys

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F


def _build_hy_v3_mock():
    class HYV3Config:
        def __init__(self, **kwargs):
            defaults = dict(
                hidden_size=64,
                intermediate_size=128,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads=2,
                head_dim=16,
                vocab_size=100,
                max_position_embeddings=512,
                rms_norm_eps=1e-6,
                rope_theta=10000.0,
                tie_word_embeddings=False,
                num_experts=2,
                num_experts_per_tok=2,
                num_shared_experts=1,
                moe_intermediate_size=32,
                attention_dropout=0.0,
                scaling=1.0,
            )
            defaults.update(kwargs)
            for k, v in defaults.items():
                setattr(self, k, v)

    class HYV3Attention(nn.Module):
        def __init__(self, config, layer_idx=0):
            super().__init__()
            self.config = config
            self.layer_idx = layer_idx
            self.head_dim = config.head_dim
            self.attention_dropout = getattr(config, "attention_dropout", 0.0)
            self.scaling = getattr(config, "scaling", 1.0)
            nq = config.num_attention_heads
            nkv = getattr(config, "num_key_value_heads", nq)
            self.num_key_value_groups = nq // nkv
            h = config.hidden_size
            qkv = nkv * config.head_dim
            self.q_proj = nn.Linear(h, nq * config.head_dim)
            self.k_proj = nn.Linear(h, qkv)
            self.v_proj = nn.Linear(h, qkv)
            self.o_proj = nn.Linear(nq * config.head_dim, h)
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

    class HYV3MLP(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.hidden_size = config.hidden_size
            self.intermediate_size = config.moe_intermediate_size
            self.act_fn = F.silu
            self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size)
            self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size)
            self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size)

        def forward(self, x):
            return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

    class _Experts(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.num_experts = config.num_experts
            self.hidden_dim = config.hidden_size
            self.intermediate_dim = config.moe_intermediate_size
            self.act_fn = F.silu
            self.gate_up_proj = nn.Parameter(
                torch.randn(self.num_experts, 2 * self.intermediate_dim, self.hidden_dim)
            )
            self.down_proj = nn.Parameter(
                torch.randn(self.num_experts, self.hidden_dim, self.intermediate_dim)
            )

    class HYV3MoE(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.config = config
            self.top_k = config.num_experts_per_tok
            self.enable_moe_fp32_combine = True
            self.e_score_correction_bias = torch.zeros(config.num_experts)
            self.gate = lambda hd, bias: (
                None,
                torch.ones(hd.shape[0], self.top_k) / self.top_k,
                torch.zeros(hd.shape[0], self.top_k, dtype=torch.long),
            )
            self.experts = _Experts(config)
            self.shared_experts = HYV3MLP(config)

    class HYV3DecoderLayer:
        pass

    mod = type(sys)("transformers.models.hy_v3.modeling_hy_v3")
    mod.HYV3Config = HYV3Config
    mod.HYV3Attention = HYV3Attention
    mod.HYV3MoE = HYV3MoE
    mod.HYV3DecoderLayer = HYV3DecoderLayer
    mod.apply_rotary_pos_emb = lambda q, k, cos, sin, unsqueeze_dim=1: (q, k)
    return mod


def pytest_configure(config):
    try:
        importlib.import_module("transformers.models.hy_v3.modeling_hy_v3")
        return
    except ImportError:
        pass
    mod = _build_hy_v3_mock()
    sys.modules.setdefault("transformers.models.hy_v3", mod)
    sys.modules.setdefault("transformers.models.hy_v3.modeling_hy_v3", mod)
    os.environ["HY_V3_MOCKED"] = "1"


def pytest_collection_modifyitems(config, items):
    """Skip ``TestHyV3Mocked`` when hy_v3 is mocked (needs from_config)."""
    if os.environ.get("HY_V3_MOCKED") != "1":
        return
    skip_marker = pytest.mark.skip(
        reason="HyV3.__init__ mocked-init needs a real transformers hy_v3 (>=5.12.1)"
    )
    for item in items:
        if "TestHyV3Mocked" in item.nodeid:
            item.add_marker(skip_marker)
