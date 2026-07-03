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
"""Tests for QuantHYV3Attn / QuantHYV3MoE using tiny in-memory configs.

These tests construct real ``HYV3Attention`` / ``HYV3MoE`` modules and run
forward, so they require a real ``transformers.models.hy_v3`` (>= 5.12.1).
When conftest has fallen back to a mock (older transformers), the whole
module is skipped via ``pytest_collection_modifyitems`` in conftest.py.
"""
import torch
import torch.nn as nn
from transformers.models.hy_v3.modeling_hy_v3 import HYV3Attention, HYV3MoE

from amct_pytorch.common.models.llm.common.quant_apply import PlainLinear
from amct_pytorch.common.models.llm.hyv3.quant_module import QuantHYV3Attn, QuantHYV3MoE
from amct_pytorch.quantization.modules.quant_linear import QuantLinear

from .common import (
    ATTN_CACHE,
    ATTN_LINEAR,
    position_embeddings,
    quant_args,
    tiny_config,
)


# ---------------------------------------------------------------------------
# QuantHYV3Attn — __init__ / forward
# ---------------------------------------------------------------------------


class TestQuantHYV3Attn:
    @staticmethod
    def test_init_attn_linear_uses_quant_linear_projections():
        cfg = tiny_config()
        attn = HYV3Attention(cfg, layer_idx=0)
        qattn = QuantHYV3Attn(quant_args([ATTN_LINEAR]), attn)
        for proj in (qattn.q_proj, qattn.k_proj, qattn.v_proj, qattn.o_proj):
            assert isinstance(proj, QuantLinear)
        assert not isinstance(qattn.inp_afq, nn.Identity)
        assert not isinstance(qattn.o_proj_afq, nn.Identity)
        assert qattn.head_dim == cfg.head_dim
        assert qattn.value_dim == cfg.head_dim * cfg.num_attention_heads

    @staticmethod
    def test_init_cache_only_uses_plain_linear_and_identity_afq():
        cfg = tiny_config()
        attn = HYV3Attention(cfg, layer_idx=0)
        qattn = QuantHYV3Attn(quant_args([ATTN_CACHE]), attn)
        for proj in (qattn.q_proj, qattn.k_proj, qattn.v_proj, qattn.o_proj):
            assert isinstance(proj, PlainLinear)
        assert qattn.input_transform is None
        assert qattn.out_transform is None
        assert isinstance(qattn.inp_afq, nn.Identity)
        assert isinstance(qattn.o_proj_afq, nn.Identity)

    @staticmethod
    def test_init_builds_qk_and_pv_matmul():
        cfg = tiny_config()
        attn = HYV3Attention(cfg, layer_idx=0)
        qattn = QuantHYV3Attn(quant_args([ATTN_CACHE]), attn)
        assert qattn.qk_matmul is not None
        assert qattn.pv_matmul is not None

    @staticmethod
    def test_forward_attn_linear_returns_correct_shape():
        cfg = tiny_config()
        attn = HYV3Attention(cfg, layer_idx=0)
        qattn = QuantHYV3Attn(quant_args([ATTN_LINEAR]), attn)
        bs, seq = 1, 6
        h = torch.randn(bs, seq, cfg.hidden_size)
        cos, sin = position_embeddings(bs, seq, cfg.head_dim)
        out, weights = qattn(h, position_embeddings=(cos, sin))
        assert out.shape == h.shape
        assert weights is None

    @staticmethod
    def test_forward_cache_only_returns_finite_output():
        cfg = tiny_config()
        attn = HYV3Attention(cfg, layer_idx=0)
        qattn = QuantHYV3Attn(quant_args([ATTN_CACHE]), attn)
        bs, seq = 1, 4
        h = torch.randn(bs, seq, cfg.hidden_size)
        cos, sin = position_embeddings(bs, seq, cfg.head_dim)
        out, _ = qattn(h, position_embeddings=(cos, sin))
        assert out.shape == h.shape
        assert torch.isfinite(out).all()

    @staticmethod
    def test_forward_with_structure_transform_branches():
        cfg = tiny_config()
        attn = HYV3Attention(cfg, layer_idx=0)
        qattn = QuantHYV3Attn(quant_args([ATTN_LINEAR]), attn)

        class _IdentityTransform(nn.Module):
            def forward(self, x, **kwargs):
                return x

        qattn.input_transform = _IdentityTransform()
        qattn.out_transform = _IdentityTransform()
        bs, seq = 1, 4
        h = torch.randn(bs, seq, cfg.hidden_size)
        cos, sin = position_embeddings(bs, seq, cfg.head_dim)
        out, _ = qattn(h, position_embeddings=(cos, sin))
        assert out.shape == h.shape


# ---------------------------------------------------------------------------
# QuantHYV3MoE — __init__ / forward
# ---------------------------------------------------------------------------


class TestQuantHYV3MoE:
    @staticmethod
    def test_init_wraps_experts_and_shared():
        cfg = tiny_config()
        moe = HYV3MoE(cfg)
        qmoe = QuantHYV3MoE(quant_args(["moe"]), moe)
        assert qmoe.top_k == moe.top_k
        assert qmoe.gate is moe.gate
        assert qmoe.enable_moe_fp32_combine == moe.enable_moe_fp32_combine
        assert torch.equal(qmoe.e_score_correction_bias, moe.e_score_correction_bias)
        from amct_pytorch.common.models.llm.qwen.moe_common import QuantGatedExperts
        from amct_pytorch.common.models.llm.hyv3.quant_module import QuantHYV3MLP
        assert isinstance(qmoe.experts, QuantGatedExperts)
        assert isinstance(qmoe.shared_experts, QuantHYV3MLP)

    @staticmethod
    def test_forward_default_combine_path():
        cfg = tiny_config()
        moe = HYV3MoE(cfg)
        moe.enable_moe_fp32_combine = False
        qmoe = QuantHYV3MoE(quant_args(["moe"]), moe)
        h = torch.randn(2, 4, cfg.hidden_size)
        out = qmoe(h)
        assert out.shape == h.shape

    @staticmethod
    def test_forward_fp32_combine_path():
        cfg = tiny_config()
        moe = HYV3MoE(cfg)
        moe.enable_moe_fp32_combine = True
        qmoe = QuantHYV3MoE(quant_args(["moe"]), moe)
        h = torch.randn(1, 3, cfg.hidden_size)
        out = qmoe(h)
        assert out.shape == h.shape
        assert out.dtype == h.dtype
