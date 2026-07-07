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
"""Tests for QuantQwen3Attn / QuantQwen3MLP using tiny in-memory configs."""

from types import SimpleNamespace

import pytest
import torch
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3Attention,
    Qwen3Config,
    Qwen3MLP,
)

from amct_pytorch.common.models.llm.common.quant_apply import PlainLinear
from amct_pytorch.common.models.llm.qwen.qwen3.quant_module import (
    QuantQwen3Attn,
    QuantQwen3MLP,
)
from amct_pytorch.quantization.bit_policy import BitPolicy
from amct_pytorch.quantization.dtypes import register_dtype
from amct_pytorch.quantization.modules.quant_linear import QuantLinear

register_dtype()

W_BITS = 'w_bits'
A_BITS = 'a_bits'
ATTN_CACHE = 'attn-cache'
ATTN_LINEAR = 'attn-linear'
MLP = 'mlp'


def _tiny_qwen3_config():
    return Qwen3Config(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        vocab_size=32,
        max_position_embeddings=64,
        rms_norm_eps=1e-6,
    )


def _quant_args(quant_target=(ATTN_LINEAR,)):
    return SimpleNamespace(
        algos=[],
        quant_dtype="int",
        w_bits=8,
        a_bits=8,
        q_bits=8, k_bits=8, p_bits=8, v_bits=8,
        quant_target=list(quant_target),
        bit_policy=BitPolicy({
            ATTN_LINEAR: {
                "q_proj": {W_BITS: 8, A_BITS: 8},
                "k_proj": {W_BITS: 8, A_BITS: 8},
                "v_proj": {W_BITS: 8, A_BITS: 8},
                "o_proj": {W_BITS: 8, A_BITS: 8},
            },
            ATTN_CACHE: {"q": 8, "k": 8, "p": 8, "v": 8},
            MLP: {
                "gate_proj": {W_BITS: 8, A_BITS: 8},
                "up_proj": {W_BITS: 8, A_BITS: 8},
                "down_proj": {W_BITS: 8, A_BITS: 8},
            },
        }),
    )


def _position_embeddings(bs, seq, head_dim):
    cos = torch.randn(bs, seq, head_dim)
    sin = torch.randn(bs, seq, head_dim)
    return cos, sin


# ---- QuantQwen3MLP -------------------------------------------------------


def test_quant_qwen3_mlp_forward_preserves_shape():
    cfg = _tiny_qwen3_config()
    mlp = Qwen3MLP(cfg)
    qmlp = QuantQwen3MLP(_quant_args(quant_target=[MLP]), mlp)
    x = torch.randn(1, 4, cfg.hidden_size)
    out = qmlp(x)
    assert out.shape == x.shape


def test_quant_qwen3_mlp_uses_quant_linear_for_each_proj():
    cfg = _tiny_qwen3_config()
    mlp = Qwen3MLP(cfg)
    qmlp = QuantQwen3MLP(_quant_args(quant_target=[MLP]), mlp)
    assert isinstance(qmlp.up_proj, QuantLinear)
    assert isinstance(qmlp.gate_proj, QuantLinear)
    assert isinstance(qmlp.down_proj, QuantLinear)


# ---- QuantQwen3Attn ------------------------------------------------------


def test_quant_qwen3_attn_with_attn_linear_uses_quant_linear_projections():
    cfg = _tiny_qwen3_config()
    attn = Qwen3Attention(cfg, layer_idx=0)
    qattn = QuantQwen3Attn(_quant_args(quant_target=[ATTN_LINEAR]), attn)
    for proj in (qattn.q_proj, qattn.k_proj, qattn.v_proj, qattn.o_proj):
        assert isinstance(proj, QuantLinear)
    # Activation quantizers are real, not Identity.
    assert not isinstance(qattn.inp_afq, torch.nn.Identity)
    assert not isinstance(qattn.o_proj_afq, torch.nn.Identity)


def test_quant_qwen3_attn_without_attn_linear_uses_plain_linear_wrappers():
    cfg = _tiny_qwen3_config()
    attn = Qwen3Attention(cfg, layer_idx=0)
    qattn = QuantQwen3Attn(_quant_args(quant_target=[ATTN_CACHE]), attn)
    for proj in (qattn.q_proj, qattn.k_proj, qattn.v_proj, qattn.o_proj):
        assert isinstance(proj, PlainLinear)
    # In this branch transforms must be None and afq replaced with Identity.
    assert qattn.input_transform is None
    assert qattn.out_transform is None
    assert isinstance(qattn.inp_afq, torch.nn.Identity)
    assert isinstance(qattn.o_proj_afq, torch.nn.Identity)


def test_quant_qwen3_attn_qk_pv_matmul_constructed():
    cfg = _tiny_qwen3_config()
    attn = Qwen3Attention(cfg, layer_idx=0)
    qattn = QuantQwen3Attn(_quant_args(quant_target=[ATTN_CACHE]), attn)
    # qk_matmul / pv_matmul are always built regardless of attn-linear flag.
    assert qattn.qk_matmul is not None
    assert qattn.pv_matmul is not None
    # Bit widths should propagate from quant_args.
    assert qattn.qk_matmul.l_bits == 8
    assert qattn.qk_matmul.r_bits == 8


@pytest.mark.parametrize("quant_target", [[ATTN_LINEAR], [ATTN_CACHE]])
def test_quant_qwen3_attn_forward_returns_correct_shape(quant_target):
    cfg = _tiny_qwen3_config()
    attn = Qwen3Attention(cfg, layer_idx=0)
    qattn = QuantQwen3Attn(_quant_args(quant_target=quant_target), attn)
    bs, seq = 1, 6
    h = torch.randn(bs, seq, cfg.hidden_size)
    cos, sin = _position_embeddings(bs, seq, cfg.head_dim)
    out, weights = qattn(h, position_embeddings=(cos, sin))
    assert out.shape == h.shape
    # Implementation always returns a 2-tuple with weights=None.
    assert weights is None


def test_quant_qwen3_attn_forward_disabled_quant_matches_plain_attn_skeleton():
    """Without quantization (all quantizers off, no quant_target match),
    the wrapper should still give a finite, real-valued tensor.
    """
    cfg = _tiny_qwen3_config()
    attn = Qwen3Attention(cfg, layer_idx=0)
    qattn = QuantQwen3Attn(_quant_args(quant_target=[ATTN_CACHE]), attn)
    bs, seq = 1, 4
    h = torch.randn(bs, seq, cfg.hidden_size)
    cos, sin = _position_embeddings(bs, seq, cfg.head_dim)
    out, _ = qattn(h, position_embeddings=(cos, sin))
    assert torch.isfinite(out).all()


class _IdentityTransform(torch.nn.Module):
    def forward(self, x, **kwargs):
        return x


def test_quant_qwen3_attn_forward_with_structure_transform():
    """Cover input_transform / out_transform branches (lines 98, 126)."""
    cfg = _tiny_qwen3_config()
    attn = Qwen3Attention(cfg, layer_idx=0)
    qattn = QuantQwen3Attn(_quant_args(quant_target=[ATTN_LINEAR]), attn)
    qattn.input_transform = _IdentityTransform()
    qattn.out_transform = _IdentityTransform()
    bs, seq = 1, 4
    h = torch.randn(bs, seq, cfg.hidden_size)
    cos, sin = _position_embeddings(bs, seq, cfg.head_dim)
    out, _ = qattn(h, position_embeddings=(cos, sin))
    assert out.shape == h.shape
