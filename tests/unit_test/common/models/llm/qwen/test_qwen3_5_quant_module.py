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
"""Tests for QuantQwen35Attn / QuantQwen35MLP / QuantQwen35LinearAttn using tiny configs."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5Attention,
    Qwen3_5GatedDeltaNet,
    Qwen3_5MLP,
    Qwen3_5TextConfig,
)

from amct_pytorch.common.models.llm.common.quant_apply import PlainLinear
from amct_pytorch.common.models.llm.qwen.qwen3_5.quant_module import (
    QuantQwen35Attn,
    QuantQwen35LinearAttn,
    QuantQwen35MLP,
)
from amct_pytorch.quantization.bit_policy import BitPolicy
from amct_pytorch.quantization.dtypes import register_dtype
from amct_pytorch.quantization.modules.quant_base import ActivationQuantizer
from amct_pytorch.quantization.modules.quant_linear import QuantLinear

register_dtype()

W_BITS = 'w_bits'

A_BITS = 'a_bits'


def _tiny_config():
    return Qwen3_5TextConfig(
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


def _linear_attn_config():
    return Qwen3_5TextConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        vocab_size=32,
        max_position_embeddings=64,
        rms_norm_eps=1e-6,
        linear_num_value_heads=4,
        linear_num_key_heads=2,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_conv_kernel_dim=4,
    )


def _quant_args(quant_target=("attn-linear",)):
    return SimpleNamespace(
        algos=[],
        quant_dtype="int",
        w_bits=8,
        a_bits=8,
        q_bits=8, k_bits=8, p_bits=8, v_bits=8,
        quant_target=list(quant_target),
        bit_policy=BitPolicy({
            "attn-linear": {
                "q_proj": {W_BITS: 8, A_BITS: 8},
                "k_proj": {W_BITS: 8, A_BITS: 8},
                "v_proj": {W_BITS: 8, A_BITS: 8},
                "o_proj": {W_BITS: 8, A_BITS: 8},
                "out_proj": {W_BITS: 8, A_BITS: 8},
                "in_proj_qkv": {W_BITS: 8, A_BITS: 8},
                "in_proj_z": {W_BITS: 8, A_BITS: 8},
                "in_proj_b": {W_BITS: 8, A_BITS: 8},
                "in_proj_a": {W_BITS: 8, A_BITS: 8},
            },
            "attn-cache": {"q": 8, "k": 8, "p": 8, "v": 8},
            "mlp": {
                "gate_proj": {W_BITS: 8, A_BITS: 8},
                "up_proj": {W_BITS: 8, A_BITS: 8},
                "down_proj": {W_BITS: 8, A_BITS: 8},
            },
        }),
    )


# ---- QuantQwen35MLP ------------------------------------------------------


def test_quant_qwen35_mlp_forward_preserves_shape():
    cfg = _tiny_config()
    qmlp = QuantQwen35MLP(_quant_args(quant_target=["mlp"]), Qwen3_5MLP(cfg, cfg.intermediate_size))
    x = torch.randn(1, 4, cfg.hidden_size)
    assert qmlp(x).shape == x.shape


def test_quant_qwen35_mlp_uses_quant_linear_for_projections():
    cfg = _tiny_config()
    qmlp = QuantQwen35MLP(_quant_args(quant_target=["mlp"]), Qwen3_5MLP(cfg, cfg.intermediate_size))
    assert isinstance(qmlp.up_proj, QuantLinear)
    assert isinstance(qmlp.gate_proj, QuantLinear)
    assert isinstance(qmlp.down_proj, QuantLinear)


def test_quant_qwen35_mlp_has_activation_quantizers():
    cfg = _tiny_config()
    qmlp = QuantQwen35MLP(_quant_args(quant_target=["mlp"]), Qwen3_5MLP(cfg, cfg.intermediate_size))
    assert isinstance(qmlp.input_quant, ActivationQuantizer)
    assert isinstance(qmlp.hidden_quant, ActivationQuantizer)


def test_quant_qwen35_mlp_export_ptq_params():
    cfg = _tiny_config()
    qmlp = QuantQwen35MLP(_quant_args(quant_target=["mlp"]), Qwen3_5MLP(cfg, cfg.intermediate_size))
    params = qmlp.export_ptq_params()
    assert isinstance(params, dict)


# ---- QuantQwen35Attn -----------------------------------------------------


def test_quant_qwen35_attn_attn_linear_branch_uses_quant_linear():
    cfg = _tiny_config()
    attn = Qwen3_5Attention(cfg, layer_idx=0)
    qattn = QuantQwen35Attn(_quant_args(quant_target=["attn-linear"]), attn)
    for proj in (qattn.q_proj, qattn.k_proj, qattn.v_proj, qattn.o_proj):
        assert isinstance(proj, QuantLinear)


def test_quant_qwen35_attn_attn_linear_branch_has_activation_quantizers():
    cfg = _tiny_config()
    attn = Qwen3_5Attention(cfg, layer_idx=0)
    qattn = QuantQwen35Attn(_quant_args(quant_target=["attn-linear"]), attn)
    assert isinstance(qattn.inp_afq, ActivationQuantizer)
    assert isinstance(qattn.o_proj_afq, ActivationQuantizer)


def test_quant_qwen35_attn_passthrough_branch_uses_plain_linear():
    cfg = _tiny_config()
    attn = Qwen3_5Attention(cfg, layer_idx=0)
    qattn = QuantQwen35Attn(_quant_args(quant_target=["attn-cache"]), attn)
    for proj in (qattn.q_proj, qattn.k_proj, qattn.v_proj, qattn.o_proj):
        assert isinstance(proj, PlainLinear)
    assert qattn.input_transform is None
    assert qattn.out_transform is None


def test_quant_qwen35_attn_passthrough_branch_uses_identity_afq():
    cfg = _tiny_config()
    attn = Qwen3_5Attention(cfg, layer_idx=0)
    qattn = QuantQwen35Attn(_quant_args(quant_target=["attn-cache"]), attn)
    assert isinstance(qattn.inp_afq, nn.Identity)
    assert isinstance(qattn.o_proj_afq, nn.Identity)


@pytest.mark.parametrize("quant_target", [["attn-linear"], ["attn-cache"]])
def test_quant_qwen35_attn_forward_returns_correct_shape(quant_target):
    cfg = _tiny_config()
    attn = Qwen3_5Attention(cfg, layer_idx=0)
    qattn = QuantQwen35Attn(_quant_args(quant_target=quant_target), attn)
    bs, seq = 1, 4
    h = torch.randn(bs, seq, cfg.hidden_size)
    cos = torch.randn(bs, seq, cfg.head_dim)
    sin = torch.randn(bs, seq, cfg.head_dim)
    out, weights = qattn(h, position_embeddings=(cos, sin))
    assert out.shape == h.shape
    assert weights is None


def test_quant_qwen35_attn_batch_forward():
    cfg = _tiny_config()
    attn = Qwen3_5Attention(cfg, layer_idx=0)
    qattn = QuantQwen35Attn(_quant_args(quant_target=["attn-linear"]), attn)
    bs, seq = 2, 8
    h = torch.randn(bs, seq, cfg.hidden_size)
    cos = torch.randn(bs, seq, cfg.head_dim)
    sin = torch.randn(bs, seq, cfg.head_dim)
    out, weights = qattn(h, position_embeddings=(cos, sin))
    assert out.shape == (bs, seq, cfg.hidden_size)
    assert weights is None


# ---- QuantQwen35LinearAttn -----------------------------------------------


def _make_linear_attn(quant_target):
    cfg = _linear_attn_config()
    gm = Qwen3_5GatedDeltaNet(cfg, layer_idx=0)
    gm.config = cfg
    return QuantQwen35LinearAttn(_quant_args(quant_target=quant_target), gm)


def test_quant_qwen35_linear_attn_attn_linear_branch_uses_quant_linear():
    qla = _make_linear_attn(["attn-linear"])
    for proj in (qla.in_proj_qkv, qla.in_proj_z, qla.in_proj_b, qla.in_proj_a, qla.out_proj):
        assert isinstance(proj, QuantLinear)


def test_quant_qwen35_linear_attn_attn_linear_branch_has_activation_quantizers():
    qla = _make_linear_attn(["attn-linear"])
    assert isinstance(qla.inp_afq, ActivationQuantizer)
    assert isinstance(qla.o_proj_afq, ActivationQuantizer)


def test_quant_qwen35_linear_attn_passthrough_branch_uses_plain_linear():
    qla = _make_linear_attn(["attn-cache"])
    for proj in (qla.in_proj_qkv, qla.in_proj_z, qla.in_proj_b, qla.in_proj_a, qla.out_proj):
        assert isinstance(proj, PlainLinear)
    assert qla.input_transform is None
    assert qla.out_transform is None


def test_quant_qwen35_linear_attn_passthrough_branch_uses_identity_afq():
    qla = _make_linear_attn(["attn-cache"])
    assert isinstance(qla.inp_afq, nn.Identity)
    assert isinstance(qla.o_proj_afq, nn.Identity)


def test_quant_qwen35_linear_attn_enable_attn_linear_flag():
    qla_linear = _make_linear_attn(["attn-linear"])
    assert qla_linear.enable_attn_linear is True
    qla_cache = _make_linear_attn(["attn-cache"])
    assert qla_cache.enable_attn_linear is False


def test_quant_qwen35_linear_attn_hidden_size():
    qla = _make_linear_attn(["attn-linear"])
    assert qla.hidden_size == 64


@pytest.mark.parametrize("quant_target", [["attn-linear"], ["attn-cache"]])
def test_quant_qwen35_linear_attn_forward_returns_correct_shape(quant_target):
    qla = _make_linear_attn(quant_target)
    x = torch.randn(1, 4, 64)
    out = qla(x)
    assert out.shape == x.shape


def test_quant_qwen35_linear_attn_forward_batch():
    qla = _make_linear_attn(["attn-linear"])
    bs, seq = 2, 8
    x = torch.randn(bs, seq, 64)
    out = qla(x)
    assert out.shape == (bs, seq, 64)


def test_quant_qwen35_linear_attn_delegated_attributes():
    qla = _make_linear_attn(["attn-linear"])
    assert hasattr(qla, "conv1d")
    assert hasattr(qla, "dt_bias")
    assert hasattr(qla, "A_log")
    assert hasattr(qla, "norm")


class _IdentityTransform(nn.Module):
    def forward(self, x, **kwargs):
        return x


def test_quant_qwen35_linear_attn_forward_with_structure_transform():
    """Cover input_transform / out_transform branches (lines 101, 145)."""
    qla = _make_linear_attn(["attn-linear"])
    qla.input_transform = _IdentityTransform()
    qla.out_transform = _IdentityTransform()
    x = torch.randn(1, 4, 64)
    out = qla(x)
    assert out.shape == x.shape


def test_quant_qwen35_attn_forward_with_structure_transform():
    """Cover input_transform / out_transform branches (lines 229, 269)."""
    cfg = _tiny_config()
    attn = Qwen3_5Attention(cfg, layer_idx=0)
    qattn = QuantQwen35Attn(_quant_args(quant_target=["attn-linear"]), attn)
    qattn.input_transform = _IdentityTransform()
    qattn.out_transform = _IdentityTransform()
    bs, seq = 1, 4
    h = torch.randn(bs, seq, cfg.hidden_size)
    cos = torch.randn(bs, seq, cfg.head_dim)
    sin = torch.randn(bs, seq, cfg.head_dim)
    out, weights = qattn(h, position_embeddings=(cos, sin))
    assert out.shape == h.shape

