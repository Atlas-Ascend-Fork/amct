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

"""Tests for QuantLongcatMLA / QuantLongcatMLP and packed-expert helpers."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from transformers.models.longcat_flash.configuration_longcat_flash import (
    LongcatFlashConfig,
)
from transformers.models.longcat_flash.modeling_longcat_flash import (
    LongcatFlashMLA,
    LongcatFlashMLP,
)

from amct_pytorch.common.models.llm.common.quant_apply import PlainLinear
from amct_pytorch.common.models.llm.longcat.longcat_lite.quant_module import (
    LongcatPackedExpertLinearView,
    LongcatPackedExpertView,
    LongcatUnpackedExperts,
    QuantLongcatMLA,
    QuantLongcatMLP,
)
from amct_pytorch.quantization.bit_policy import BitPolicy
from amct_pytorch.quantization.dtypes import register_dtype
from amct_pytorch.quantization.modules.quant_linear import QuantLinear

register_dtype()

W_BITS = 'w_bits'

A_BITS = 'a_bits'


def _tiny_config():
    return LongcatFlashConfig(
        hidden_size=64,
        num_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=32,
        max_position_embeddings=64,
        ffn_hidden_size=128,
        q_lora_rank=32,
        kv_lora_rank=16,
        qk_rope_head_dim=8,
        qk_nope_head_dim=8,
        v_head_dim=8,
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
            "mlp": {
                "gate_proj": {W_BITS: 8, A_BITS: 8},
                "up_proj": {W_BITS: 8, A_BITS: 8},
                "down_proj": {W_BITS: 8, A_BITS: 8},
            },
            "attn-linear": {
                "q_proj": {W_BITS: 8, A_BITS: 8},
                "k_proj": {W_BITS: 8, A_BITS: 8},
                "v_proj": {W_BITS: 8, A_BITS: 8},
                "o_proj": {W_BITS: 8, A_BITS: 8},
            },
            "attn-cache": {"q": 8, "k": 8, "p": 8, "v": 8},
        }),
    )


# ---- LongcatPackedExpertLinearView --------------------------------------


def test_packed_expert_linear_view_returns_full_slice():
    experts = SimpleNamespace(gate_up_proj=torch.randn(2, 6, 4))
    view = LongcatPackedExpertLinearView(experts, expert_idx=1, weight_name="gate_up_proj")
    assert torch.equal(view.weight, experts.gate_up_proj[1])


def test_packed_expert_linear_view_returns_partial_slice():
    experts = SimpleNamespace(gate_up_proj=torch.arange(2 * 6 * 4, dtype=torch.float32).reshape(2, 6, 4))
    view = LongcatPackedExpertLinearView(experts, expert_idx=0, weight_name="gate_up_proj", start=0, end=3)
    assert torch.equal(view.weight, experts.gate_up_proj[0, 0:3])


def test_packed_expert_linear_view_bias_default_none():
    experts = SimpleNamespace(gate_up_proj=torch.zeros(2, 6, 4))
    view = LongcatPackedExpertLinearView(experts, expert_idx=0, weight_name="gate_up_proj")
    assert view.bias is None


# ---- LongcatPackedExpertView --------------------------------------------


class _PackedExperts:
    hidden_size = 4
    intermediate_size = 3
    act_fn = staticmethod(torch.nn.functional.silu)
    gate_up_proj = torch.arange(2 * 6 * 4, dtype=torch.float32).reshape(2, 6, 4)
    down_proj = torch.zeros(2, 4, 3)


def test_packed_expert_view_constructs_three_views():
    experts = _PackedExperts()
    view = LongcatPackedExpertView(experts, expert_idx=0)
    # gate_proj [0:intermediate=3], up_proj [3:None]
    assert torch.equal(view.gate_proj.weight, experts.gate_up_proj[0, 0:3])
    assert torch.equal(view.up_proj.weight, experts.gate_up_proj[0, 3:])
    assert torch.equal(view.down_proj.weight, experts.down_proj[0])
    assert view.act_fn is experts.act_fn


# ---- LongcatUnpackedExperts ---------------------------------------------


class _FakePackedExpertsForUnpack(nn.Module):
    """Minimal fake matching the attribute surface QuantLongcatMLP needs."""

    def __init__(self, num_routed=2, total=3, hidden_size=8, intermediate_size=16):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.act_fn = nn.SiLU()
        self.num_routed_experts = num_routed
        self.total_experts = total
        self.register_buffer(
            "gate_up_proj",
            torch.randn(num_routed, intermediate_size * 2, hidden_size),
        )
        self.register_buffer(
            "down_proj",
            torch.randn(num_routed, hidden_size, intermediate_size),
        )


def test_unpacked_experts_creates_one_module_per_routed_expert_and_pads_with_identity():
    args = _quant_args()
    experts = _FakePackedExpertsForUnpack(num_routed=2, total=3)
    unpacked = LongcatUnpackedExperts(args, experts)
    assert len(unpacked.expert_modules) == 3
    # First two are QuantLongcatMLP (routed), last one is Identity padding.
    assert isinstance(unpacked.expert_modules[0], QuantLongcatMLP)
    assert isinstance(unpacked.expert_modules[1], QuantLongcatMLP)
    assert isinstance(unpacked.expert_modules[2], nn.Identity)
    assert unpacked.num_routed_experts == 2
    assert unpacked.total_experts == 3


def test_unpacked_experts_forward_returns_zeros_when_top_k_index_empty():
    args = _quant_args()
    experts = _FakePackedExpertsForUnpack()
    unpacked = LongcatUnpackedExperts(args, experts)
    h = torch.randn(2, experts.hidden_size)
    empty_top_k = torch.empty(0, 1, dtype=torch.long)
    weights = torch.empty(0, 1)
    out = unpacked(h, empty_top_k, weights)
    assert torch.equal(out, torch.zeros_like(h))


# ---- QuantLongcatMLP -----------------------------------------------------


def test_quant_longcat_mlp_forward_preserves_shape():
    cfg = _tiny_config()
    mlp = LongcatFlashMLP(cfg, intermediate_size=cfg.ffn_hidden_size)
    qmlp = QuantLongcatMLP(_quant_args(quant_target=["mlp"]), mlp)
    x = torch.randn(1, 4, cfg.hidden_size)
    assert qmlp(x).shape == x.shape


def test_quant_longcat_mlp_uses_quant_linear():
    cfg = _tiny_config()
    qmlp = QuantLongcatMLP(
        _quant_args(quant_target=["mlp"]),
        LongcatFlashMLP(cfg, intermediate_size=cfg.ffn_hidden_size),
    )
    assert isinstance(qmlp.up_proj, QuantLinear)
    assert isinstance(qmlp.gate_proj, QuantLinear)
    assert isinstance(qmlp.down_proj, QuantLinear)


# ---- QuantLongcatMLA -----------------------------------------------------


def test_quant_longcat_mla_attn_linear_branch_uses_quant_linear():
    cfg = _tiny_config()
    attn = LongcatFlashMLA(cfg, layer_idx=0)
    qattn = QuantLongcatMLA(_quant_args(quant_target=["attn-linear"]), attn)
    # When q_lora_rank is set, the LoRA q_a/q_b path is used; q_proj is absent.
    assert not hasattr(qattn, "q_proj")
    assert isinstance(qattn.q_a_proj, QuantLinear)
    assert isinstance(qattn.q_b_proj, QuantLinear)
    assert isinstance(qattn.kv_a_proj_with_mqa, QuantLinear)
    assert isinstance(qattn.kv_b_proj, QuantLinear)
    assert isinstance(qattn.o_proj, QuantLinear)


def test_quant_longcat_mla_passthrough_branch_uses_plain_linear():
    cfg = _tiny_config()
    attn = LongcatFlashMLA(cfg, layer_idx=0)
    qattn = QuantLongcatMLA(_quant_args(quant_target=["attn-cache"]), attn)
    assert isinstance(qattn.q_a_proj, PlainLinear)
    assert isinstance(qattn.kv_a_proj_with_mqa, PlainLinear)
    assert isinstance(qattn.kv_b_proj, PlainLinear)
    assert isinstance(qattn.o_proj, PlainLinear)
    assert qattn.input_transform is None
    assert qattn.out_transform is None
    assert isinstance(qattn.inp_afq, nn.Identity)


def test_quant_longcat_mla_qk_pv_matmul_built():
    cfg = _tiny_config()
    attn = LongcatFlashMLA(cfg, layer_idx=0)
    qattn = QuantLongcatMLA(_quant_args(), attn)
    assert qattn.qk_matmul is not None
    assert qattn.pv_matmul is not None
    # Bit widths should propagate from quant_args.
    assert qattn.qk_matmul.l_bits == 8
    assert qattn.pv_matmul.r_bits == 8


def _mla_config():
    return LongcatFlashConfig(
        hidden_size=64,
        num_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        vocab_size=32,
        max_position_embeddings=64,
        ffn_hidden_size=128,
        q_lora_rank=32,
        kv_lora_rank=16,
        qk_rope_head_dim=8,
        qk_nope_head_dim=8,
        v_head_dim=8,
    )


def _mla_attn_cache_args():
    return SimpleNamespace(
        algos=[],
        quant_dtype="int",
        w_bits=8, a_bits=8, q_bits=8, k_bits=8, p_bits=8, v_bits=8,
        quant_target=["attn-cache"],
        bit_policy=BitPolicy({
            "mlp": {},
            "attn-linear": {
                "q_proj": {"w_bits": 8, "a_bits": 8},
                "k_proj": {"w_bits": 8, "a_bits": 8},
                "v_proj": {"w_bits": 8, "a_bits": 8},
                "o_proj": {"w_bits": 8, "a_bits": 8},
                "kv_a_proj_with_mqa": {"w_bits": 8, "a_bits": 8},
                "kv_b_proj": {"w_bits": 8, "a_bits": 8},
                "q_a_proj": {"w_bits": 8, "a_bits": 8},
                "q_b_proj": {"w_bits": 8, "a_bits": 8},
            },
            "attn-cache": {"q": 8, "k": 8, "p": 8, "v": 8},
        }),
    )


def _mla_attn_linear_args():
    return SimpleNamespace(
        algos=[],
        quant_dtype="int",
        w_bits=8, a_bits=8, q_bits=8, k_bits=8, p_bits=8, v_bits=8,
        quant_target=["attn-linear", "attn-cache"],
        bit_policy=BitPolicy({
            "mlp": {},
            "attn-linear": {
                "q_proj": {"w_bits": 8, "a_bits": 8},
                "k_proj": {"w_bits": 8, "a_bits": 8},
                "v_proj": {"w_bits": 8, "a_bits": 8},
                "o_proj": {"w_bits": 8, "a_bits": 8},
                "kv_a_proj_with_mqa": {"w_bits": 8, "a_bits": 8},
                "kv_b_proj": {"w_bits": 8, "a_bits": 8},
                "q_a_proj": {"w_bits": 8, "a_bits": 8},
                "q_b_proj": {"w_bits": 8, "a_bits": 8},
            },
            "attn-cache": {"q": 8, "k": 8, "p": 8, "v": 8},
        }),
    )


def test_quant_longcat_mla_forward_attn_cache():
    cfg = _mla_config()
    attn = LongcatFlashMLA(cfg, layer_idx=0)
    qattn = QuantLongcatMLA(_mla_attn_cache_args(), attn)
    batch, seq = 1, 4
    hidden = torch.randn(batch, seq, cfg.hidden_size)
    cos = torch.randn(1, seq, attn.qk_rope_head_dim)
    sin = torch.randn(1, seq, attn.qk_rope_head_dim)
    out, _ = qattn(hidden, (cos, sin), attention_mask=None)
    assert out.shape == (batch, seq, cfg.hidden_size)


def test_quant_longcat_mla_forward_attn_linear():
    cfg = _mla_config()
    attn = LongcatFlashMLA(cfg, layer_idx=0)
    qattn = QuantLongcatMLA(_mla_attn_linear_args(), attn)
    batch, seq = 1, 4
    hidden = torch.randn(batch, seq, cfg.hidden_size)
    cos = torch.randn(1, seq, attn.qk_rope_head_dim)
    sin = torch.randn(1, seq, attn.qk_rope_head_dim)
    out, _ = qattn(hidden, (cos, sin), attention_mask=None)
    assert out.shape == (batch, seq, cfg.hidden_size)


def test_unpacked_experts_forward_non_empty():
    args = _quant_args(quant_target=["mlp"])
    experts = _FakePackedExpertsForUnpack(num_routed=2, total=3)
    unpacked = LongcatUnpackedExperts(args, experts)
    h = torch.randn(4, experts.hidden_size)
    top_k_index = torch.tensor([[0], [1], [0], [1]], dtype=torch.long)
    top_k_weights = torch.tensor([[1.0], [0.5], [0.8], [0.3]])
    out = unpacked(h, top_k_index, top_k_weights)
    assert out.shape == (4, experts.hidden_size)

