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
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from amct_pytorch.common.models.llm.qwen.moe_common import (
    QuantGatedExperts,
    is_packed_experts,
    pack_gated_expert_weights,
)
from amct_pytorch.quantization.bit_policy import BitPolicy
from amct_pytorch.quantization.dtypes import register_dtype

register_dtype()


_MLP_BP = BitPolicy({
    "mlp": {
        "gate_proj": {"w_bits": 8, "a_bits": 8},
        "up_proj": {"w_bits": 8, "a_bits": 8},
        "down_proj": {"w_bits": 8, "a_bits": 8},
    },
    "moe": {
        "routed": {
            "gate_proj": {"w_bits": 8, "a_bits": 8},
            "up_proj": {"w_bits": 8, "a_bits": 8},
            "down_proj": {"w_bits": 8, "a_bits": 8},
        },
    },
})


def _make_args(quant_target=("moe",)):
    return SimpleNamespace(
        algos=[], quant_dtype="int", w_bits=8, a_bits=8,
        quant_target=list(quant_target), bit_policy=_MLP_BP,
    )


# ---- is_packed_experts ----------------------------------------------------


class _Packed:
    gate_up_proj = "x"
    down_proj = "y"


class _Unpacked:
    pass


def test_is_packed_experts_true_when_both_attrs_present():
    assert is_packed_experts(_Packed()) is True


def test_is_packed_experts_false_when_attr_missing():
    assert is_packed_experts(_Unpacked()) is False


def test_is_packed_experts_false_when_only_one_attr_present():
    class _Half:
        gate_up_proj = "x"

    assert is_packed_experts(_Half()) is False


# ---- pack_gated_expert_weights -------------------------------------------


def _expert_weights(num_experts=2, hidden=4, intermediate=3):
    sd = {}
    for idx in range(num_experts):
        sd[f"mlp.experts.{idx}.gate_proj.weight"] = torch.full((intermediate, hidden), float(idx) + 0.1)
        sd[f"mlp.experts.{idx}.up_proj.weight"] = torch.full((intermediate, hidden), float(idx) + 0.2)
        sd[f"mlp.experts.{idx}.down_proj.weight"] = torch.full((hidden, intermediate), float(idx) + 0.3)
    return sd


def test_pack_gated_expert_weights_combines_gate_up_into_single_tensor():
    sd = _expert_weights(num_experts=2, hidden=4, intermediate=3)
    packed = pack_gated_expert_weights(sd)
    # Original per-expert keys removed.
    assert "mlp.experts.0.gate_proj.weight" not in packed
    assert "mlp.experts.0.up_proj.weight" not in packed
    # Stacked tensors expose the expected shape.
    gu = packed["mlp.experts.gate_up_proj"]
    dp = packed["mlp.experts.down_proj"]
    assert gu.shape == (2, 6, 4)        # (num_experts, 2*intermediate, hidden)
    assert dp.shape == (2, 4, 3)        # (num_experts, hidden, intermediate)


def test_pack_gated_expert_weights_preserves_unrelated_keys():
    sd = _expert_weights()
    sd["other.weight"] = torch.zeros(4)
    packed = pack_gated_expert_weights(sd)
    assert torch.equal(packed["other.weight"], torch.zeros(4))


def test_pack_gated_expert_weights_returns_input_when_no_expert_keys_match():
    sd = {"some.module.weight": torch.ones(4)}
    out = pack_gated_expert_weights(sd)
    assert out is sd


def test_pack_gated_expert_weights_respects_custom_prefix():
    hidden, inter = 4, 3
    sd = {}
    for idx in range(2):
        sd[f"backbone.experts.{idx}.gate_proj.weight"] = torch.zeros(inter, hidden)
        sd[f"backbone.experts.{idx}.up_proj.weight"] = torch.zeros(inter, hidden)
        sd[f"backbone.experts.{idx}.down_proj.weight"] = torch.zeros(hidden, inter)
    packed = pack_gated_expert_weights(sd, expert_prefix="backbone.experts")
    assert "backbone.experts.gate_up_proj" in packed


def test_pack_gated_expert_weights_orders_experts_by_index():
    # Insert in scrambled order; result must still be sorted by expert index.
    sd = _expert_weights(num_experts=3)
    scrambled = {k: sd[k] for k in reversed(list(sd))}
    packed = pack_gated_expert_weights(scrambled)
    gate_up = packed["mlp.experts.gate_up_proj"]
    # Expert 0's gate_proj rows are filled with 0.1.
    assert gate_up[0, 0, 0].item() == pytest.approx(0.1)
    assert gate_up[1, 0, 0].item() == pytest.approx(1.1)
    assert gate_up[2, 0, 0].item() == pytest.approx(2.1)


def test_pack_gated_expert_weights_raises_on_inconsistent_experts():
    sd = _expert_weights(num_experts=2)
    # Drop one expert's down_proj to create asymmetry.
    sd.pop("mlp.experts.1.down_proj.weight")
    with pytest.raises(KeyError, match="Inconsistent expert weights"):
        pack_gated_expert_weights(sd)


# ---- QuantGatedExperts ----------------------------------------------------


class _FakeExpertModule(nn.Module):
    """A stand-in GatedExpertView compatible with QuantGatedMLP."""
    def __init__(self, experts_module, expert_idx=0, hidden_attr="hidden_dim",
                 intermediate_attr="intermediate_dim", act_attr="act_fn",
                 gate_up_name="gate_up_proj", down_name="down_proj",
                 materialize=False):
        super().__init__()
        self.hidden_size = 4
        self.intermediate_size = 8
        self.act_fn = nn.SiLU()
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size)


class _FakePackedExperts:
    """Minimal packed-expert module with num_experts attr."""
    def __init__(self, num_experts=2, **kwargs):
        self.num_experts = num_experts


def test_quant_gated_experts_init_creates_expert_modules(monkeypatch):
    args = _make_args()
    pe = _FakePackedExperts(num_experts=2)
    qge = QuantGatedExperts(args, pe, view_cls=_FakeExpertModule, group="moe.routed")
    assert qge.num_experts == 2
    assert qge.group == "moe.routed"
    assert len(qge.expert_modules) == 2


def test_quant_gated_experts_build_ptq_expert_module_returns_materialized(monkeypatch):
    args = _make_args()
    pe = _FakePackedExperts(num_experts=2, hidden=4, intermediate=8)
    qge = QuantGatedExperts(args, pe, view_cls=_FakeExpertModule, group="moe.routed")
    m = qge.build_ptq_expert_module(0)
    assert m.hidden_size == 4


def test_quant_gated_experts_iter_ptq_expert_modules_yields_all(monkeypatch):
    args = _make_args()
    pe = _FakePackedExperts(num_experts=3, hidden=4, intermediate=8)
    qge = QuantGatedExperts(args, pe, view_cls=_FakeExpertModule, group="moe.routed")
    modules = list(qge.iter_ptq_expert_modules())
    assert len(modules) == 3


def test_quant_gated_experts_forward_empty_top_k_returns_zeros(monkeypatch):
    args = _make_args()
    pe = _FakePackedExperts(num_experts=2, hidden=4, intermediate=8)
    qge = QuantGatedExperts(args, pe, view_cls=_FakeExpertModule, group="moe.routed")
    hs = torch.randn(2, 4)
    top_k_index = torch.empty(2, 0, dtype=torch.long)
    top_k_weights = torch.empty(2, 0)
    out = qge.forward(hs, top_k_index, top_k_weights)
    assert torch.equal(out, torch.zeros_like(hs))


def test_is_packed_experts_returns_false_for_non_packed():
    experts = nn.ModuleList([nn.Linear(4, 8) for _ in range(4)])
    assert is_packed_experts(experts) is False


def test_pack_gated_expert_weights_raises_on_missing_key():
    state_dict = {
        "mlp.experts.0.gate_proj.weight": torch.randn(4, 8),
        "mlp.experts.0.up_proj.weight": torch.randn(4, 8),
    }
    with pytest.raises(KeyError):
        pack_gated_expert_weights(state_dict, expert_prefix="mlp.experts")


def test_pack_gated_expert_weights_packs_correctly():
    state_dict = {
        "mlp.experts.0.gate_proj.weight": torch.randn(4, 8),
        "mlp.experts.0.up_proj.weight": torch.randn(4, 8),
        "mlp.experts.0.down_proj.weight": torch.randn(8, 4),
        "mlp.experts.1.gate_proj.weight": torch.randn(4, 8),
        "mlp.experts.1.up_proj.weight": torch.randn(4, 8),
        "mlp.experts.1.down_proj.weight": torch.randn(8, 4),
    }
    result = pack_gated_expert_weights(state_dict, expert_prefix="mlp.experts")
    assert "mlp.experts.gate_up_proj" in result
    assert "mlp.experts.down_proj" in result
    assert "mlp.experts.0.gate_proj.weight" not in result
    assert result["mlp.experts.gate_up_proj"].shape == (2, 8, 8)
