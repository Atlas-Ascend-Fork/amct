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

import pytest
import torch
import torch.nn as nn

from amct_pytorch.common.models.llm.common.moe_unpack import (
    ExpertLinearView,
    GatedExpertView,
    find_moe_module,
)

# ---- ExpertLinearView ----------------------------------------------------


class _FakeExperts:
    """Stand-in for an MoE experts container: holds packed [E, ...] tensors."""

    def __init__(self):
        # Two experts, packed gate/up: [E, 2*intermediate, hidden] = [2, 6, 4]
        self.gate_up_proj = torch.arange(2 * 6 * 4, dtype=torch.float32).reshape(2, 6, 4)
        self.down_proj = torch.arange(2 * 4 * 3, dtype=torch.float32).reshape(2, 4, 3) + 1000


def test_expert_linear_view_returns_full_slice_when_no_range():
    experts = _FakeExperts()
    view = ExpertLinearView(experts, expert_idx=1, weight_name="down_proj")
    assert torch.equal(view.weight, experts.down_proj[1])


def test_expert_linear_view_returns_partial_range():
    experts = _FakeExperts()
    view = ExpertLinearView(experts, expert_idx=0, weight_name="gate_up_proj", start=0, end=3)
    assert torch.equal(view.weight, experts.gate_up_proj[0, 0:3])


def test_expert_linear_view_lazy_reflects_external_updates():
    experts = _FakeExperts()
    view = ExpertLinearView(experts, expert_idx=0, weight_name="down_proj")
    new_value = torch.zeros_like(experts.down_proj[0])
    experts.down_proj[0] = new_value
    assert torch.equal(view.weight, new_value)


def test_expert_linear_view_materialize_clones_and_decouples():
    experts = _FakeExperts()
    view = ExpertLinearView(
        experts, expert_idx=0, weight_name="down_proj", materialize=True
    )
    snapshot = view.weight.clone()
    # Mutating the source should not affect the materialized copy.
    experts.down_proj[0].fill_(-1.0)
    assert torch.equal(view.weight, snapshot)
    assert isinstance(view._weight, nn.Parameter)
    assert view._weight.requires_grad is False


def test_expert_linear_view_bias_default_is_none():
    view = ExpertLinearView(_FakeExperts(), 0, "down_proj")
    assert view.bias is None


# ---- GatedExpertView -----------------------------------------------------


class _GatedExperts(_FakeExperts):
    hidden_dim = 4
    intermediate_dim = 3
    act_fn = staticmethod(lambda x: x)


def test_gated_expert_view_splits_gate_up_into_two_views():
    experts = _GatedExperts()
    gv = GatedExpertView(experts, expert_idx=0)
    # gate_proj views rows [0:intermediate] of gate_up_proj
    assert torch.equal(gv.gate_proj.weight, experts.gate_up_proj[0, 0:3])
    # up_proj views rows [intermediate:] of gate_up_proj
    assert torch.equal(gv.up_proj.weight, experts.gate_up_proj[0, 3:])
    # down_proj is the full down_proj slice
    assert torch.equal(gv.down_proj.weight, experts.down_proj[0])


def test_gated_expert_view_records_dimensions_and_act_fn():
    experts = _GatedExperts()
    gv = GatedExpertView(experts, expert_idx=1)
    assert gv.hidden_size == experts.hidden_dim
    assert gv.intermediate_size == experts.intermediate_dim
    assert gv.act_fn is experts.act_fn


def test_gated_expert_view_materialize_passes_through():
    experts = _GatedExperts()
    gv = GatedExpertView(experts, expert_idx=0, materialize=True)
    assert isinstance(gv.gate_proj._weight, nn.Parameter)
    assert isinstance(gv.up_proj._weight, nn.Parameter)
    assert isinstance(gv.down_proj._weight, nn.Parameter)


# ---- find_moe_module -----------------------------------------------------


def test_find_moe_module_returns_mlp_when_it_has_experts():
    block = nn.Module()
    block.mlp = nn.Module()
    block.mlp.experts = nn.ModuleList()
    assert find_moe_module(block) is block.mlp


def test_find_moe_module_descends_when_mlp_has_no_experts():
    block = nn.Module()
    block.mlp = nn.Linear(4, 4)  # no `experts` attribute
    block.other = nn.Module()
    block.other.experts = nn.ModuleList()
    assert find_moe_module(block) is block.other


def test_find_moe_module_returns_none_when_no_match():
    block = nn.Module()
    block.mlp = nn.Linear(4, 4)
    block.other = nn.Linear(4, 4)
    assert find_moe_module(block) is None
