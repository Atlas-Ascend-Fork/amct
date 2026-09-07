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

from amct_pytorch.quantization.bit_policy import BitPolicy
from amct_pytorch.quantization.dtypes import register_dtype
from amct_pytorch.quantization.modules.quant_linear import QuantLinear


register_dtype()


class _SituAndMul(nn.Module):
    def forward(self, x):
        gate, up = x.chunk(2, dim=-1)
        return torch.tanh(gate) * torch.sigmoid(gate) * up


class _DenseMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_size = 8
        self.intermediate_size = 12
        self.config = SimpleNamespace(hidden_act="situ")
        self.act_fn = _SituAndMul()
        self.gate_proj = nn.Linear(8, 12, bias=False)
        self.up_proj = nn.Linear(8, 12, bias=False)
        self.down_proj = nn.Linear(12, 8, bias=False)

    def forward(self, x):
        gate_up = torch.cat([self.gate_proj(x), self.up_proj(x)], dim=-1)
        return self.down_proj(self.act_fn(gate_up))


class _Expert(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_dim = 8
        self.ffn_dim = 12
        self.config = SimpleNamespace(hidden_act="situ")
        self.act_fn = _SituAndMul()
        self.w1 = nn.Linear(8, 12, bias=False)
        self.w2 = nn.Linear(12, 8, bias=False)
        self.w3 = nn.Linear(8, 12, bias=False)

    def forward(self, x):
        return self.w2(self.act_fn(torch.cat([self.w1(x), self.w3(x)], dim=-1)))


class _Attention(nn.Module):
    def __init__(self):
        super().__init__()
        for name in (
            "q_proj", "k_proj", "v_proj", "f_a_proj", "f_b_proj",
            "b_proj", "g_a_proj", "g_b_proj", "o_proj",
        ):
            setattr(self, name, nn.Linear(8, 8, bias=False))
        self.q_conv1d = nn.Conv1d(8, 8, 1)
        self.A_log = nn.Parameter(torch.ones(2))

    def forward(self, x):
        return self.o_proj(self.q_proj(x))


class _MoeBlock(nn.Module):
    def __init__(self, num_experts):
        super().__init__()
        self.self_attn = nn.Linear(2, 2, bias=False)
        self.input_layernorm = nn.Identity()
        self.block_sparse_moe = nn.Module()
        self.block_sparse_moe.gate = nn.Linear(2, num_experts, bias=False)
        self.block_sparse_moe.shared_experts = nn.Linear(2, 2, bias=False)
        self.block_sparse_moe.experts = nn.ModuleList(
            [nn.Linear(2, 2, bias=False) for _ in range(num_experts)]
        )


def _args(target):
    groups = {
        "attn-linear": {"w_bits": 8, "a_bits": 8},
        "mlp": {"w_bits": 8, "a_bits": 8},
        "moe": {
            "routed": {"w_bits": 8, "a_bits": 8},
            "shared": {"w_bits": 8, "a_bits": 8},
        },
    }
    return SimpleNamespace(
        algos=[], quant_dtype="int", quant_target=list(target),
        w_bits=8, a_bits=8, bit_policy=BitPolicy(groups), device="cpu",
    )


def _disable_quant(module):
    from amct_pytorch.common.models.llm.common.quant_apply import (
        set_model_act_quant_state,
        set_model_weight_quant_state,
    )

    set_model_act_quant_state(module, False)
    set_model_weight_quant_state(module, False)


def test_quant_kimi_mlp_preserves_situ_forward_when_quant_is_disabled():
    from amct_pytorch.common.models.llm.kimi.kimi_k3.quant_module import QuantKimiMLP

    torch.manual_seed(0)
    float_mlp = _DenseMLP()
    x = torch.randn(2, 3, 8)
    expected = float_mlp(x)

    quant_mlp = QuantKimiMLP(_args(["mlp"]), float_mlp, group="mlp")
    _disable_quant(quant_mlp)

    assert isinstance(quant_mlp.gate_proj, QuantLinear)
    assert isinstance(quant_mlp.up_proj, QuantLinear)
    assert isinstance(quant_mlp.down_proj, QuantLinear)
    torch.testing.assert_close(quant_mlp(x), expected)


def test_quant_kimi_expert_keeps_checkpoint_projection_names():
    from amct_pytorch.common.models.llm.kimi.kimi_k3.quant_module import QuantKimiMLP

    torch.manual_seed(1)
    expert = _Expert()
    x = torch.randn(4, 8)
    expected = expert(x)

    quant_expert = QuantKimiMLP(_args(["moe"]), expert, group="moe.routed")
    _disable_quant(quant_expert)

    assert isinstance(quant_expert.w1, QuantLinear)
    assert isinstance(quant_expert.w2, QuantLinear)
    assert isinstance(quant_expert.w3, QuantLinear)
    assert not hasattr(quant_expert, "gate_proj")
    torch.testing.assert_close(quant_expert(x), expected)


def test_quantize_kimi_attention_only_replaces_selected_linears():
    from amct_pytorch.common.models.llm.kimi.kimi_k3.quant_module import quantize_kimi_attention

    attention = _Attention()
    conv = attention.q_conv1d
    a_log = attention.A_log

    result = quantize_kimi_attention(_args(["attn-linear"]), attention)

    assert result is attention
    for name in (
        "q_proj", "k_proj", "v_proj", "f_a_proj", "f_b_proj",
        "b_proj", "g_a_proj", "g_b_proj", "o_proj",
    ):
        assert isinstance(getattr(result, name), QuantLinear)
    assert result.q_conv1d is conv
    assert result.A_log is a_log


def test_kimi_adapter_builds_dense_and_moe_targets_without_cross_wrapping():
    from amct_pytorch.common.models.llm.kimi.kimi_k3.kimi_k3 import KimiK3
    from amct_pytorch.common.models.llm.kimi.kimi_k3.quant_module import QuantKimiMLP

    block = nn.Module()
    block.self_attn = _Attention()
    block.mlp = _DenseMLP()
    block.block_sparse_moe = nn.Module()
    block.block_sparse_moe.experts = nn.ModuleList([_Expert(), _Expert()])
    block.block_sparse_moe.shared_experts = _DenseMLP()

    pipeline = KimiK3.__new__(KimiK3)
    pipeline.args = _args(["attn-linear", "mlp", "moe"])
    pipeline.quant_target = pipeline.args.quant_target
    pipeline.block = lambda _: block

    quant_block = pipeline.build_quant_block(3)

    assert isinstance(quant_block.mlp, QuantKimiMLP)
    assert all(isinstance(expert, QuantKimiMLP) for expert in quant_block.block_sparse_moe.experts)
    assert isinstance(quant_block.block_sparse_moe.shared_experts, QuantKimiMLP)
    assert isinstance(quant_block.self_attn.q_proj, QuantLinear)

    units = list(pipeline.iter_ptq_units(3, quant_block))
    assert [(unit.kind, unit.name) for unit in units] == [
        ("attn", "self_attn"),
        ("mlp", "mlp"),
        ("moe", "expert_0"),
        ("moe", "expert_1"),
    ]


@pytest.mark.parametrize("n_npu", [4, 8])
def test_kimi_device_map_evenly_distributes_all_routed_experts(monkeypatch, n_npu):
    from amct_pytorch.common.models.llm.kimi.kimi_k3.kimi_k3 import KimiK3

    monkeypatch.setattr(
        torch, "npu", SimpleNamespace(device_count=lambda: n_npu), raising=False,
    )
    pipeline = KimiK3.__new__(KimiK3)
    pipeline.args = SimpleNamespace(device="npu:0")

    device_map = pipeline._build_block_device_map(_MoeBlock(896))

    assert device_map[""] == "npu:0"
    counts = [0] * n_npu
    for expert_idx in range(896):
        device = device_map[f"block_sparse_moe.experts.{expert_idx}"]
        counts[int(device.split(":")[1])] += 1
    assert counts == [896 // n_npu] * n_npu
    assert pipeline._resolve_param_device(
        "self_attn.weight", device_map,
    ) == "npu:0"
    assert pipeline._resolve_param_device(
        "block_sparse_moe.shared_experts.weight", device_map,
    ) == "npu:0"


def test_kimi_single_npu_keeps_existing_dispatch(monkeypatch):
    from amct_pytorch.common.models.llm.kimi.kimi_k3.kimi_k3 import KimiK3

    monkeypatch.setattr(
        torch, "npu", SimpleNamespace(device_count=lambda: 1), raising=False,
    )
    pipeline = KimiK3.__new__(KimiK3)
    pipeline.args = SimpleNamespace(device="npu:0")

    assert pipeline._build_block_device_map(_MoeBlock(8)) is None


def test_kimi_multi_npu_dispatch_pins_experts_and_returns_outputs_to_root(monkeypatch):
    from amct_pytorch.common.models.llm.kimi.kimi_k3.kimi_k3 import KimiK3

    monkeypatch.setattr(
        torch, "npu", SimpleNamespace(device_count=lambda: 4), raising=False,
    )
    pipeline = KimiK3.__new__(KimiK3)
    pipeline.args = SimpleNamespace(device="npu:0")
    block = _MoeBlock(8)

    result = pipeline._dispatch_block(block)

    assert result is block
    assert block._hf_hook.execution_device == "npu:0"
    for expert_idx, expert in enumerate(block.block_sparse_moe.experts):
        assert expert._hf_hook.execution_device == f"npu:{expert_idx % 4}"
        assert expert._hf_hook.io_same_device is True


def test_kimi_dispatch_places_quant_wrappers_on_their_owner_devices(monkeypatch):
    from amct_pytorch.common.models.llm.kimi.kimi_k3.kimi_k3 import KimiK3
    from amct_pytorch.common.models.llm.kimi.kimi_k3.quant_module import (
        QuantKimiLinear,
        QuantKimiMLP,
        quantize_kimi_attention,
    )

    monkeypatch.setattr(
        torch, "npu", SimpleNamespace(device_count=lambda: 4), raising=False,
    )
    pipeline = KimiK3.__new__(KimiK3)
    pipeline.args = _args(["attn-linear", "moe"])
    pipeline.args.device = "npu:0"
    block = nn.Module()
    block.self_attn = quantize_kimi_attention(pipeline.args, _Attention())
    block.block_sparse_moe = nn.Module()
    block.block_sparse_moe.experts = nn.ModuleList(
        [QuantKimiMLP(pipeline.args, _Expert(), group="moe.routed") for _ in range(4)]
    )
    block.block_sparse_moe.shared_experts = QuantKimiMLP(
        pipeline.args, _DenseMLP(), group="moe.shared",
    )
    placements = {}

    def record_placement(module, device):
        placements[id(module)] = device
        return module

    monkeypatch.setattr(QuantKimiMLP, "to", record_placement)
    monkeypatch.setattr(QuantKimiLinear, "to", record_placement)

    pipeline._dispatch_block(block)

    assert placements[id(block.block_sparse_moe.shared_experts)] == "npu:0"
    for expert_idx, expert in enumerate(block.block_sparse_moe.experts):
        assert placements[id(expert)] == f"npu:{expert_idx}"
    for module in block.self_attn.modules():
        if isinstance(module, QuantKimiLinear):
            assert placements[id(module)] == "npu:0"
