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


class _MLAAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(
            hidden_size=8,
            max_position_embeddings=16,
            rope_theta=10000.0,
            rope_scaling=None,
            mla_use_output_gate=True,
        )
        self.layer_idx = 0
        self.hidden_size = 8
        self.num_heads = 2
        self.num_key_value_heads = 2
        self.num_key_value_groups = 1
        self.q_lora_rank = 4
        self.kv_lora_rank = 3
        self.qk_nope_head_dim = 2
        self.qk_rope_head_dim = 2
        self.q_head_dim = 4
        self.v_head_dim = 2
        self.scaling = self.q_head_dim ** -0.5
        self.attention_dropout = 0.0
        self.use_output_gate = True
        self.q_a_proj = nn.Linear(8, 4, bias=False)
        self.q_a_layernorm = nn.Identity()
        self.q_b_proj = nn.Linear(4, 8, bias=False)
        self.kv_a_proj_with_mqa = nn.Linear(8, 5, bias=False)
        self.kv_a_layernorm = nn.Identity()
        self.kv_b_proj = nn.Linear(3, 8, bias=False)
        self.g_proj = nn.Linear(8, 4, bias=False)
        self.o_proj = nn.Linear(4, 8, bias=False)

    @staticmethod
    def _rope(value, position_ids):
        inv_freq = 1.0 / (
            10000.0 ** (torch.arange(0, value.shape[-1], 2) / value.shape[-1])
        )
        angles = position_ids.float().unsqueeze(-1) * inv_freq
        cos, sin = angles.cos().unsqueeze(2), angles.sin().unsqueeze(2)
        pairs = value.float().reshape(*value.shape[:-1], -1, 2)
        even, odd = pairs.unbind(-1)
        return torch.stack(
            (even * cos - odd * sin, even * sin + odd * cos), dim=-1,
        ).flatten(-2).to(value.dtype)

    def forward(self, hidden_states, position_ids=None, attention_mask=None):
        batch_size, seq_length, _ = hidden_states.shape
        if position_ids is None:
            position_ids = torch.arange(seq_length).unsqueeze(0)
        q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
        q = q.view(batch_size, seq_length, self.num_heads, self.q_head_dim)
        q_nope, q_rope = torch.split(q, [2, 2], dim=-1)
        q_rope = self._rope(q_rope, position_ids)
        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        kv_nope, kv_rope = torch.split(compressed_kv, [3, 2], dim=-1)
        kv_nope = self.kv_a_layernorm(kv_nope)
        kv_rope = self._rope(kv_rope.unsqueeze(2), position_ids)
        expanded = self.kv_b_proj(kv_nope).view(
            batch_size, seq_length, self.num_heads, 4,
        )
        k_nope, value = torch.split(expanded, [2, 2], dim=-1)
        key = torch.cat((k_nope, kv_rope.expand(-1, -1, 2, -1)), dim=-1)
        query = torch.cat((q_nope, q_rope), dim=-1)
        scores = torch.einsum("bshd,bthd->bsht", query, key) * self.scaling
        if attention_mask is None:
            attention_mask = torch.full(
                (seq_length, seq_length), float("-inf"),
            ).triu(1).unsqueeze(0).unsqueeze(2)
        scores = scores + attention_mask
        probs = scores.softmax(dim=-1, dtype=torch.float32).to(hidden_states.dtype)
        output = torch.einsum("bsht,bthd->bshd", probs, value).flatten(2)
        output = output * self.g_proj(hidden_states).sigmoid()
        return self.o_proj(output)


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
        "attn-cache": {"kv_nope": 8, "kv_rope": 8},
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


def test_kimi_adapter_leaves_kda_untouched_for_attention_targets():
    from amct_pytorch.common.models.llm.kimi.kimi_k3.kimi_k3 import KimiK3

    block = nn.Module()
    block.is_linear_attn = True
    block.self_attn = _Attention()
    attention = block.self_attn
    pipeline = KimiK3.__new__(KimiK3)
    pipeline.args = _args(["attn-linear", "attn-cache"])
    pipeline.quant_target = pipeline.args.quant_target
    pipeline.block = lambda _: block

    result = pipeline.build_quant_block(0)

    assert result.self_attn is attention
    assert all(
        isinstance(getattr(result.self_attn, name), nn.Linear)
        for name in (
            "q_proj", "k_proj", "v_proj", "f_a_proj", "f_b_proj",
            "b_proj", "g_a_proj", "g_b_proj", "o_proj",
        )
    )


def test_quant_kimi_mla_only_quantizes_three_selected_projections():
    from amct_pytorch.common.models.llm.kimi.kimi_k3.quant_module import (
        QuantKimiMLAAttention,
    )

    source = _MLAAttention()
    result = QuantKimiMLAAttention(_args(["attn-linear"]), source)

    assert isinstance(result, QuantKimiMLAAttention)
    assert isinstance(result.q_a_proj, QuantLinear)
    assert isinstance(result.q_b_proj, QuantLinear)
    assert isinstance(result.kv_a_proj_with_mqa, QuantLinear)
    assert result.kv_b_proj is source.kv_b_proj
    assert result.g_proj is source.g_proj
    assert result.o_proj is source.o_proj


def test_quant_kimi_mla_quantizes_kv_rope_after_rotary(monkeypatch):
    from amct_pytorch.common.models.llm.kimi.kimi_k3 import quant_module

    source = _MLAAttention()
    wrapper = quant_module.QuantKimiMLAAttention(
        _args(["attn-cache"]), source,
    )
    assert wrapper.kv_nope_cache_quantizer.bits == 8
    assert wrapper.kv_rope_cache_quantizer.bits == 8
    seen = {}

    class _Record(nn.Module):
        def __init__(self, name):
            super().__init__()
            self.name = name

        def forward(self, value):
            seen[self.name] = value.detach().clone()
            return value

    def fake_rotary(value, freqs_cis):
        del freqs_cis
        return value + 10

    monkeypatch.setattr(quant_module, "apply_rotary_emb", fake_rotary)
    wrapper.kv_nope_cache_quantizer = _Record("kv_nope")
    wrapper.kv_rope_cache_quantizer = _Record("kv_rope")

    hidden_states = torch.randn(1, 3, 8)
    raw_kv = source.kv_a_proj_with_mqa(hidden_states)
    raw_kv_nope, raw_kv_rope = torch.split(raw_kv, [3, 2], dim=-1)
    output = wrapper(hidden_states, position_ids=torch.arange(3).unsqueeze(0))

    torch.testing.assert_close(seen["kv_nope"], raw_kv_nope)
    torch.testing.assert_close(seen["kv_rope"], raw_kv_rope.unsqueeze(2) + 10)
    assert output.shape == hidden_states.shape


def test_quant_kimi_mla_matches_expanded_float_path_when_quant_is_disabled():
    from amct_pytorch.common.models.llm.kimi.kimi_k3.quant_module import (
        QuantKimiMLAAttention,
    )

    torch.manual_seed(2)
    source = _MLAAttention()
    hidden_states = torch.randn(2, 4, 8)
    position_ids = torch.arange(4).unsqueeze(0)
    expected = source(hidden_states, position_ids=position_ids)
    wrapper = QuantKimiMLAAttention(_args(["attn-linear"]), source)
    _disable_quant(wrapper)

    actual = wrapper(hidden_states, position_ids=position_ids)

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_quant_kimi_mla_cache_accumulates_prefill_and_decode_states():
    from amct_pytorch.common.models.llm.kimi.kimi_k3.quant_module import (
        QuantKimiMLAAttention,
    )

    class _Cache:
        def __init__(self):
            self.layers = {}

        def update(self, key, value, layer_idx, cache_kwargs=None):
            del cache_kwargs
            if layer_idx in self.layers:
                old_key, old_value = self.layers[layer_idx]
                key = torch.cat((old_key, key), dim=-2)
                value = torch.cat((old_value, value), dim=-2)
            self.layers[layer_idx] = key, value
            return key, value

    source = _MLAAttention()
    wrapper = QuantKimiMLAAttention(_args(["attn-cache"]), source)
    cache = _Cache()
    prefill = wrapper(
        torch.randn(1, 2, 8),
        position_ids=torch.tensor([[0, 1]]),
        past_key_values=cache,
    )
    decode = wrapper(
        torch.randn(1, 1, 8),
        cache_position=torch.tensor([2]),
        past_key_values=cache,
    )

    latent_cache, rope_cache = cache.layers[0]
    assert prefill.shape == (1, 2, 8)
    assert decode.shape == (1, 1, 8)
    assert latent_cache.shape == (1, 1, 3, 3)
    assert rope_cache.shape == (1, 1, 3, 2)


def test_quant_kimi_mla_ptq_params_can_be_exported_and_reloaded():
    from amct_pytorch.common.models.llm.kimi.kimi_k3.quant_module import (
        QuantKimiMLAAttention,
    )

    wrapper = QuantKimiMLAAttention(
        _args(["attn-linear", "attn-cache"]), _MLAAttention(),
    )
    params = wrapper.export_ptq_params()
    saved = params["q_a_proj.linear.weight"].clone()
    wrapper.q_a_proj.linear.weight.data.zero_()

    wrapper.load_ptq_params(params)

    torch.testing.assert_close(wrapper.q_a_proj.linear.weight, saved)


def test_kimi_adapter_builds_dense_and_moe_targets_without_cross_wrapping():
    from amct_pytorch.common.models.llm.kimi.kimi_k3.kimi_k3 import KimiK3
    from amct_pytorch.common.models.llm.kimi.kimi_k3.quant_module import QuantKimiMLP

    block = nn.Module()
    block.self_attn = _MLAAttention()
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
    assert isinstance(quant_block.self_attn.q_a_proj, QuantLinear)

    units = list(pipeline.iter_ptq_units(3, quant_block))
    assert [(unit.kind, unit.name) for unit in units] == [
        ("attn", "self_attn"),
        ("mlp", "mlp"),
        ("moe", "expert_0"),
        ("moe", "expert_1"),
    ]


def test_kimi_adapter_builds_attn_cache_without_attn_linear():
    from amct_pytorch.common.models.llm.kimi.kimi_k3.kimi_k3 import KimiK3
    from amct_pytorch.common.models.llm.kimi.kimi_k3.quant_module import (
        QuantKimiMLAAttention,
    )

    block = nn.Module()
    block.self_attn = _MLAAttention()
    pipeline = KimiK3.__new__(KimiK3)
    pipeline.args = _args(["attn-cache"])
    pipeline.quant_target = pipeline.args.quant_target
    pipeline.block = lambda _: block

    quant_block = pipeline.build_quant_block(2)

    assert isinstance(quant_block.self_attn, QuantKimiMLAAttention)
    assert not isinstance(quant_block.self_attn.q_a_proj, QuantLinear)
    units = list(pipeline.iter_ptq_units(2, quant_block))
    assert [(unit.kind, unit.name) for unit in units] == [("attn", "self_attn")]


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
        QuantKimiMLAAttention,
        QuantKimiMLP,
    )

    monkeypatch.setattr(
        torch, "npu", SimpleNamespace(device_count=lambda: 4), raising=False,
    )
    pipeline = KimiK3.__new__(KimiK3)
    pipeline.args = _args(["attn-linear", "moe"])
    pipeline.args.device = "npu:0"
    block = nn.Module()
    block.self_attn = QuantKimiMLAAttention(pipeline.args, _MLAAttention())
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
    monkeypatch.setattr(QuantKimiMLAAttention, "to", record_placement)

    pipeline._dispatch_block(block)

    assert placements[id(block.block_sparse_moe.shared_experts)] == "npu:0"
    for expert_idx, expert in enumerate(block.block_sparse_moe.experts):
        assert placements[id(expert)] == f"npu:{expert_idx}"
    assert placements[id(block.self_attn)] == "npu:0"
    for module in block.self_attn.modules():
        if isinstance(module, QuantKimiLinear):
            assert placements[id(module)] == "npu:0"
