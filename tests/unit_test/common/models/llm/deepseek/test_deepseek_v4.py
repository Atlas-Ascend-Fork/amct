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

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn
from safetensors.torch import save_file

from amct_pytorch.common.models.llm.common.expert_offload import (
    ExpertOffloadHook,
    estimate_module_bytes,
)
from amct_pytorch.common.models.llm.deepseek.deepseek_v4 import deepseekv4
from amct_pytorch.common.models.llm.deepseek.deepseek_v4.deepseekv4 import (
    DeepseekV4,
)


def _make_model(model_path, weight_map):
    model = DeepseekV4.__new__(DeepseekV4)
    model.model_path = str(model_path)
    model.safetensors_files = None
    model._weight_map = weight_map
    return model


def test_init_requires_trust_remote_code():
    args = SimpleNamespace(
        model_name="deepseek_v4",
        trust_remote_code=False,
    )

    with pytest.raises(
        ValueError,
        match="deepseek_v4 requires --trust_remote_code",
    ):
        DeepseekV4(args)


def test_empty_weights_model_uses_trust_remote_code_from_args(monkeypatch):
    from amct_pytorch.common.models.llm.deepseek.deepseek_v4 import deepseekv4

    model = DeepseekV4.__new__(DeepseekV4)
    model.config = SimpleNamespace()
    model.trust_remote_code = True
    monkeypatch.setattr(deepseekv4, "init_empty_weights", lambda **_: nullcontext())

    with patch.object(deepseekv4.AutoModelForCausalLM, "from_config") as loader:
        model.empty_weights_model()

    assert loader.call_args.kwargs["trust_remote_code"] is True


def test_block_sharded_rejects_shard_outside_model_directory(tmp_path, monkeypatch):
    from amct_pytorch.common.models.llm.deepseek.deepseek_v4 import deepseekv4

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    weight_name = "model.layers.0.weight"
    save_file({"allowed": torch.ones(1)}, str(model_dir / "allowed.safetensors"))
    save_file({weight_name: torch.ones(1, 1)}, str(tmp_path / "outside.safetensors"))
    model = _make_model(model_dir, {weight_name: "../outside.safetensors"})
    model.cls = lambda config, layer_idx: nn.Linear(1, 1, bias=False)
    model.config = SimpleNamespace()
    model.args = SimpleNamespace(device="cpu")
    model._build_block_device_map = lambda block: {"": "cpu"}
    model.get_layer_weight_prefix = lambda layer_idx: "model.layers.0."
    monkeypatch.setattr(deepseekv4, "init_empty_weights", nullcontext)

    with pytest.raises(ValueError, match="not in the model directory allowlist"):
        model._block_sharded(0)


def test_top_level_hc_params_reject_shard_outside_model_directory(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    save_file({"allowed": torch.ones(1)}, str(model_dir / "allowed.safetensors"))
    save_file({"hc_head_fn": torch.ones(1)}, str(tmp_path / "outside.safetensors"))
    model = _make_model(model_dir, {"hc_head_fn": "../outside.safetensors"})
    model.model = nn.Module()

    with pytest.raises(ValueError, match="not in the model directory allowlist"):
        model._load_top_level_hc_head_params()


@pytest.mark.parametrize(
    "target,attr", [("moe", "ffn"), ("attn-linear", "attn"), ("attn-cache", "attn")]
)
def test_iter_ptq_units_for_load_preserves_v4_module_paths(target, attr, tmp_path):
    from amct_pytorch.common.models.llm.common.base import BaseModel

    model = DeepseekV4.__new__(DeepseekV4)
    args = SimpleNamespace(
        model="/fake/model", quant_target=[target], quant_dtype="int"
    )
    with (
        patch("amct_pytorch.common.models.llm.common.base.AutoConfig.from_pretrained"),
        patch(
            "amct_pytorch.common.models.llm.common.base.AutoTokenizer.from_pretrained"
        ),
    ):
        BaseModel.__init__(model, args)
    module = nn.Linear(2, 2, bias=False)
    block = SimpleNamespace(
        **{attr: SimpleNamespace(experts=[module]) if target == "moe" else module}
    )
    train_unit = list(model.iter_ptq_units(0, block))[0]
    loaded_unit = list(model.iter_ptq_units(0, block, for_load=True))[0]
    assert train_unit.module is loaded_unit.module is module
    assert train_unit.save_name == loaded_unit.save_name
    expected = torch.full_like(module.weight, 0.75)
    torch.save({"weight": expected}, tmp_path / f"layer_0_{train_unit.save_name}.pt")
    result = model.load_layer_ptq_params(0, block, str(tmp_path), strict=True)
    assert result == {"loaded": [train_unit.name], "missing": []}
    torch.testing.assert_close(module.weight, expected)


# ---------------------------------------------------------------------------
# single-accelerator expert CPU offload (Issue #248)
# ---------------------------------------------------------------------------
GIB = 2**30


class _FakeV4Block(nn.Module):
    """Minimal stand-in for v4 ``Block``: attn (+optional indexer) and MoE ffn."""

    def __init__(self, num_experts=4, indexer=False, hidden=8):
        super().__init__()
        self.attn = nn.Module()
        self.attn.q = nn.Linear(hidden, hidden, bias=False)
        if indexer:
            self.attn.indexer = nn.Linear(hidden, hidden, bias=False)
        self.ffn = nn.Module()
        self.ffn.gate = nn.Linear(hidden, num_experts, bias=False)
        self.ffn.experts = nn.ModuleList(
            [nn.Linear(hidden, hidden, bias=False) for _ in range(num_experts)]
        )
        self.ffn.shared_experts = nn.Linear(hidden, hidden, bias=False)
        self.attn_norm = nn.LayerNorm(hidden)
        self.ffn_norm = nn.LayerNorm(hidden)


def _make_v4_model(device="cpu", mode="auto", quant_block=False):
    model = DeepseekV4.__new__(DeepseekV4)
    model.args = SimpleNamespace(device=device)
    model.expert_offload = mode
    model.pending_quant_block = quant_block
    model._pending_block_device_map = None
    return model


def _fake_accelerator(monkeypatch, count, total_bytes):
    from amct_pytorch.common.models.llm.common import expert_offload as eo

    monkeypatch.setattr(eo, "get_accelerator_device_count", lambda device: count)
    monkeypatch.setattr(eo, "get_accelerator_total_memory", lambda device: total_bytes)
    monkeypatch.setattr(deepseekv4.torch.npu, "device_count", lambda: count)


def test_iter_offloadable_expert_modules_uses_v4_ffn_paths():
    model = _make_v4_model()
    block = _FakeV4Block(num_experts=3)
    block.ffn.experts[1] = None  # off-rank expert under TP

    found = dict(model.iter_offloadable_expert_modules(block))

    assert sorted(found) == ["ffn.experts.0", "ffn.experts.2"]
    assert found["ffn.experts.0"] is block.ffn.experts[0]


def test_device_map_single_npu_parks_experts_on_cpu_when_offload_needed(monkeypatch):
    _fake_accelerator(monkeypatch, count=1, total_bytes=64 * GIB)
    model = _make_v4_model(device="npu:0", mode="on")
    block = _FakeV4Block(num_experts=3)

    device_map = model._build_block_device_map(block)

    assert device_map == {
        "": "npu:0",
        "ffn.experts.0": "cpu",
        "ffn.experts.1": "cpu",
        "ffn.experts.2": "cpu",
    }


def test_device_map_single_npu_auto_offloads_only_when_layer_does_not_fit(
    monkeypatch,
):
    """Issue #248: a quant eval block does not fit on one accelerator because
    QuantLinear's ``cached_eval_weight`` doubles every weight, while the bf16
    baseline and extract_ptq_data use the same layer at 1x and do fit."""
    block = _FakeV4Block(num_experts=3)
    block_bytes = estimate_module_bytes(block)
    # 2x the layer size: bf16 (1x) stays under the 0.75 budget, quant (2x) does not.
    _fake_accelerator(monkeypatch, count=1, total_bytes=2 * block_bytes)

    bf16 = _make_v4_model(device="npu:0", mode="auto", quant_block=False)
    quant = _make_v4_model(device="npu:0", mode="auto", quant_block=True)

    assert bf16._build_block_device_map(block) is None
    assert quant._build_block_device_map(block) == {
        "": "npu:0",
        "ffn.experts.0": "cpu",
        "ffn.experts.1": "cpu",
        "ffn.experts.2": "cpu",
    }


def test_device_map_single_npu_auto_keeps_small_layer_resident(monkeypatch):
    _fake_accelerator(monkeypatch, count=1, total_bytes=64 * GIB)
    model = _make_v4_model(device="npu:0", mode="auto", quant_block=True)

    assert model._build_block_device_map(_FakeV4Block(num_experts=3)) is None


def test_device_map_single_npu_off_disabled_keeps_old_behavior(monkeypatch):
    _fake_accelerator(monkeypatch, count=1, total_bytes=1)
    model = _make_v4_model(device="npu:0", mode="off")

    assert model._build_block_device_map(_FakeV4Block()) is None


def test_device_map_multi_npu_shards_experts_across_cards(monkeypatch):
    _fake_accelerator(monkeypatch, count=4, total_bytes=64 * GIB)
    model = _make_v4_model(device="npu:0", mode="on")
    block = _FakeV4Block(num_experts=4, indexer=True)

    device_map = model._build_block_device_map(block)

    assert device_map[""] == "npu:0"
    assert device_map["attn"] == "npu:1"
    assert device_map["attn.indexer"] == "npu:2"
    # experts round-robin over the cards not reserved for attn / indexer
    assert [device_map[f"ffn.experts.{i}"] for i in range(4)] == [
        "npu:0",
        "npu:3",
        "npu:0",
        "npu:3",
    ]
    assert "cpu" not in device_map.values()


def test_dispatch_block_attaches_offload_hook_for_cpu_entries(monkeypatch):
    _fake_accelerator(monkeypatch, count=1, total_bytes=64 * GIB)
    model = _make_v4_model(device="cpu", mode="on")
    block = _FakeV4Block(num_experts=2)
    model._pending_block_device_map = {
        "": "cpu",
        "ffn.experts.0": "cpu",
        "ffn.experts.1": "cpu",
    }

    dispatched = model._dispatch_block(block)

    for expert in dispatched.ffn.experts:
        assert isinstance(expert._hf_hook, ExpertOffloadHook)
        assert expert._hf_hook.offload is True
        assert expert._hf_hook.execution_device == "cpu"
        assert expert.weight.is_meta
    # stem keeps the non-moving hook and stays io_same_device
    assert not isinstance(dispatched._hf_hook, ExpertOffloadHook)
    assert dispatched._hf_hook.io_same_device is True
    assert not dispatched.ffn.shared_experts.weight.is_meta


def test_dispatch_block_consumes_pending_device_map(monkeypatch):
    _fake_accelerator(monkeypatch, count=4, total_bytes=64 * GIB)
    model = _make_v4_model(device="cpu", mode="on")
    block = _FakeV4Block(num_experts=2)
    model._pending_block_device_map = {"": "cpu", "ffn.experts.0": "cpu"}

    model._dispatch_block(block)

    assert model._pending_block_device_map is None


def test_dispatch_block_recomputes_map_when_not_preloaded(monkeypatch):
    _fake_accelerator(monkeypatch, count=1, total_bytes=64 * GIB)
    model = _make_v4_model(device="cpu", mode="on")
    block = _FakeV4Block(num_experts=2)

    dispatched = model._dispatch_block(block)

    assert isinstance(dispatched.ffn.experts[0]._hf_hook, ExpertOffloadHook)


def test_offload_hook_streams_to_the_stem_device(monkeypatch):
    """Experts parked on the host must still execute on the accelerator."""
    _fake_accelerator(monkeypatch, count=1, total_bytes=64 * GIB)
    model = _make_v4_model(device="cpu", mode="on")
    block = _FakeV4Block(num_experts=2)
    model._pending_block_device_map = {
        "": "npu:0",
        "ffn.experts.0": "cpu",
        "ffn.experts.1": "cpu",
    }

    model._dispatch_block(block)

    for expert in block.ffn.experts:
        assert isinstance(expert._hf_hook, ExpertOffloadHook)
        assert expert._hf_hook.execution_device == "npu:0"


def test_retag_descendant_hooks_skips_host_entries():
    """Re-tagging a ``cpu`` entry would point the streaming hook at host memory."""
    from accelerate.hooks import add_hook_to_module

    block = _FakeV4Block(num_experts=1)
    add_hook_to_module(
        block.ffn.experts[0],
        ExpertOffloadHook(
            execution_device="npu:0", offload=True, place_submodules=True
        ),
    )

    DeepseekV4._retag_descendant_hooks(
        block, {"": "npu:0", "attn": "npu:0", "ffn.experts.0": "cpu"}
    )

    hook = block.ffn.experts[0]._hf_hook
    assert isinstance(hook, ExpertOffloadHook)
    assert hook.execution_device == "npu:0"
    assert hook.offload is True
