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

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------

import sys
from importlib.machinery import ModuleSpec
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from amct_pytorch.common.models.llm.common import expert_offload as eo
from amct_pytorch.common.models.llm.common.base import BaseModel
from amct_pytorch.quantization.modules.quant_linear import QuantLinear

GIB = 2**30


@pytest.fixture(autouse=True)
def _usable_torch_npu_spec():
    """Keep ``importlib.util.find_spec("torch_npu")`` working.

    Other suites (e.g. ``tests/unit_test/classic/test_deploy_kv_quant.py``)
    register a bare ``types.ModuleType`` as ``torch_npu`` at import time, whose
    ``__spec__`` is ``None``.  ``accelerate`` calls ``find_spec("torch_npu")``
    from ``clear_device_cache()``, which raises ``ValueError`` in that case.  Give
    such a stub a minimal spec for the duration of these tests and restore the
    previous state afterwards.
    """
    module = sys.modules.get("torch_npu")
    needs_stub = module is not None and getattr(module, "__spec__", None) is None
    if needs_stub:
        module.__spec__ = ModuleSpec("torch_npu", loader=None)
    yield
    if needs_stub:
        module.__spec__ = None


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _make_model(device="cpu", mode="auto"):
    model = BaseModel.__new__(BaseModel)
    model.args = SimpleNamespace(device=device)
    model.expert_offload = mode
    model.pending_quant_block = False
    return model


def _fake_quant_linear(weight):
    """QuantLinear-ish stub: same cached attributes, no quantizer machinery."""
    linear = nn.Linear(weight.shape[1], weight.shape[0], bias=False)
    linear.weight = nn.Parameter(weight)
    stub = nn.Module()
    stub.linear = linear
    stub.eval_mode = True
    stub.cached_eval_weight = None
    stub._cached_transform_key = None
    return stub


class _FakeBlock(nn.Module):
    """Decoder block with a routed ``mlp.experts`` ModuleList."""

    def __init__(self, num_experts=3, packed=False):
        super().__init__()
        self.input_layernorm = nn.LayerNorm(4)
        self.mlp = nn.Module()
        self.mlp.shared_expert = nn.Linear(4, 4, bias=False)
        if packed:
            experts = nn.Module()
            experts.packed_experts = nn.Linear(4, 4, bias=False)
            experts.expert_modules = nn.ModuleList(
                [nn.Linear(4, 4, bias=False) for _ in range(num_experts)]
            )
        else:
            experts = nn.ModuleList(
                [nn.Linear(4, 4, bias=False) for _ in range(num_experts)]
            )
        self.mlp.experts = experts


def _fake_accelerator(monkeypatch, count, total_bytes):
    monkeypatch.setattr(eo, "get_accelerator_device_count", lambda device: count)
    monkeypatch.setattr(eo, "get_accelerator_total_memory", lambda device: total_bytes)


# ---------------------------------------------------------------------------
# clear_quant_weight_cache
# ---------------------------------------------------------------------------
def test_clear_quant_weight_cache_drops_cached_eval_weight():
    stub = _fake_quant_linear(torch.ones(2, 3))
    stub.cached_eval_weight = torch.zeros(2, 3)
    stub._cached_transform_key = 12345

    assert eo.clear_quant_weight_cache(stub) == 1
    assert stub.cached_eval_weight is None
    assert stub._cached_transform_key is None


def test_clear_quant_weight_cache_recurses_and_counts_only_cached():
    root = nn.Module()
    filled = _fake_quant_linear(torch.ones(2, 2))
    filled.cached_eval_weight = torch.zeros(2, 2)
    empty = _fake_quant_linear(torch.ones(2, 2))
    root.add_module("filled", filled)
    root.add_module("empty", empty)
    root.add_module("plain", nn.Linear(2, 2))

    assert eo.clear_quant_weight_cache(root) == 1


def test_clear_quant_weight_cache_is_noop_on_plain_module():
    assert eo.clear_quant_weight_cache(nn.Linear(2, 2)) == 0


# ---------------------------------------------------------------------------
# estimate_module_bytes
# ---------------------------------------------------------------------------
def test_estimate_module_bytes_counts_params_and_buffers():
    module = nn.Linear(4, 4)  # weight 4x4 fp32 + bias 4 fp32
    assert eo.estimate_module_bytes(module) == (16 + 4) * 4


def test_estimate_module_bytes_works_on_meta_tensors():
    with torch.device("meta"):
        module = nn.Linear(8, 8, bias=False)
    assert module.weight.is_meta
    assert eo.estimate_module_bytes(module) == 64 * 4


# ---------------------------------------------------------------------------
# should_offload_experts policy
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("mode", ["off", "OFF", " off "])
def test_policy_off_never_offloads(monkeypatch, mode):
    _fake_accelerator(monkeypatch, count=1, total_bytes=1)
    assert (
        eo.should_offload_experts(
            mode, quantized=True, block_bytes=10 * GIB, device="npu:0"
        )
        is False
    )


@pytest.mark.parametrize("mode", ["on", "ON"])
def test_policy_on_always_offloads(monkeypatch, mode):
    _fake_accelerator(monkeypatch, count=8, total_bytes=64 * GIB)
    assert (
        eo.should_offload_experts(mode, quantized=True, block_bytes=1, device="npu:0")
        is True
    )


def test_policy_rejects_unknown_mode():
    with pytest.raises(ValueError, match="Unsupported expert_offload mode"):
        eo.should_offload_experts("maybe", quantized=True, block_bytes=1)


def test_policy_auto_defaults_to_auto_when_mode_missing(monkeypatch):
    _fake_accelerator(monkeypatch, count=1, total_bytes=64 * GIB)
    # 20 GiB bf16 block, quantized -> 40 GiB needed > 0.75 * 64 = 48? no -> False
    assert (
        eo.should_offload_experts(
            None, quantized=True, block_bytes=20 * GIB, device="npu:0"
        )
        is False
    )


def test_policy_auto_offloads_when_quantized_block_exceeds_budget(monkeypatch):
    _fake_accelerator(monkeypatch, count=1, total_bytes=64 * GIB)
    # 25.8 GiB bf16 layer doubled by cached_eval_weight -> 51.6 > 0.75 * 64 = 48
    assert (
        eo.should_offload_experts(
            "auto", quantized=True, block_bytes=26 * GIB, device="npu:0"
        )
        is True
    )


def test_policy_auto_keeps_bf16_block_resident(monkeypatch):
    _fake_accelerator(monkeypatch, count=1, total_bytes=64 * GIB)
    # Same layer without the eval-time weight twin fits: 26 < 48
    assert (
        eo.should_offload_experts(
            "auto", quantized=False, block_bytes=26 * GIB, device="npu:0"
        )
        is False
    )


def test_policy_auto_disabled_on_multi_accelerator(monkeypatch):
    _fake_accelerator(monkeypatch, count=4, total_bytes=64 * GIB)
    assert (
        eo.should_offload_experts(
            "auto", quantized=True, block_bytes=1000 * GIB, device="npu:0"
        )
        is False
    )


@pytest.mark.parametrize("count,total", [(None, 64 * GIB), (1, None), (1, 0)])
def test_policy_auto_degrades_to_resident_without_memory_info(
    monkeypatch, count, total
):
    _fake_accelerator(monkeypatch, count=count, total_bytes=total)
    assert (
        eo.should_offload_experts(
            "auto", quantized=True, block_bytes=1000 * GIB, device="npu:0"
        )
        is False
    )


def test_policy_auto_degrades_when_block_size_unknown(monkeypatch):
    _fake_accelerator(monkeypatch, count=1, total_bytes=64 * GIB)
    assert (
        eo.should_offload_experts(
            "auto", quantized=True, block_bytes=None, device="npu:0"
        )
        is False
    )


def test_get_accelerator_helpers_report_none_off_accelerator():
    assert eo.get_accelerator_device_count("cpu") is None
    assert eo.get_accelerator_total_memory("cpu") is None


# ---------------------------------------------------------------------------
# ExpertOffloadHook
# ---------------------------------------------------------------------------
def _add_offload_hook(module, execution_device="cpu"):
    from accelerate.hooks import add_hook_to_module

    add_hook_to_module(
        module,
        eo.ExpertOffloadHook(
            execution_device=execution_device, offload=True, place_submodules=True
        ),
    )


def test_expert_offload_hook_parks_weights_on_host_between_forwards():
    expert = nn.Linear(4, 4, bias=False)
    expected = expert.weight.detach().clone()
    _add_offload_hook(expert)

    assert expert.weight.is_meta  # parked
    out = expert(torch.ones(2, 4))
    assert out.shape == (2, 4)
    assert expert.weight.is_meta  # parked again after forward

    from accelerate.hooks import remove_hook_from_module

    remove_hook_from_module(expert, recurse=True)
    torch.testing.assert_close(expert.weight.detach(), expected)


def _make_quant_linear(weight):
    """QuantLinear with a counting stand-in for the weight quantizer."""
    calls = []
    module = QuantLinear.__new__(QuantLinear)
    nn.Module.__init__(module)
    linear = nn.Linear(weight.shape[1], weight.shape[0], bias=False)
    linear.weight = nn.Parameter(weight)
    module.linear = linear
    module.eval_mode = True
    module.cached_eval_weight = None
    module._cached_transform_key = None
    module.weight_quantizer = lambda w: (calls.append(w), w)[1]
    module.args = None
    module.name = "expert"
    module.w_bits = 8
    return module, calls


def test_offloaded_expert_drops_cached_eval_weight_between_forwards():
    """``cached_eval_weight`` is a plain attribute, so ``nn.Module.to()`` neither
    moves nor frees it: an offloaded expert would keep one device-sized copy alive
    per expert unless the hook drops it explicitly."""
    module, calls = _make_quant_linear(torch.ones(4, 4))
    _add_offload_hook(module)

    module(torch.ones(2, 4))
    module(torch.ones(2, 4))

    assert module.cached_eval_weight is None
    assert len(calls) == 2  # cache was released, so each forward re-quantizes


def test_resident_expert_keeps_cached_eval_weight():
    """Control for the test above: without offload the cache is reused."""
    module, calls = _make_quant_linear(torch.ones(4, 4))

    module(torch.ones(2, 4))
    module(torch.ones(2, 4))

    assert module.cached_eval_weight is not None
    assert len(calls) == 1


def test_remove_hook_from_offloaded_expert_restores_host_weights():
    module, _ = _make_quant_linear(torch.full((4, 4), 0.5))
    expected = module.linear.weight.detach().clone()
    _add_offload_hook(module)
    module(torch.ones(2, 4))

    from accelerate.hooks import remove_hook_from_module

    remove_hook_from_module(module, recurse=True)

    assert not module.linear.weight.is_meta
    torch.testing.assert_close(module.linear.weight.detach(), expected)
    assert module.cached_eval_weight is None


# ---------------------------------------------------------------------------
# BaseModel generic discovery + dispatch
# ---------------------------------------------------------------------------
def test_iter_offloadable_expert_modules_finds_routed_experts():
    model = _make_model()
    block = _FakeBlock(num_experts=3)
    found = dict(model.iter_offloadable_expert_modules(block))
    assert sorted(found) == ["mlp.experts.0", "mlp.experts.1", "mlp.experts.2"]
    assert found["mlp.experts.1"] is block.mlp.experts[1]


def test_iter_offloadable_expert_modules_skips_packed_experts():
    """Packed views share one storage: streaming them frees no memory."""
    model = _make_model()
    assert list(model.iter_offloadable_expert_modules(_FakeBlock(packed=True))) == []


def test_iter_offloadable_expert_modules_skips_single_expert():
    model = _make_model()
    assert list(model.iter_offloadable_expert_modules(_FakeBlock(num_experts=1))) == []


def test_iter_offloadable_expert_modules_skips_none_entries():
    model = _make_model()
    block = _FakeBlock(num_experts=2)
    block.mlp.experts[1] = None
    assert [p for p, _ in model.iter_offloadable_expert_modules(block)] == [
        "mlp.experts.0"
    ]


def test_resolve_expert_offload_paths_respects_mode_off(monkeypatch):
    _fake_accelerator(monkeypatch, count=1, total_bytes=1)
    model = _make_model(mode="off")
    assert model._resolve_expert_offload_paths(_FakeBlock()) == []


def test_resolve_expert_offload_paths_empty_when_policy_says_no(monkeypatch):
    _fake_accelerator(monkeypatch, count=4, total_bytes=64 * GIB)
    model = _make_model(mode="auto")
    assert model._resolve_expert_offload_paths(_FakeBlock()) == []


def test_resolve_expert_offload_paths_lists_experts_when_forced(monkeypatch):
    _fake_accelerator(monkeypatch, count=1, total_bytes=64 * GIB)
    model = _make_model(mode="on")
    paths = [p for p, _ in model._resolve_expert_offload_paths(_FakeBlock())]
    assert paths == ["mlp.experts.0", "mlp.experts.1", "mlp.experts.2"]


def test_block_is_quantized_uses_pending_flag_before_wrappers_exist():
    model = _make_model()
    block = _FakeBlock()
    assert model._block_is_quantized(block) is False
    model.pending_quant_block = True
    assert model._block_is_quantized(block) is True


def test_dispatch_block_keeps_offloaded_experts_on_host(monkeypatch):
    _fake_accelerator(monkeypatch, count=1, total_bytes=1)
    model = _make_model(mode="on")
    block = _FakeBlock(num_experts=2)

    dispatched = model._dispatch_block(block)

    for expert in dispatched.mlp.experts:
        assert expert.weight.is_meta  # snapshotted into the hook's weights_map
        assert isinstance(expert._hf_hook, eo.ExpertOffloadHook)
    assert not dispatched.mlp.shared_expert.weight.is_meta
    assert not dispatched.input_layernorm.weight.is_meta


def test_dispatch_block_without_offload_moves_everything(monkeypatch):
    _fake_accelerator(monkeypatch, count=4, total_bytes=64 * GIB)
    model = _make_model(mode="auto")
    block = _FakeBlock(num_experts=2)

    dispatched = model._dispatch_block(block)

    for expert in dispatched.mlp.experts:
        assert not expert.weight.is_meta
        assert getattr(expert, "_hf_hook", None) is None
