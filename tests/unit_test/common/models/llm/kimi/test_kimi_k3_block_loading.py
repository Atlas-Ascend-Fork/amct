# -*- coding: UTF-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Streaming block loading with real small tensors and safetensors shards."""

import json
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from amct_pytorch.common.models.llm.kimi.kimi_k3.kimi_k3 import KimiK3


class TinyBlock(torch.nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.self_attn = torch.nn.Module()
        self.self_attn.A_log = torch.nn.Parameter(torch.ones(2))
        if layer_idx == 0:
            self.mlp = torch.nn.Linear(4, 4, bias=False)
        else:
            self.block_sparse_moe = torch.nn.Module()
            self.block_sparse_moe.experts = torch.nn.ModuleList()
            for _ in range(2):
                expert = torch.nn.Module()
                for name in ("w1", "w2", "w3"):
                    setattr(expert, name, torch.nn.Linear(4, 4, bias=False))
                self.block_sparse_moe.experts.append(expert)
        self.register_buffer("counter", torch.tensor([7], dtype=torch.int64))
        self.register_buffer("derived", torch.arange(4).float(), persistent=False)

    def forward(self, x):
        if hasattr(self, "mlp"):
            return self.mlp(x) + self.derived
        out = torch.zeros_like(x)
        for expert in self.block_sparse_moe.experts:
            out = out + expert.w2(torch.nn.functional.silu(expert.w1(x)) * expert.w3(x))
        return out + self.derived


def make_checkpoint(tmp_path, layer_idx=1, mutation=None, model_class=TinyBlock, config=None):
    if config is None:
        config = SimpleNamespace(linear_attn_config={"num_heads": 2, "head_dim": 4})
    model = model_class(config, layer_idx)
    state = {
        name: torch.full_like(value, 0.125) if value.is_floating_point() else value.clone()
        for name, value in model.state_dict().items()
    }
    if "self_attn.A_log" in state:
        state["self_attn.A_log"] = torch.tensor([0.25, 0.5, 0.0, 0.0])
    if mutation is not None:
        mutation(state)
    weight_map = {}
    # Interleaved shard assignment checks loading by index rather than layer filename.
    for shard in range(2):
        filename = f"weights-{shard}.safetensors"
        tensors = {
            f"language_model.model.layers.{layer_idx}.{name}": value
            for i, (name, value) in enumerate(state.items()) if i % 2 == shard
        }
        save_file(tensors, str(tmp_path / filename))
        weight_map.update({name: filename for name in tensors})
    # An unrelated layer must not be loaded or validated.
    weight_map["language_model.model.layers.10.unused.weight"] = "not-opened.safetensors"
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map}), encoding="utf-8",
    )
    pipeline = KimiK3.__new__(KimiK3)
    pipeline.args = SimpleNamespace(device="cpu")
    pipeline.config = config
    pipeline.cls = model_class
    pipeline.model_path = str(tmp_path)
    pipeline._weight_map = None
    pipeline.sharded_block = True
    return pipeline


@pytest.mark.parametrize("layer_idx", [0, 1])
def test_direct_load_matches_cpu_staging_without_cpu_linear_initialization(tmp_path, monkeypatch, layer_idx):
    pipeline = make_checkpoint(tmp_path, layer_idx)
    pipeline.sharded_block = False
    reference = pipeline.block(layer_idx)
    pipeline.sharded_block = True

    original_reset = torch.nn.Linear.reset_parameters

    def meta_reset(module):
        assert module.weight.is_meta, "direct loader initialized a real CPU Linear weight"
        original_reset(module)

    monkeypatch.setattr(torch.nn.Linear, "reset_parameters", meta_reset)
    result = pipeline.block(layer_idx)
    assert not result.training
    for name, value in result.state_dict().items():
        torch.testing.assert_close(value, reference.state_dict()[name], rtol=0, atol=0)
    for value in list(result.parameters()) + list(result.buffers()):
        assert value.device == torch.device("cpu")
        assert not value.is_meta
        if value.is_floating_point():
            assert value.dtype == torch.bfloat16
    assert result.counter.dtype == torch.int64
    torch.testing.assert_close(result.self_attn.A_log, torch.tensor([0.25, 0.5], dtype=torch.bfloat16))
    inputs = torch.ones(1, 4, dtype=torch.bfloat16)
    torch.testing.assert_close(result(inputs), reference(inputs), rtol=0, atol=0)


@pytest.mark.parametrize("issue", ["missing", "unexpected", "shape", "padding", "packed", "integer"])
def test_direct_load_rejects_invalid_checkpoint(tmp_path, issue):
    key = "block_sparse_moe.experts.0.w1.weight"

    def mutate(state):
        if issue == "missing":
            del state[key]
        elif issue == "unexpected":
            state["unknown.weight"] = torch.ones(1)
        elif issue == "shape":
            state[key] = torch.ones(3, 4)
        elif issue == "padding":
            state["self_attn.A_log"][-1] = 1
        elif issue == "packed":
            state[key.replace(".weight", ".weight_packed")] = state.pop(key).to(torch.uint8)
        else:
            state[key] = state[key].to(torch.int8)

    pipeline = make_checkpoint(tmp_path, mutation=mutate)
    with pytest.raises((RuntimeError, ValueError), match={
        "missing": "(?i)missing", "unexpected": "(?i)unexpected", "shape": "shape",
        "padding": "non-zero padding", "packed": "weight_packed", "integer": "floating",
    }[issue]):
        pipeline.block(1)


def test_direct_load_routes_every_tensor_to_args_device(tmp_path, monkeypatch):
    from amct_pytorch.common.models.llm.kimi.kimi_k3 import kimi_k3 as adapter

    pipeline = make_checkpoint(tmp_path)
    pipeline.args.device = "npu:3"
    real_set_tensor = adapter.set_module_tensor_to_device
    transfers = []

    def cpu_transport(module, name, device, *, value, dtype):
        # Replace only the unavailable hardware transfer, retaining real loading.
        assert device == "npu:3"
        transfers.append(name)
        real_set_tensor(module, name, "cpu", value=value, dtype=dtype)

    monkeypatch.setattr(adapter, "set_module_tensor_to_device", cpu_transport)
    monkeypatch.setattr(adapter, "_perf_sync", lambda device: None)
    result = pipeline.block(1)
    expected_names = set(dict(result.named_parameters())) | set(dict(result.named_buffers()))
    assert set(transfers) == expected_names


@pytest.mark.parametrize("layer_idx", [0, 1, 2])
def test_real_kimi_dense_moe_and_mla_weights_match_cpu_loading(tmp_path, monkeypatch, layer_idx):
    # Only external NPU kernels are unavailable here; construct the real K3 modules.
    def unavailable_kernel(*args, **kwargs):
        raise AssertionError("Loading tests must not invoke an external NPU kernel")

    for package in ("ops", "ops.cannbot_dsl"):
        module = ModuleType(package)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, package, module)
    for path, symbol in (
        ("ops.cannbot_dsl.flash_kda", "flash_kda"),
        ("ops.cannbot_dsl.fused_recurrent_kda", "fused_recurrent_kda_op"),
    ):
        module = ModuleType(path)
        setattr(module, symbol, unavailable_kernel)
        monkeypatch.setitem(sys.modules, path, module)

    from amct_pytorch.common.models.llm.kimi.kimi_k3.modeling.configuration_kimi_k3 import (
        KimiLinearConfig,
    )
    from amct_pytorch.common.models.llm.kimi.kimi_k3.modeling.modeling_kimi import (
        KimiDecoderLayer,
    )

    config = KimiLinearConfig(
        hidden_size=8, intermediate_size=16, num_hidden_layers=3, num_attention_heads=2,
        num_experts=2, num_experts_per_token=1, num_shared_experts=1,
        moe_intermediate_size=4, first_k_dense_replace=1,
        routed_expert_hidden_size=4, latent_moe_use_norm=True, attn_res_block_size=2,
        q_lora_rank=4, kv_lora_rank=4, qk_nope_head_dim=4, qk_rope_head_dim=0,
        v_head_dim=4, mla_use_nope=True,
        linear_attn_config={
            "num_heads": 2, "head_dim": 4, "short_conv_kernel_size": 3,
            "kda_layers": [1, 2], "full_attn_layers": [3],
        },
    )
    pipeline = make_checkpoint(tmp_path, layer_idx, model_class=KimiDecoderLayer, config=config)
    pipeline.sharded_block = False
    reference = pipeline.block(layer_idx)
    pipeline.sharded_block = True

    def no_cpu_init(tensor, *args, **kwargs):
        assert tensor.is_meta, "K3 initialized an expert/projection on CPU"
        return tensor

    monkeypatch.setattr(torch.nn.init, "kaiming_uniform_", no_cpu_init)
    result = pipeline.block(layer_idx)
    assert set(result.state_dict()) == set(reference.state_dict())
    for name, value in result.state_dict().items():
        torch.testing.assert_close(value, reference.state_dict()[name], rtol=0, atol=0)
        assert not value.is_meta
    inputs = torch.ones(1, 2, 8, dtype=torch.bfloat16)
    mlp_name = "mlp" if layer_idx == 0 else "block_sparse_moe"
    torch.testing.assert_close(
        getattr(result, mlp_name)(inputs), getattr(reference, mlp_name)(inputs), rtol=0, atol=0,
    )
