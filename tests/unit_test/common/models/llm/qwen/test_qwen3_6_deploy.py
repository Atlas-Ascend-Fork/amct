# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from types import SimpleNamespace

import pytest
import torch

from amct_pytorch.common.models.llm.common.deploy_selection import (
    DeploySelection,
    build_ascend_deploy_plan,
)
from amct_pytorch.common.models.llm.qwen.qwen3_6.qwen3_6_moe import Qwen3_6Moe
from amct_pytorch.quantization.bit_policy import BitPolicy


def _adapter():
    adapter = object.__new__(Qwen3_6Moe)
    adapter.args = SimpleNamespace(bit_policy=BitPolicy({"w_bits": 8, "a_bits": 8}))
    adapter.config = SimpleNamespace(
        num_hidden_layers=4,
        num_experts=2,
        layer_types=[
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "full_attention",
        ],
        mtp_num_hidden_layers=1,
    )
    return adapter


def test_virtual_inventory_expands_packed_experts_for_json_selection():
    adapter = _adapter()
    weight_map = {
        "model.language_model.layers.0.mlp.experts.gate_up_proj": "a.safetensors",
        "model.language_model.layers.0.mlp.experts.down_proj": "a.safetensors",
        "model.language_model.layers.3.self_attn.q_proj.weight": "b.safetensors",
        "model.language_model.layers.3.self_attn.k_proj.weight": "b.safetensors",
        "model.language_model.layers.3.self_attn.v_proj.weight": "b.safetensors",
        "model.language_model.layers.3.self_attn.o_proj.weight": "b.safetensors",
    }
    selection = DeploySelection(
        (
            "model.language_model.layers.*.mlp.experts.*.gate_proj",
            "model.language_model.layers.*.mlp.experts.*.up_proj",
            "model.language_model.layers.*.mlp.experts.*.down_proj",
            "model.language_model.layers.*.self_attn.q_proj",
            "model.language_model.layers.*.self_attn.k_proj",
            "model.language_model.layers.*.self_attn.v_proj",
        ),
        ("model.language_model.layers.*.self_attn.o_proj",),
    )

    plan = build_ascend_deploy_plan(weight_map, selection, adapter)

    assert len(plan.selected_modules) == 9
    assert (
        "model.language_model.layers.0.mlp.experts.1.up_proj" in plan.selected_modules
    )
    assert "model.language_model.layers.3.self_attn.q_proj" in plan.selected_modules
    assert not any(name.endswith("o_proj") for name in plan.selected_modules)


def test_expand_packed_expert_tensors_uses_vllm_checkpoint_names():
    adapter = _adapter()
    gate_up = torch.arange(2 * 6 * 4, dtype=torch.bfloat16).reshape(2, 6, 4)
    down = torch.arange(2 * 4 * 3, dtype=torch.bfloat16).reshape(2, 4, 3)
    state_dict = {
        "model.language_model.layers.0.mlp.experts.gate_up_proj": gate_up,
        "model.language_model.layers.0.mlp.experts.down_proj": down,
        "model.language_model.layers.0.mlp.gate.weight": torch.ones(2, 4),
    }

    expanded = adapter.expand_ascend_deploy_tensors(state_dict)

    assert "model.language_model.layers.0.mlp.experts.gate_up_proj" not in expanded
    assert torch.equal(
        expanded["model.language_model.layers.0.mlp.experts.1.gate_proj.weight"],
        gate_up[1, :3],
    )
    assert torch.equal(
        expanded["model.language_model.layers.0.mlp.experts.1.up_proj.weight"],
        gate_up[1, 3:],
    )
    assert torch.equal(
        expanded["model.language_model.layers.0.mlp.experts.1.down_proj.weight"],
        down[1],
    )
    assert "model.language_model.layers.0.mlp.gate.weight" in expanded


def test_expand_mtp_packed_experts_for_float_output():
    adapter = _adapter()
    state_dict = {
        "mtp.layers.0.mlp.experts.gate_up_proj": torch.ones(2, 6, 4),
        "mtp.layers.0.mlp.experts.down_proj": torch.ones(2, 4, 3),
    }

    expanded = adapter.expand_ascend_deploy_tensors(state_dict)

    assert "mtp.layers.0.mlp.experts.gate_up_proj" not in expanded
    assert "mtp.layers.0.mlp.experts.0.gate_proj.weight" in expanded
    assert "mtp.layers.0.mlp.experts.1.down_proj.weight" in expanded
    storage_pointers = {
        tensor.untyped_storage().data_ptr() for tensor in expanded.values()
    }
    assert len(storage_pointers) == len(expanded)


def test_rejects_attention_quantization_on_linear_attention_layer():
    adapter = _adapter()
    selected = {"model.language_model.layers.0.self_attn.q_proj"}

    try:
        adapter.validate_ascend_deploy_selection(selected)
    except ValueError as exc:
        assert "Unsupported" in str(exc)
    else:
        raise AssertionError("linear-attention QKV must not be accepted")


def test_rejects_partial_routed_expert_selection():
    adapter = _adapter()
    selected = {"model.language_model.layers.0.mlp.experts.0.gate_proj"}

    with torch.no_grad(), pytest.raises(ValueError, match="all experts"):
        adapter.validate_ascend_deploy_selection(selected)
