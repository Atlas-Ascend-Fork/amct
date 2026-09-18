# -*- coding: UTF-8 -*-

from types import SimpleNamespace

import pytest
import torch

from amct_pytorch.common.models.llm.common.packed_expert_export import (
    PackedGatedExpertLayout,
    detect_packed_gated_expert_layout,
)


def _config(**kwargs):
    return SimpleNamespace(num_experts=2, moe_intermediate_size=3, **kwargs)


def test_detect_layout_and_expand_packed_index_to_vllm_names():
    weight_map = {
        "model.layers.0.mlp.experts.gate_up_proj": "a.safetensors",
        "model.layers.0.mlp.experts.down_proj": "a.safetensors",
        "model.layers.0.input_layernorm.weight": "a.safetensors",
    }

    layout = detect_packed_gated_expert_layout(weight_map, _config())

    assert isinstance(layout, PackedGatedExpertLayout)
    expanded = layout.expand_weight_map(weight_map)
    assert expanded["model.layers.0.mlp.experts.0.gate_proj.weight"] == "a.safetensors"
    assert expanded["model.layers.0.mlp.experts.1.up_proj.weight"] == "a.safetensors"
    assert expanded["model.layers.0.mlp.experts.1.down_proj.weight"] == "a.safetensors"
    assert expanded["model.layers.0.input_layernorm.weight"] == "a.safetensors"


def test_expand_packed_tensors_splits_gate_up_and_materializes_unselected_slices():
    gate_up = torch.arange(2 * 6 * 4, dtype=torch.bfloat16).reshape(2, 6, 4)
    down = torch.arange(2 * 4 * 3, dtype=torch.bfloat16).reshape(2, 4, 3)
    state_dict = {
        "model.layers.0.mlp.experts.gate_up_proj": gate_up,
        "model.layers.0.mlp.experts.down_proj": down,
    }
    layout = PackedGatedExpertLayout(num_experts=2, moe_intermediate_size=3)
    selected = {"model.layers.0.mlp.experts.0.gate_proj.weight"}

    expanded = layout.expand_tensors(state_dict, selected_weight_keys=selected)

    assert torch.equal(
        expanded["model.layers.0.mlp.experts.1.gate_proj.weight"], gate_up[1, :3]
    )
    assert torch.equal(
        expanded["model.layers.0.mlp.experts.0.up_proj.weight"], gate_up[0, 3:]
    )
    assert torch.equal(
        expanded["model.layers.0.mlp.experts.1.down_proj.weight"], down[1]
    )
    assert (
        expanded["model.layers.0.mlp.experts.1.gate_proj.weight"].untyped_storage().data_ptr()
        != gate_up.untyped_storage().data_ptr()
    )


def test_detect_layout_supports_routed_expert_config_alias():
    weight_map = {
        "model.layers.0.mlp.experts.gate_up_proj": "a.safetensors",
        "model.layers.0.mlp.experts.down_proj": "a.safetensors",
    }

    layout = detect_packed_gated_expert_layout(
        weight_map, SimpleNamespace(n_routed_experts=2, moe_intermediate_size=3)
    )

    assert layout.num_experts == 2


def test_expand_packed_tensors_rejects_invalid_gate_up_shape():
    layout = PackedGatedExpertLayout(num_experts=2, moe_intermediate_size=3)
    state_dict = {
        "model.layers.0.mlp.experts.gate_up_proj": torch.zeros(2, 5, 4),
    }

    with pytest.raises(ValueError, match="even gate/up dimension"):
        layout.expand_tensors(state_dict)


def test_detect_layout_returns_none_for_unpacked_checkpoint():
    weight_map = {"model.layers.0.mlp.experts.0.gate_proj.weight": "a.safetensors"}

    assert detect_packed_gated_expert_layout(weight_map, _config()) is None
