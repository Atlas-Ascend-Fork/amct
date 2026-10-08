# Copyright (c) 2026 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from types import SimpleNamespace

import pytest
import torch

from amct_pytorch.common.models.llm.common.packed_expert_export import (
    PackedGatedExpertLayout,
    detect_packed_gated_expert_layout,
)

PREFIX = "model.layers.0.mlp.experts"
GATE_UP = f"{PREFIX}.gate_up_proj"
DOWN = f"{PREFIX}.down_proj"


def test_detect_returns_none_for_unpacked_checkpoint():
    assert (
        detect_packed_gated_expert_layout(
            {f"{PREFIX}.0.gate_proj.weight": "model.safetensors"},
            SimpleNamespace(num_experts=2),
        )
        is None
    )


def test_detect_reads_nested_text_config_and_builds_layout():
    config = SimpleNamespace(
        text_config=SimpleNamespace(n_routed_experts=2, intermediate_size=3)
    )

    layout = detect_packed_gated_expert_layout(
        {GATE_UP: "model.safetensors", DOWN: "model.safetensors"}, config
    )

    assert layout == PackedGatedExpertLayout(num_experts=2, moe_intermediate_size=3)


@pytest.mark.parametrize(
    "weight_map,config,message",
    [
        ({GATE_UP: "model.safetensors"}, SimpleNamespace(num_experts=2), "both"),
        (
            {GATE_UP: "model.safetensors", DOWN: "model.safetensors"},
            SimpleNamespace(),
            "positive num_experts",
        ),
        (
            {GATE_UP: "model.safetensors", DOWN: "model.safetensors"},
            SimpleNamespace(num_local_experts=0),
            "positive num_experts",
        ),
    ],
)
def test_detect_rejects_incomplete_or_invalid_layout(weight_map, config, message):
    with pytest.raises(ValueError, match=message):
        detect_packed_gated_expert_layout(weight_map, config)


def test_expand_weight_map_emits_per_expert_names_and_preserves_other_entries():
    layout = PackedGatedExpertLayout(num_experts=2, moe_intermediate_size=3)

    expanded = layout.expand_weight_map(
        {
            GATE_UP: "experts.safetensors",
            DOWN: "experts.safetensors",
            "model.embed_tokens.weight": "model.safetensors",
        }
    )

    assert expanded == {
        f"{PREFIX}.0.gate_proj.weight": "experts.safetensors",
        f"{PREFIX}.0.up_proj.weight": "experts.safetensors",
        f"{PREFIX}.0.down_proj.weight": "experts.safetensors",
        f"{PREFIX}.1.gate_proj.weight": "experts.safetensors",
        f"{PREFIX}.1.up_proj.weight": "experts.safetensors",
        f"{PREFIX}.1.down_proj.weight": "experts.safetensors",
        "model.embed_tokens.weight": "model.safetensors",
    }


def test_expand_tensors_splits_values_and_clones_unselected_slices():
    layout = PackedGatedExpertLayout(num_experts=2, moe_intermediate_size=2)
    gate_up = torch.arange(16, dtype=torch.float32).reshape(2, 4, 2)
    down = torch.arange(12, dtype=torch.float32).reshape(2, 3, 2)
    other = torch.tensor([99.0])
    selected = {f"{PREFIX}.0.gate_proj.weight"}

    expanded = layout.expand_tensors(
        {GATE_UP: gate_up, DOWN: down, "model.norm.weight": other}, selected
    )

    assert set(expanded) == {
        f"{PREFIX}.{expert}.{projection}.weight"
        for expert in range(2)
        for projection in ("gate_proj", "up_proj", "down_proj")
    } | {"model.norm.weight"}
    torch.testing.assert_close(
        expanded[f"{PREFIX}.0.gate_proj.weight"],
        torch.tensor([[0.0, 1.0], [2.0, 3.0]]),
    )
    torch.testing.assert_close(
        expanded[f"{PREFIX}.1.up_proj.weight"],
        torch.tensor([[12.0, 13.0], [14.0, 15.0]]),
    )
    torch.testing.assert_close(
        expanded[f"{PREFIX}.1.down_proj.weight"],
        torch.tensor([[6.0, 7.0], [8.0, 9.0], [10.0, 11.0]]),
    )
    assert expanded["model.norm.weight"] is other

    gate_up[0, 0, 0] = -1
    gate_up[0, 2, 0] = -2
    assert expanded[f"{PREFIX}.0.gate_proj.weight"][0, 0].item() == -1
    assert expanded[f"{PREFIX}.0.up_proj.weight"][0, 0].item() == 4


@pytest.mark.parametrize(
    "name,tensor,message",
    [
        (GATE_UP, torch.ones(2, 4), "must have shape"),
        (GATE_UP, torch.ones(3, 4, 2), "must have shape"),
        (GATE_UP, torch.ones(2, 5, 2), "even gate/up"),
        (GATE_UP, torch.ones(2, 6, 2), "expected 4"),
        (DOWN, torch.ones(2, 3, 3), "expected 2"),
    ],
)
def test_expand_tensors_rejects_invalid_packed_shapes(name, tensor, message):
    layout = PackedGatedExpertLayout(num_experts=2, moe_intermediate_size=2)

    with pytest.raises(ValueError, match=message):
        layout.expand_tensors({name: tensor})
