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
"""Lightweight adapter tests for Qwen3 / Qwen3-MoE / Qwen3.5 families.

These tests bypass `BaseModel.__init__` (which loads HF tokenizer + config) and
exercise the adapter-level overrides that are pure logic: `parse_quant_mode`,
`get_layer_weight_prefix`, `_embed_base_prefix`, and the MoE-specific
`iter_deploy_bindings` name parsing.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from torch import nn

from amct_pytorch.common.models import MODEL_REGISTRY
from amct_pytorch.common.models.llm.common.base import BaseModel
from amct_pytorch.common.models.llm.qwen.qwen3.qwen3 import Qwen3
from amct_pytorch.common.models.llm.qwen.qwen3.qwen3_moe import Qwen3Moe
from amct_pytorch.common.models.llm.qwen.qwen3_5.qwen3_5 import Qwen3_5
from amct_pytorch.common.models.llm.qwen.qwen3_5.qwen3_5_moe import Qwen3_5Moe
from amct_pytorch.common.models.llm.qwen.qwen3_next.qwen3_next import Qwen3Next
from amct_pytorch.quantization.dtypes import register_dtype
from amct_pytorch.quantization.modules.quant_linear import QuantLinear

register_dtype()

MOE = 'moe'
MLP = 'mlp'
ATTN_LINEAR = 'attn-linear'
ATTN_CACHE = 'attn-cache'


def _stub_adapter(adapter_cls, quant_target):
    """Bypass BaseModel.__init__ and seed only the attributes the adapter
    overrides actually rely on."""
    obj = adapter_cls.__new__(adapter_cls)
    obj.args = SimpleNamespace(quant_target=list(quant_target))
    obj.quant_target = list(quant_target)
    return obj


# ---- parse_quant_mode ----------------------------------------------------


def test_qwen3_dense_rejects_moe_target():
    obj = _stub_adapter(Qwen3, [MOE])
    with pytest.raises(ValueError, match="dense model does not support"):
        obj.parse_quant_mode()


def test_qwen3_dense_accepts_attn_and_mlp_targets():
    for target in ([ATTN_LINEAR], [ATTN_CACHE], [MLP]):
        obj = _stub_adapter(Qwen3, target)
        assert obj.parse_quant_mode() is None


def test_qwen3_moe_rejects_mlp_target():
    obj = _stub_adapter(Qwen3Moe, [MLP])
    with pytest.raises(ValueError, match="does not support quant_target='mlp'"):
        obj.parse_quant_mode()


def test_qwen3_moe_accepts_moe_and_attn_targets():
    for target in ([MOE], [ATTN_LINEAR], [ATTN_CACHE]):
        obj = _stub_adapter(Qwen3Moe, target)
        assert obj.parse_quant_mode() is None


def test_qwen3_5_dense_rejects_moe_target():
    obj = _stub_adapter(Qwen3_5, [MOE])
    with pytest.raises(ValueError, match="dense model and does not support"):
        obj.parse_quant_mode()


def test_qwen3_5_moe_rejects_mlp_target():
    obj = _stub_adapter(Qwen3_5Moe, [MLP])
    with pytest.raises(ValueError, match="does not support quant_target='mlp'"):
        obj.parse_quant_mode()


def test_qwen3_5_moe_accepts_moe_and_attn_targets():
    for target in ([MOE], [ATTN_LINEAR], [ATTN_CACHE]):
        obj = _stub_adapter(Qwen3_5Moe, target)
        assert obj.parse_quant_mode() is None


def _structured_qwen_plan(adapter_cls, names, config):
    from amct_pytorch.common.models.llm.common.deployment.deploy_selection import (
        DeploySelection,
        QuantLayerRule,
        QuantSpec,
        build_tensor_deploy_plan,
        validate_tensor_deploy_plan,
    )

    obj = _stub_adapter(adapter_cls, [MLP])
    obj.args.deploy_platform = "A3"
    obj.config = config
    selection = DeploySelection(
        tuple(QuantLayerRule(name, QuantSpec("int", 8, 8)) for name in names)
    )
    weight_map = {name + ".weight": "model.safetensors" for name in names}
    plan = build_tensor_deploy_plan(weight_map, selection, config=config)
    return validate_tensor_deploy_plan(plan, weight_map, obj)


def test_qwen3_5_structured_tensor_selection_validates_complete_groups():
    prefix = "model.language_model.layers.0."
    names = {
        *(
            prefix + "mlp." + projection
            for projection in ("gate_proj", "up_proj", "down_proj")
        ),
        *(
            prefix + "self_attn." + projection
            for projection in ("q_proj", "k_proj", "v_proj", "o_proj")
        ),
    }
    plan = _structured_qwen_plan(
        Qwen3_5,
        names,
        SimpleNamespace(num_hidden_layers=1, layer_types=["full_attention"]),
    )
    assert plan.selected_modules == names


def test_qwen3_5_moe_structured_tensor_selection_validates_experts():
    prefix = "model.language_model.layers.0."
    names = {
        *(
            prefix + f"mlp.experts.{expert}.{projection}"
            for expert in range(2)
            for projection in ("gate_proj", "up_proj", "down_proj")
        ),
        *(
            prefix + "self_attn." + projection
            for projection in ("q_proj", "k_proj", "v_proj")
        ),
    }
    plan = _structured_qwen_plan(
        Qwen3_5Moe,
        names,
        SimpleNamespace(
            num_hidden_layers=1, num_experts=2, layer_types=["full_attention"]
        ),
    )
    assert plan.selected_modules == names


def test_qwen3_next_rejects_mlp_target():
    obj = _stub_adapter(Qwen3Next, [MLP])
    with pytest.raises(ValueError, match="does not support quant_target='mlp'"):
        obj.parse_quant_mode()


# ---- get_layer_weight_prefix ---------------------------------------------


def test_qwen3_layer_weight_prefix_uses_model_layers():
    obj = _stub_adapter(Qwen3, [MLP])
    assert obj.get_layer_weight_prefix(7) == "model.layers.7."


def test_qwen3_moe_layer_weight_prefix_uses_model_layers():
    obj = _stub_adapter(Qwen3Moe, [MOE])
    assert obj.get_layer_weight_prefix(0) == "model.layers.0."


def test_qwen3_5_uses_model_language_model_prefix():
    obj = _stub_adapter(Qwen3_5, [MLP])
    assert obj.get_layer_weight_prefix(2) == "model.language_model.layers.2."


def test_qwen3_next_uses_model_layers_prefix():
    obj = _stub_adapter(Qwen3Next, [MOE])
    assert obj.get_layer_weight_prefix(5) == "model.layers.5."


# ---- registry registration -----------------------------------------------


@pytest.mark.parametrize(
    "key,expected_cls",
    [
        ("qwen3", Qwen3),
        ("qwen3_moe", Qwen3Moe),
        ("qwen3_5", Qwen3_5),
    ],
)
def test_adapters_register_under_expected_keys(key, expected_cls):
    assert MODEL_REGISTRY.get(key) is expected_cls


def _qwen3_6_dense_stub():
    from amct_pytorch.common.models.llm.qwen.qwen3_6.qwen3_6 import Qwen3_6

    obj = _stub_adapter(Qwen3_6, [MLP])
    obj.config = SimpleNamespace(
        num_hidden_layers=2,
        layer_types=["full_attention", "linear_attention"],
    )
    return obj


def test_qwen3_6_dense_candidates_include_mlp_and_full_attention():
    obj = _qwen3_6_dense_stub()
    expected = {
        "model.language_model.layers.0.mlp.gate_proj",
        "model.language_model.layers.0.mlp.up_proj",
        "model.language_model.layers.0.mlp.down_proj",
        "model.language_model.layers.0.self_attn.q_proj",
        "model.language_model.layers.0.self_attn.k_proj",
        "model.language_model.layers.0.self_attn.v_proj",
        "model.language_model.layers.0.self_attn.o_proj",
        "model.language_model.layers.1.linear_attn.in_proj_qkv",
        "model.language_model.layers.1.linear_attn.in_proj_z",
        "model.language_model.layers.1.linear_attn.in_proj_b",
        "model.language_model.layers.1.linear_attn.in_proj_a",
        "model.language_model.layers.1.linear_attn.out_proj",
    }
    unsupported = {
        "model.language_model.layers.1.self_attn.q_proj",
        "model.visual.blocks.0.attn.qkv",
        "mtp.layers.0.eh_proj",
        "model.language_model.layers.1.linear_attn.conv1d",
        "model.language_model.layers.1.linear_attn.dt_bias",
    }

    assert obj.get_deploy_candidates(expected | unsupported) == expected


def test_qwen3_6_dense_allows_partial_linear_attention_selection():
    obj = _qwen3_6_dense_stub()
    name = "model.language_model.layers.1.linear_attn.in_proj_qkv"

    obj.validate_deploy_selection({name})


def test_qwen3_6_dense_requires_complete_mlp_and_qkvo_groups():
    obj = _qwen3_6_dense_stub()
    mlp_prefix = "model.language_model.layers.0.mlp."
    attn_prefix = "model.language_model.layers.0.self_attn."
    complete = {
        *(mlp_prefix + name for name in ("gate_proj", "up_proj", "down_proj")),
        *(attn_prefix + name for name in ("q_proj", "k_proj", "v_proj", "o_proj")),
    }
    obj.validate_deploy_selection(complete)

    with pytest.raises(ValueError, match="MLP.*together"):
        obj.validate_deploy_selection({mlp_prefix + "gate_proj"})
    with pytest.raises(ValueError, match="QKVO.*together"):
        obj.validate_deploy_selection({attn_prefix + "q_proj"})


def _qwen3_6_moe_stub():
    from amct_pytorch.common.models.llm.qwen.qwen3_6.qwen3_6_moe import Qwen3_6Moe

    obj = _stub_adapter(Qwen3_6Moe, [MOE])
    obj.config = SimpleNamespace(
        num_hidden_layers=2,
        num_experts=2,
        layer_types=["full_attention", "linear_attention"],
    )
    return obj


def test_qwen3_6_moe_candidates_include_linear_attention_projections():
    obj = _qwen3_6_moe_stub()
    linear = {
        "model.language_model.layers.1.linear_attn." + projection
        for projection in (
            "in_proj_qkv",
            "in_proj_z",
            "in_proj_b",
            "in_proj_a",
            "out_proj",
        )
    }
    unsupported = {
        "model.language_model.layers.1.linear_attn.conv1d",
        "model.language_model.layers.1.linear_attn.dt_bias",
    }

    assert obj.get_deploy_candidates(linear | unsupported) == linear


def test_qwen3_6_moe_allows_partial_linear_attention_selection():
    obj = _qwen3_6_moe_stub()
    name = "model.language_model.layers.1.linear_attn.out_proj"

    obj.validate_deploy_selection({name})


def _qwen3_6_plan(module_names, quant_dtype):
    from amct_pytorch.common.models.llm.common.deployment.deploy_selection import (
        DeploySelection,
        QuantLayerRule,
        QuantSpec,
        build_tensor_deploy_plan,
    )

    selection = DeploySelection(
        tuple(
            QuantLayerRule(name, QuantSpec(quant_dtype, 8, 8)) for name in module_names
        )
    )
    weight_map = {name + ".weight": "model.safetensors" for name in module_names}
    return weight_map, build_tensor_deploy_plan(weight_map, selection)


@pytest.mark.parametrize(
    "platform,quant_dtype",
    [("A3", "mxfp")],
)
def test_qwen3_6_dense_rejects_dtype_unsupported_by_platform(platform, quant_dtype):
    from amct_pytorch.common.models.llm.common.deployment.deploy_selection import (
        validate_tensor_deploy_plan,
    )

    obj = _qwen3_6_dense_stub()
    obj.args.deploy_platform = platform
    prefix = "model.language_model.layers.0.mlp."
    names = {prefix + name for name in ("gate_proj", "up_proj", "down_proj")}
    weight_map, plan = _qwen3_6_plan(names, quant_dtype)

    with pytest.raises(ValueError, match=platform):
        validate_tensor_deploy_plan(plan, weight_map, obj)


@pytest.mark.parametrize(
    "platform,quant_dtype",
    [("A3", "int"), ("ascend950", "int"), ("ascend950", "mxfp")],
)
def test_qwen3_6_dense_accepts_supported_platform_dtype_pairs(platform, quant_dtype):
    from amct_pytorch.common.models.llm.common.deployment.deploy_selection import (
        validate_tensor_deploy_plan,
    )

    obj = _qwen3_6_dense_stub()
    obj.args.deploy_platform = platform
    prefix = "model.language_model.layers.0.mlp."
    names = {prefix + name for name in ("gate_proj", "up_proj", "down_proj")}
    weight_map, plan = _qwen3_6_plan(names, quant_dtype)

    assert validate_tensor_deploy_plan(plan, weight_map, obj) is plan


# ---- iter_deploy_bindings (Qwen3Moe) -------------------------------------


def _build_moe_block_with_quant_linears():
    """Build a fake decoder layer matching Qwen3Moe's iter_deploy_bindings name
    scheme (`mlp.experts.expert_modules.<idx>.<proj>`)."""
    block = nn.Module()
    block.self_attn = nn.Module()
    block.self_attn.q_proj = QuantLinear(
        SimpleNamespace(algos=[], quant_dtype="int", w_bits=8),
        nn.Linear(4, 4),
        w_bits=8,
        name="q_proj",
        a_bits=16,
    )

    # MoE expert hierarchy: mlp.experts.expert_modules.{idx}.{proj_name}
    block.mlp = nn.Module()
    block.mlp.experts = nn.Module()
    block.mlp.experts.expert_modules = nn.ModuleList()
    for _ in range(2):
        expert = nn.Module()
        expert.up_proj = QuantLinear(
            SimpleNamespace(algos=[], quant_dtype="int", w_bits=8),
            nn.Linear(4, 4),
            w_bits=8,
            name="up_proj",
            a_bits=16,
        )
        block.mlp.experts.expert_modules.append(expert)
    return block


def test_qwen3_moe_iter_deploy_bindings_rewrites_expert_path():
    obj = _stub_adapter(Qwen3Moe, [MOE])
    block = _build_moe_block_with_quant_linears()
    bindings = list(obj.iter_deploy_bindings(layer_idx=3, block=block))
    names = [b[0] for b in bindings]
    # Self-attn QuantLinear stays untouched (just gets the prefix prepended).
    assert "model.layers.3.self_attn.q_proj.weight" in names
    # Each expert is rewritten to model.layers.3.mlp.experts.<idx>.up_proj.weight.
    assert "model.layers.3.mlp.experts.0.up_proj.weight" in names
    assert "model.layers.3.mlp.experts.1.up_proj.weight" in names
    # The original expert_modules.<idx> path must NOT appear.
    assert all("expert_modules" not in n for n in names)


# ---- Qwen3 adapter delegation methods (lines 35, 70, 82) --------------------


def test_qwen3_float_model_delegates_to_super():
    obj = _stub_adapter(Qwen3, [MLP])
    with patch.object(BaseModel, 'float_model', return_value="called"):
        assert obj.float_model() == "called"


def test_qwen3_build_quant_block_applies_attn():
    obj = _stub_adapter(Qwen3, [ATTN_LINEAR])
    decoder_layer = MagicMock()
    decoder_layer.self_attn = MagicMock()
    with (
        patch.object(obj, 'block', return_value=decoder_layer),
        patch(
            'amct_pytorch.common.models.llm.qwen.qwen3.qwen3.apply_quant_to_attn'
        ) as mock_apply,
    ):
        result = obj.build_quant_block(0)
        mock_apply.assert_called_once()
        assert result is decoder_layer


def test_qwen3_load_unit_inputs_delegates_to_super():
    obj = _stub_adapter(Qwen3, [MLP])
    with patch.object(BaseModel, 'load_unit_inputs', return_value="called"):
        assert obj.load_unit_inputs("/data", "unit") == "called"


def test_qwen3_moe_iter_deploy_bindings_raises_on_malformed_expert_name():
    """Construct a name that starts with the expert prefix but has the wrong arity."""
    obj = _stub_adapter(Qwen3Moe, [MOE])
    block = nn.Module()
    block.mlp = nn.Module()
    block.mlp.experts = nn.Module()
    block.mlp.experts.expert_modules = nn.Module()
    # Single attribute `bad` -> name `mlp.experts.expert_modules.bad`, only 4 parts.
    block.mlp.experts.expert_modules.bad = QuantLinear(
        SimpleNamespace(algos=[], quant_dtype="int", w_bits=8),
        nn.Linear(4, 4),
        w_bits=8,
        name="bad",
        a_bits=16,
    )
    with pytest.raises(ValueError, match="Unexpected Qwen3 MoE expert module name"):
        list(obj.iter_deploy_bindings(layer_idx=0, block=block))


# ---- iter_deploy_bindings (Qwen3_5Moe) -----------------------------------


def test_qwen3_5_moe_iter_deploy_bindings_rewrites_expert_path():
    obj = _stub_adapter(Qwen3_5Moe, [MOE])
    block = _build_moe_block_with_quant_linears()
    bindings = list(obj.iter_deploy_bindings(layer_idx=4, block=block))
    names = [b[0] for b in bindings]

    assert "model.language_model.layers.4.self_attn.q_proj.weight" in names
    assert "model.language_model.layers.4.mlp.experts.0.up_proj.weight" in names
    assert "model.language_model.layers.4.mlp.experts.1.up_proj.weight" in names
    assert all("expert_modules" not in n for n in names)


def test_qwen3_5_moe_iter_deploy_bindings_keeps_non_expert_path():
    obj = _stub_adapter(Qwen3_5Moe, [MOE])
    block = nn.Module()
    block.mlp = nn.Module()
    block.mlp.shared_expert = nn.Module()
    block.mlp.shared_expert.down_proj = QuantLinear(
        SimpleNamespace(algos=[], quant_dtype="int", w_bits=8),
        nn.Linear(4, 4),
        w_bits=8,
        name="down_proj",
        a_bits=16,
    )

    bindings = list(obj.iter_deploy_bindings(layer_idx=1, block=block))

    assert len(bindings) == 1
    assert (
        bindings[0][0]
        == "model.language_model.layers.1.mlp.shared_expert.down_proj.weight"
    )


def test_qwen3_5_moe_iter_deploy_bindings_raises_on_malformed_expert_name():
    obj = _stub_adapter(Qwen3_5Moe, [MOE])
    block = nn.Module()
    block.mlp = nn.Module()
    block.mlp.experts = nn.Module()
    block.mlp.experts.expert_modules = nn.Module()
    block.mlp.experts.expert_modules.bad = QuantLinear(
        SimpleNamespace(algos=[], quant_dtype="int", w_bits=8),
        nn.Linear(4, 4),
        w_bits=8,
        name="bad",
        a_bits=16,
    )

    with pytest.raises(ValueError, match="Unexpected Qwen3.5 MoE expert module name"):
        list(obj.iter_deploy_bindings(layer_idx=0, block=block))
