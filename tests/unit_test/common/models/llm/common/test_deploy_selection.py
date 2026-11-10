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

"""Deploy-only selection schema, glob semantics and argument contract."""

import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from amct_pytorch.common.models.llm.common.deployment.deploy_selection import (
    DeploySelection,
    QuantLayerRule,
    QuantSpec,
    build_tensor_deploy_plan,
    load_deploy_selection,
    match_deploy_modules,
    prepare_deploy_args,
)
from amct_pytorch.quantization.bit_policy import BitPolicy


def _structured_rules(*patterns, quant_dtype="int"):
    return tuple(
        QuantLayerRule(pattern, QuantSpec(quant_dtype, 8, 8)) for pattern in patterns
    )


def test_tensor_validation_requires_json_for_tensor_deploy():
    args = SimpleNamespace(
        model_name="deepseek_v4", granularity="tensor", quant_dtype=""
    )
    with pytest.raises(
        ValueError, match=re.escape("Deployment requires --quant_layers_config")
    ):
        prepare_deploy_args(args)


def test_tensor_validation_replaces_prebuilt_selection_with_json_rules(tmp_path):
    config = tmp_path / "selection.json"
    config.write_text(
        json.dumps(
            {
                "quant_layers": [
                    {
                        "pattern": "model.layers.*.self_attn.q_proj",
                        "quant_dtype": "int",
                        "w_bits": 8,
                        "a_bits": 8,
                    }
                ]
            }
        )
    )
    args = SimpleNamespace(
        model_name="qwen3_5",
        granularity="tensor",
        quant_dtype="int",
        quant_layers_config=str(config),
        deploy_selection=DeploySelection(("model.layers.*.mlp.down_proj",)),
    )

    prepare_deploy_args(args)

    assert args.deploy_selection == DeploySelection(
        _structured_rules("model.layers.*.self_attn.q_proj")
    )
    assert args.deploy_selection.uses_structured_rules


def test_tensor_validation_rejects_legacy_json_rules(tmp_path):
    config = tmp_path / "selection.json"
    config.write_text(json.dumps({"quant_layers": ["model.layers.*"]}))
    args = SimpleNamespace(
        model_name="qwen3_5",
        granularity="tensor",
        quant_dtype="int",
        quant_layers_config=str(config),
    )
    with pytest.raises(ValueError, match="structured rules"):
        prepare_deploy_args(args)


def test_prepare_deploy_args_leaves_model_specific_dtype_to_adapter(tmp_path):
    config = tmp_path / "selection.json"
    config.write_text(
        json.dumps(
            {
                "quant_layers": [
                    {
                        "pattern": "model.layers.*.mlp.down_proj",
                        "quant_dtype": "mxfp",
                        "w_bits": 8,
                        "a_bits": 8,
                    }
                ]
            }
        )
    )
    args = SimpleNamespace(
        model_name="deepseek_v4_1",
        granularity="tensor",
        quant_dtype="",
        deploy_platform="ascend950",
        quant_layers_config=str(config),
    )

    prepare_deploy_args(args)

    assert args.quant_dtype == ""
    assert args.deploy_selection.uses_structured_rules


@pytest.mark.parametrize(
    "quant_dtype, error, message",
    [
        ("hifp", NotImplementedError, "temporarily unsupported"),
        ("hifp8", ValueError, "requires --quant_layers_config"),
    ],
)
def test_tensor_hifp_is_rejected_early(quant_dtype, error, message):
    with pytest.raises(error, match=message):
        prepare_deploy_args(
            SimpleNamespace(granularity="tensor", quant_dtype=quant_dtype)
        )


def test_load_preserves_legacy_string_rules_for_model_adapters(tmp_path):
    path = tmp_path / "layers.json"
    path.write_text(json.dumps({"quant_layers": ["model.*", "model.*"]}))
    assert load_deploy_selection(path) == DeploySelection(("model.*",), ())


def test_load_structured_quant_rules(tmp_path):
    path = tmp_path / "layers.json"
    path.write_text(
        json.dumps(
            {
                "quant_layers": [
                    {
                        "pattern": "model.layers.*.self_attn.q_proj",
                        "quant_dtype": "int",
                        "w_bits": 8,
                        "a_bits": 8,
                    },
                    {
                        "pattern": "model.layers.*.mlp.down_proj",
                        "quant_dtype": "mxfp",
                        "w_bits": 8,
                        "a_bits": 8,
                    },
                ],
                "ignore_layers": ["model.layers.1.*"],
            }
        )
    )

    assert load_deploy_selection(path) == DeploySelection(
        (
            QuantLayerRule("model.layers.*.self_attn.q_proj", QuantSpec("int", 8, 8)),
            QuantLayerRule("model.layers.*.mlp.down_proj", QuantSpec("mxfp", 8, 8)),
        ),
        ("model.layers.1.*",),
    )


@pytest.mark.parametrize(
    "filename,quant_dtype",
    [
        ("qwen3_6/qwen3_6_w8a8_int.json", "int"),
        ("qwen3_6/qwen3_6_w8a8_mxfp8.json", "mxfp"),
        ("qwen3_6_moe/qwen3_6_moe_w8a8_int.json", "int"),
        ("qwen3_6_moe/qwen3_6_moe_w8a8_mxfp8.json", "mxfp"),
    ],
)
def test_qwen3_6_linear_attention_configs_select_only_linear_projections(
    filename, quant_dtype
):
    import amct_pytorch

    config_path = (
        Path(amct_pytorch.__file__).parent / "configs" / "deploy" / "qwen" / filename
    )
    selection = load_deploy_selection(config_path)
    linear_patterns = {
        "model.language_model.layers.*.linear_attn." + projection
        for projection in (
            "in_proj_qkv",
            "in_proj_z",
            "in_proj_b",
            "in_proj_a",
            "out_proj",
        )
    }

    assert selection.uses_structured_rules
    assert {
        rule.pattern for rule in selection.quant_layers if "linear_attn" in rule.pattern
    } == linear_patterns
    assert all(
        rule.quant_spec == QuantSpec(quant_dtype, 8, 8)
        for rule in selection.quant_layers
    )
    assert "model.language_model.layers.*.linear_attn.*" not in selection.ignore_layers


def test_structured_tensor_args_do_not_require_global_dtype_or_bit_policy(tmp_path):
    path = tmp_path / "layers.json"
    path.write_text(
        json.dumps(
            {
                "quant_layers": [
                    {
                        "pattern": "model.language_model.layers.*.mlp.gate_proj",
                        "quant_dtype": "mxfp",
                        "w_bits": 8,
                        "a_bits": 8,
                    }
                ]
            }
        )
    )
    args = SimpleNamespace(
        deploy_platform="ascend950",
        model_name="qwen3_6",
        granularity="tensor",
        quant_dtype=None,
        quant_layers_config=str(path),
        bit_config=None,
        algos=[],
    )

    prepare_deploy_args(args)

    assert args.deploy_selection.uses_structured_rules
    assert not hasattr(args, "bit_policy")


@pytest.mark.parametrize(
    "rule, message",
    [
        (
            {"quant_dtype": "int", "w_bits": 8, "a_bits": 8},
            "pattern",
        ),
        (
            {"pattern": "model.*", "w_bits": 8, "a_bits": 8},
            "quant_dtype",
        ),
        (
            {
                "pattern": "model.*",
                "quant_dtype": "fp8",
                "w_bits": 8,
                "a_bits": 8,
            },
            "quant_dtype",
        ),
        (
            {
                "pattern": "model.*",
                "quant_dtype": [],
                "w_bits": 8,
                "a_bits": 8,
            },
            "quant_dtype",
        ),
        (
            {
                "pattern": "model.*",
                "quant_dtype": "int",
                "w_bits": True,
                "a_bits": 8,
            },
            "w_bits",
        ),
        (
            {
                "pattern": "model.*.weight",
                "quant_dtype": "int",
                "w_bits": 8,
                "a_bits": 8,
            },
            "weight",
        ),
        (
            {
                "pattern": "model.*",
                "quant_dtype": "int",
                "w_bits": 8,
                "a_bits": 8,
                "group_size": 128,
            },
            "group_size",
        ),
    ],
)
def test_reject_invalid_structured_quant_rule(tmp_path, rule, message):
    path = tmp_path / "layers.json"
    path.write_text(json.dumps({"quant_layers": [rule]}))

    with pytest.raises(ValueError, match=message):
        load_deploy_selection(path)


def test_reject_mixed_legacy_and_structured_quant_rules(tmp_path):
    path = tmp_path / "layers.json"
    path.write_text(
        json.dumps(
            {
                "quant_layers": [
                    "model.layers.*.self_attn.q_proj",
                    {
                        "pattern": "model.layers.*.mlp.down_proj",
                        "quant_dtype": "int",
                        "w_bits": 8,
                        "a_bits": 8,
                    },
                ]
            }
        )
    )

    with pytest.raises(ValueError, match="same schema"):
        load_deploy_selection(path)


@pytest.mark.parametrize(
    "data",
    [
        [],
        {},
        {"quant_layers": []},
        {"quant_layers": "*"},
        {"quant_layers": [1]},
        {"quant_layers": [" "]},
        {"quant_layers": ["*"], "ignore_layers": None},
        {"quant_layers": ["*"], "ignore_layers": [False]},
        {"quant_layers": ["*"], "ignored_layers": []},
    ],
)
def test_reject_invalid_schema(tmp_path, data):
    path = tmp_path / "layers.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="quant_layers|ignore_layers|fields|object"):
        load_deploy_selection(path)


def test_invalid_json_reports_source(tmp_path):
    path = tmp_path / "broken.json"
    path.write_text("{")
    with pytest.raises(ValueError, match="broken.json"):
        load_deploy_selection(path)


def test_missing_file_reports_source(tmp_path):
    with pytest.raises(ValueError, match="missing.json"):
        load_deploy_selection(tmp_path / "missing.json")


def test_ignore_precedence_and_raw_match_counts():
    names = {"model.layers.0.q", "model.layers.78.q", "model.norm"}
    rules = DeploySelection(
        _structured_rules("model.layers.*.q"),
        ("model.layers.78.*", "model.norm"),
    )
    selected, counts = match_deploy_modules(names, rules)
    assert selected == {"model.layers.0.q"}
    assert counts == {"model.layers.*.q": 2, "model.layers.78.*": 1, "model.norm": 1}


def test_full_case_sensitive_glob_and_question_mark():
    names = {
        "model.layers.0.q",
        "model.layers.10.q",
        "MODEL.layers.0.q",
        "prefix.model.layers.0.q",
    }
    selected, _ = match_deploy_modules(
        names, DeploySelection(_structured_rules("model.layers.?.q"), ())
    )
    assert selected == {"model.layers.0.q"}


@pytest.mark.parametrize(
    "rules",
    [
        DeploySelection(_structured_rules("missing.*"), ()),
        DeploySelection(_structured_rules("model.*", "missing.*"), ("missing.*",)),
    ],
)
def test_unmatched_rule_is_an_error(rules):
    with pytest.raises(ValueError, match="missing"):
        match_deploy_modules({"model.q"}, rules)


def test_empty_final_selection_is_an_error():
    with pytest.raises(ValueError, match="No modules"):
        match_deploy_modules(
            {"model.q"}, DeploySelection(_structured_rules("*"), ("*",))
        )


def test_duplicate_structured_patterns_are_rejected():
    with pytest.raises(ValueError, match="multiple quant_layers patterns"):
        match_deploy_modules(
            {"model.q"}, DeploySelection(_structured_rules("*", "*"), ())
        )


def test_duplicate_legacy_patterns_do_not_double_count():
    selected, counts = match_deploy_modules(
        {"model.q"}, DeploySelection(("*", "*"), ())
    )
    assert selected == {"model.q"}
    assert counts == {"*": 1}


def test_duplicate_structured_patterns_with_different_specs_are_rejected():
    selection = DeploySelection(
        (
            QuantLayerRule("*", QuantSpec("int", 8, 8)),
            QuantLayerRule("*", QuantSpec("mxfp", 8, 8)),
        )
    )

    with pytest.raises(ValueError, match="multiple quant_layers patterns"):
        match_deploy_modules({"model.q"}, selection)


def _module_policy():
    return BitPolicy(
        {group: {"w_bits": 8, "a_bits": 8} for group in ("attn-linear", "mlp", "moe")}
    )


def test_public_plan_resolves_ignore_and_explicit_bits():
    import amct_pytorch.common.models.llm.common.deployment.deploy_selection as selection

    names = [
        "model.layers.0.self_attn.o_proj",
        "model.layers.1.mlp.experts.0.up_proj",
        "lm_head",
    ]
    weights = {name + ".weight": "one.safetensors" for name in names}
    weights[names[0] + ".bias"] = "one.safetensors"
    plan = selection.build_tensor_deploy_plan(
        weights, DeploySelection(_structured_rules("*"), ("lm_head",))
    )
    assert plan.quant_layers == {name: 8 for name in names[:2]}
    assert plan.selected_weight_keys == {name + ".weight" for name in names[:2]}
    assert plan.ignored_modules == {"lm_head"}
    assert plan.unselected_weight_modules == {"lm_head"}
    assert plan.module_bits[names[0]] == (8, 8)


def test_structured_plan_resolves_per_module_quant_specs_without_bit_policy():
    q_proj = "model.layers.0.self_attn.q_proj"
    down_proj = "model.layers.0.mlp.down_proj"
    selection = DeploySelection(
        (
            QuantLayerRule("*.self_attn.q_proj", QuantSpec("int", 8, 8)),
            QuantLayerRule("*.mlp.down_proj", QuantSpec("mxfp", 8, 8)),
        )
    )

    plan = build_tensor_deploy_plan(
        {
            q_proj + ".weight": "one.safetensors",
            down_proj + ".weight": "one.safetensors",
        },
        selection,
    )

    assert plan.module_quant_specs == {
        q_proj: QuantSpec("int", 8, 8),
        down_proj: QuantSpec("mxfp", 8, 8),
    }
    assert plan.module_bits == {q_proj: (8, 8), down_proj: (8, 8)}
    assert plan.quant_layers == {q_proj: 8, down_proj: 8}


def test_structured_plan_rejects_overlapping_quant_rules():
    name = "model.layers.0.self_attn.q_proj"
    selection = DeploySelection(
        (
            QuantLayerRule("model.layers.*.self_attn.*", QuantSpec("int", 8, 8)),
            QuantLayerRule("*.q_proj", QuantSpec("int", 8, 8)),
        )
    )

    with pytest.raises(ValueError, match="multiple quant_layers patterns.*q_proj"):
        build_tensor_deploy_plan({name + ".weight": "one.safetensors"}, selection)


def test_structured_plan_ignore_precedence_suppresses_quant_rule_overlap():
    selected = "model.layers.0.self_attn.q_proj"
    ignored = "model.layers.1.self_attn.q_proj"
    selection = DeploySelection(
        (
            QuantLayerRule("model.layers.*.self_attn.*", QuantSpec("int", 8, 8)),
            QuantLayerRule("model.layers.1.*", QuantSpec("mxfp", 8, 8)),
        ),
        ("model.layers.1.*",),
    )

    plan = build_tensor_deploy_plan(
        {
            selected + ".weight": "one.safetensors",
            ignored + ".weight": "one.safetensors",
        },
        selection,
    )

    assert plan.module_quant_specs == {selected: QuantSpec("int", 8, 8)}
    assert plan.ignored_modules == {ignored}


@pytest.mark.parametrize("spec", [QuantSpec("int", 4, 8), QuantSpec("mxfp", 4, 4)])
def test_structured_plan_rejects_reserved_w4_specs(spec):
    name = "model.layers.0.self_attn.q_proj"

    with pytest.raises(ValueError, match="W8A8"):
        build_tensor_deploy_plan(
            {name + ".weight": "one.safetensors"},
            DeploySelection((QuantLayerRule(name, spec),)),
        )


def test_missing_group_is_not_implicit_16_or_global_fallback():
    import amct_pytorch.common.models.llm.common.deployment.deploy_selection as selection

    with pytest.raises(ValueError, match="structured"):
        selection.build_tensor_deploy_plan(
            {"model.layers.0.self_attn.o_proj.weight": "x"},
            DeploySelection(("*",)),
        )


@pytest.mark.parametrize(
    "ignore,quant, status",
    [
        (("model.layers.2.*",), ("model.layers.*.self_attn.o_proj",), "fully_ignored"),
        ((), ("model.layers.0.*",), "not_selected"),
        ((), ("model.layers.*.self_attn.o_proj",), "selected"),
        (
            ("model.layers.2.eh_proj",),
            ("model.layers.*.self_attn.o_proj",),
            "partially_ignored",
        ),
    ],
)
def test_mtp_report_does_not_change_selection(ignore, quant, status):
    import amct_pytorch.common.models.llm.common.deployment.deploy_selection as selection

    weights = {
        name + ".weight": "x"
        for name in (
            "model.layers.0.self_attn.o_proj",
            "model.layers.2.self_attn.o_proj",
            "model.layers.2.eh_proj",
        )
    }
    plan = selection.build_tensor_deploy_plan(
        weights,
        DeploySelection(_structured_rules(*quant), ignore),
        config=SimpleNamespace(num_hidden_layers=2, num_nextn_predict_layers=1),
    )
    assert plan.mtp.status == status
    assert ("model.layers.2.self_attn.o_proj" in plan.selected_modules) == (
        status in {"selected", "partially_ignored"}
    )
    assert len(plan.mtp.modules) == 2


def test_no_mtp_and_nonweight_selection_rejected():
    import amct_pytorch.common.models.llm.common.deployment.deploy_selection as selection

    weights = {"model.layers.0.self_attn.o_proj.weight": "x"}
    plan = selection.build_tensor_deploy_plan(
        weights, DeploySelection(_structured_rules("*"))
    )
    assert plan.mtp.status == "not_present"
    with pytest.raises(ValueError, match="weight"):
        selection.build_tensor_deploy_plan(
            {"model.layers.0.self_attn.o_proj.bias": "x"},
            DeploySelection(_structured_rules("*")),
        )


def test_mtp_log_distinguishes_planned_and_exported_with_bits():
    import amct_pytorch.common.models.llm.common.deployment.deploy_selection as selection

    messages = []
    sink = selection.logger.add(
        lambda message: messages.append(str(message)), format="{message}"
    )
    name = "model.layers.2.self_attn.o_proj"
    report = selection.MtpSelectionReport(
        frozenset({name}), frozenset(), frozenset({name})
    )
    try:
        selection.log_mtp_selection(report, module_bits={name: (8, 8)})
        selection.log_mtp_selection(
            report,
            phase="exported",
            quantized_modules={name},
            module_bits={name: (8, 8)},
        )
    finally:
        selection.logger.remove(sink)
    assert "planned" in messages[0] and "quantized=0" in messages[0]
    assert "exported" in messages[1] and "quantized=1" in messages[1]
    assert "(8, 8)" in messages[0] and name in messages[0]


def test_unmatched_ignore_warns_and_is_recorded_without_changing_selection():
    import amct_pytorch.common.models.llm.common.deployment.deploy_selection as selection

    name = "model.layers.0.self_attn.o_proj"
    absent = "model.layers.*.shared_head.head"
    messages = []
    sink = selection.logger.add(
        lambda message: messages.append(str(message)), format="{level}: {message}"
    )
    try:
        plan = selection.build_tensor_deploy_plan(
            {name + ".weight": "x"},
            DeploySelection(_structured_rules(name), (absent,)),
        )
    finally:
        selection.logger.remove(sink)
    assert plan.quant_layers == {name: 8}
    assert plan.ignored_modules == frozenset()
    assert plan.unmatched_ignore_patterns == {absent}
    assert plan.match_counts[absent] == 0
    assert any(
        "WARNING" in message and "ignore_layers" in message and absent in message
        for message in messages
    )


def test_ascend_plan_requires_adapter_capabilities():
    from amct_pytorch.common.models.llm.common.deployment.deploy_selection import (
        validate_tensor_deploy_plan,
    )

    with pytest.raises(ValueError, match="Deployment.*adapter.*get_deploy_candidates"):
        validate_tensor_deploy_plan(SimpleNamespace(), {}, SimpleNamespace())


def test_ascend_plan_rejects_mixed_quant_dtypes_before_export():
    from amct_pytorch.common.models.llm.common.deployment.deploy_selection import (
        validate_tensor_deploy_plan,
    )

    names = {
        "model.layers.0.self_attn.q_proj": QuantSpec("int", 8, 8),
        "model.layers.0.mlp.down_proj": QuantSpec("mxfp", 8, 8),
    }
    plan = build_tensor_deploy_plan(
        {name + ".weight": "one.safetensors" for name in names},
        DeploySelection(
            tuple(QuantLayerRule(name, spec) for name, spec in names.items())
        ),
    )
    adapter = SimpleNamespace(
        args=SimpleNamespace(deploy_platform="A3"),
        get_deploy_candidates=lambda module_names: module_names,
        validate_deploy_selection=lambda selected: None,
    )

    with pytest.raises(ValueError, match="one quant_dtype"):
        validate_tensor_deploy_plan(
            plan,
            {name + ".weight": "one.safetensors" for name in names},
            adapter,
        )
