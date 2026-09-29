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
from types import SimpleNamespace

import pytest

from amct_pytorch.common.models.llm.common.deploy_selection import (
    DeploySelection,
    load_deploy_selection,
    match_deploy_modules,
    validate_ascend_deploy_args,
    validate_tensor_deploy_args,
)
from amct_pytorch.quantization.bit_policy import BitPolicy


def test_tensor_validation_skips_model_without_json_deploy_config():
    args = SimpleNamespace(model_name="deepseek_v4", quant_dtype="")
    validate_tensor_deploy_args(args)


def test_load_defaults_and_deduplicates(tmp_path):
    path = tmp_path / "layers.json"
    path.write_text(json.dumps({"quant_layers": ["model.*", "model.*"]}))
    assert load_deploy_selection(path) == DeploySelection(("model.*",), ())


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
    rules = DeploySelection(("model.layers.*.q",), ("model.layers.78.*", "model.norm"))
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
        names, DeploySelection(("model.layers.?.q",), ())
    )
    assert selected == {"model.layers.0.q"}


@pytest.mark.parametrize(
    "rules",
    [
        DeploySelection(("missing.*",), ()),
        DeploySelection(("model.*", "missing.*"), ("missing.*",)),
    ],
)
def test_unmatched_rule_is_an_error(rules):
    with pytest.raises(ValueError, match="missing"):
        match_deploy_modules({"model.q"}, rules)


def test_empty_final_selection_is_an_error():
    with pytest.raises(ValueError, match="No modules"):
        match_deploy_modules({"model.q"}, DeploySelection(("*",), ("*",)))


def test_duplicate_rules_do_not_double_count():
    selected, counts = match_deploy_modules(
        {"model.q"}, DeploySelection(("*", "*"), ())
    )
    assert selected == {"model.q"}
    assert counts == {"*": 1}


def _args(**overrides):
    args = dict(
        deploy_format="ascend",
        model_name="glm5_2",
        granularity="tensor",
        quant_dtype="int",
        quant_layers_config="layers.json",
        bit_policy=BitPolicy({"w_bits": 8, "a_bits": 8}),
        algos=[],
    )
    args.update(overrides)
    return SimpleNamespace(**args)


def test_valid_ascend_args_and_legacy_defaults():
    validate_ascend_deploy_args(_args())
    validate_ascend_deploy_args(_args(deploy_format="legacy"))
    validate_ascend_deploy_args(_args(model_name="qwen3_6_moe"))
    validate_ascend_deploy_args(SimpleNamespace())


@pytest.mark.parametrize(
    "override, message",
    [
        ({"deploy_format": "other"}, "deploy_format"),
        ({"quant_layers_config": ""}, "quant_layers_config"),
        ({"granularity": "block"}, "tensor"),
        ({"quant_dtype": "mxfp"}, "int"),
        ({"algos": ["lwc"]}, "algos"),
        ({"attn_linear_param_dir": "/ptq"}, "attn_linear_param_dir"),
        ({"attn_cache_param_dir": "/ptq"}, "attn_cache_param_dir"),
        ({"moe_mlp_param_dir": "/ptq"}, "moe_mlp_param_dir"),
    ],
)
def test_reject_incompatible_args(override, message):
    with pytest.raises(ValueError, match=message):
        validate_ascend_deploy_args(_args(**override))


def _module_policy():
    return BitPolicy(
        {group: {"w_bits": 8, "a_bits": 8} for group in ("attn-linear", "mlp", "moe")}
    )


def test_public_plan_resolves_ignore_and_explicit_bits():
    from amct_pytorch.common.models.llm.common import deploy_selection as selection

    names = [
        "model.layers.0.self_attn.o_proj",
        "model.layers.1.mlp.experts.0.up_proj",
        "lm_head",
    ]
    weights = {name + ".weight": "one.safetensors" for name in names}
    weights[names[0] + ".bias"] = "one.safetensors"
    plan = selection.build_tensor_deploy_plan(
        weights, DeploySelection(("*",), ("lm_head",)), _module_policy()
    )
    assert plan.quant_layers == {name: 8 for name in names[:2]}
    assert plan.selected_weight_keys == {name + ".weight" for name in names[:2]}
    assert plan.ignored_modules == {"lm_head"}
    assert plan.unselected_weight_modules == {"lm_head"}
    assert plan.module_bits[names[0]] == (8, 8)


@pytest.mark.parametrize(
    "cfg, message",
    [
        ({"moe": {"w_bits": 4, "a_bits": 8}}, "moe.w_bits"),
        ({"attn-linear": {"w_bits": 16, "a_bits": 16}}, "W8A8"),
        ({"attn-linear": {"w_bit": 8, "a_bits": 8}}, "w_bit"),
        ({"attn-lienar": {"w_bits": 8, "a_bits": 8}}, "attn-lienar"),
    ],
)
def test_tensor_policy_fail_fast_including_unused_groups(cfg, message):
    from amct_pytorch.common.models.llm.common import deploy_selection as selection

    # Direct cfg validation also precedes BitPolicy's general YAML validation.
    with pytest.raises(ValueError, match=message):
        selection.validate_tensor_bit_config(cfg)


def test_missing_group_is_not_implicit_16_or_global_fallback():
    from amct_pytorch.common.models.llm.common import deploy_selection as selection

    with pytest.raises(ValueError, match="Missing.*attn-linear"):
        selection.build_tensor_deploy_plan(
            {"model.layers.0.self_attn.o_proj.weight": "x"},
            DeploySelection(("*",)),
            BitPolicy({"moe": {"w_bits": 8, "a_bits": 8}}),
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
    from amct_pytorch.common.models.llm.common import deploy_selection as selection

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
        DeploySelection(quant, ignore),
        _module_policy(),
        config=SimpleNamespace(num_hidden_layers=2, num_nextn_predict_layers=1),
    )
    assert plan.mtp.status == status
    assert ("model.layers.2.self_attn.o_proj" in plan.selected_modules) == (
        status in {"selected", "partially_ignored"}
    )
    assert len(plan.mtp.modules) == 2


def test_no_mtp_and_nonweight_selection_rejected():
    from amct_pytorch.common.models.llm.common import deploy_selection as selection

    weights = {"model.layers.0.self_attn.o_proj.weight": "x"}
    plan = selection.build_tensor_deploy_plan(
        weights, DeploySelection(("*",)), _module_policy()
    )
    assert plan.mtp.status == "not_present"
    with pytest.raises(ValueError, match="weight"):
        selection.build_tensor_deploy_plan(
            {"model.layers.0.self_attn.o_proj.bias": "x"},
            DeploySelection(("*",)),
            _module_policy(),
        )


def test_mtp_log_distinguishes_planned_and_exported_with_bits():
    from amct_pytorch.common.models.llm.common import deploy_selection as selection

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


@pytest.mark.parametrize(
    "cfg",
    [
        {"w_bits": 4, "a_bits": 8, "moe": {"w_bits": 8, "a_bits": 8}},
        {"moe": {"w_bits": 8, "a_bits": 8, "routed": {"w_bits": 4, "a_bits": 8}}},
    ],
)
def test_explicit_w4_rejected_even_when_overridden_or_unselected(cfg):
    from amct_pytorch.common.models.llm.common import deploy_selection as selection

    with pytest.raises(ValueError, match="W8A8"):
        selection.validate_tensor_bit_config(cfg)


def test_unmatched_ignore_warns_and_is_recorded_without_changing_selection():
    from amct_pytorch.common.models.llm.common import deploy_selection as selection

    name = "model.layers.0.self_attn.o_proj"
    absent = "model.layers.*.shared_head.head"
    messages = []
    sink = selection.logger.add(
        lambda message: messages.append(str(message)), format="{level}: {message}"
    )
    try:
        plan = selection.build_tensor_deploy_plan(
            {name + ".weight": "x"},
            DeploySelection((name,), (absent,)),
            _module_policy(),
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


def test_ascend_args_do_not_restrict_model_name():
    validate_ascend_deploy_args(_args(model_name="hy_v3"))


def test_ascend_plan_requires_adapter_capabilities():
    from amct_pytorch.common.models.llm.common.deploy_selection import (
        validate_ascend_deploy_plan,
    )

    with pytest.raises(
        ValueError, match="Ascend.*adapter.*get_ascend_deploy_candidates"
    ):
        validate_ascend_deploy_plan(SimpleNamespace(), {}, SimpleNamespace())
