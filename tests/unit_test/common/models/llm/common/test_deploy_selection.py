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
)
from amct_pytorch.quantization.bit_policy import BitPolicy


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
        DeploySelection(("model.*",), ("missing.*",)),
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
    validate_ascend_deploy_args(SimpleNamespace())


@pytest.mark.parametrize(
    "override, message",
    [
        ({"deploy_format": "other"}, "deploy_format"),
        ({"deploy_format": "legacy"}, "quant_layers_config"),
        ({"quant_layers_config": ""}, "quant_layers_config"),
        ({"model_name": "hy_v3"}, "glm5_2"),
        ({"granularity": "block"}, "tensor"),
        ({"quant_dtype": "mxfp"}, "int"),
        ({"bit_policy": BitPolicy()}, "W8A8"),
        ({"algos": ["lwc"]}, "algos"),
        ({"attn_linear_param_dir": "/ptq"}, "attn_linear_param_dir"),
        ({"attn_cache_param_dir": "/ptq"}, "attn_cache_param_dir"),
        ({"moe_mlp_param_dir": "/ptq"}, "moe_mlp_param_dir"),
    ],
)
def test_reject_incompatible_args(override, message):
    with pytest.raises(ValueError, match=message):
        validate_ascend_deploy_args(_args(**override))
