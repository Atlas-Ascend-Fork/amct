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

import sys
from unittest.mock import patch

import pytest

from amct_pytorch.cli.llm.args import parser_gen


def test_trust_remote_code_defaults_to_false():
    with patch.object(sys, "argv", ["amct"]):
        args = parser_gen()

    assert args.trust_remote_code is False


def test_trust_remote_code_flag_sets_true():
    with patch.object(sys, "argv", ["amct", "--trust_remote_code"]):
        args = parser_gen()

    assert args.trust_remote_code is True


@pytest.mark.parametrize("value", ["True", "False", "yes"])
def test_trust_remote_code_rejects_explicit_value(value):
    with (
        patch.object(sys, "argv", ["amct", "--trust_remote_code", value]),
        pytest.raises(SystemExit) as exc_info,
    ):
        parser_gen()

    assert exc_info.value.code == 2


def test_deploy_format_defaults_to_legacy(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["amct"])
    args = parser_gen(command="deploy")
    assert args.deploy_format == "legacy"
    assert args.quant_layers_config is None
    assert args.deploy_selection is None


@pytest.mark.parametrize("command", [None, "eval", "ptq", "extract_ptq_data"])
@pytest.mark.parametrize("option", ["--deploy_format", "--quant_layers_config"])
def test_deploy_options_unavailable_elsewhere(monkeypatch, command, option):
    monkeypatch.setattr(sys, "argv", ["amct", option, "ascend"])
    with pytest.raises(SystemExit) as exc:
        parser_gen(command=command)
    assert exc.value.code == 2


def test_ascend_cli_loads_selection(monkeypatch, tmp_path):
    config = tmp_path / "layers.json"
    config.write_text('{"quant_layers": ["model.layers.*.self_attn.q_a_proj"]}')
    bits = tmp_path / "bits.yaml"
    bits.write_text("w_bits: 8\na_bits: 8\n")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "amct",
            "--deploy_format",
            "ascend",
            "--model_name",
            "glm5_2",
            "--granularity",
            "tensor",
            "--quant_dtype",
            "int",
            "--quant_layers_config",
            str(config),
            "--bit_config",
            str(bits),
        ],
    )
    args = parser_gen(command="deploy")
    assert args.deploy_selection.quant_layers == ("model.layers.*.self_attn.q_a_proj",)
    assert args.deploy_selection.ignore_layers == ()


def test_legacy_rejects_selection_instead_of_ignoring_it(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["amct", "--quant_layers_config", "layers.json"])
    with pytest.raises(ValueError, match="quant_layers_config"):
        parser_gen(command="deploy")
