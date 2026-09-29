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
    with patch.object(sys, "argv", ["amct", "--model_name", "qwen3_5"]):
        args = parser_gen()

    assert args.trust_remote_code is False


def test_trust_remote_code_flag_sets_true():
    with patch.object(
        sys, "argv", ["amct", "--model_name", "qwen3_5", "--trust_remote_code"]
    ):
        args = parser_gen()

    assert args.trust_remote_code is True


@pytest.mark.parametrize("value", ["True", "False", "yes"])
def test_trust_remote_code_rejects_explicit_value(value):
    with (
        patch.object(
            sys,
            "argv",
            ["amct", "--model_name", "qwen3_5", "--trust_remote_code", value],
        ),
        pytest.raises(SystemExit) as exc_info,
    ):
        parser_gen()

    assert exc_info.value.code == 2


# Minimal argv per sub-command to satisfy parser-level requirements
# (--model_name is the only always-required argument).
REQUIRED_ARGV = {
    "ptq": ["--model_name", "qwen3_5"],
    "eval": ["--model_name", "qwen3_5"],
    "extract_ptq_data": ["--model_name", "qwen3_5"],
    "deploy": ["--model_name", "qwen3_5"],
    None: ["--model_name", "qwen3_5"],
}


@pytest.mark.parametrize("command", ["ptq", "eval", "extract_ptq_data", "deploy", None])
def test_granularity_defaults_to_block(command):
    with patch.object(sys, "argv", ["amct"] + REQUIRED_ARGV[command]):
        args = parser_gen(command=command)

    assert args.granularity == "block"


def _argv_without_model_name(command):
    # REQUIRED_ARGV minus the --model_name pair, so model_name is the only
    # missing required argument in the model_name test below.
    argv = list(REQUIRED_ARGV[command])
    if "--model_name" in argv:
        idx = argv.index("--model_name")
        del argv[idx : idx + 2]
    return argv


def test_granularity_accepts_model_for_eval():
    argv = ["amct", "--model_name", "qwen3_5", "--granularity", "model"]
    with patch.object(sys, "argv", argv):
        args = parser_gen(command="eval")

    assert args.granularity == "model"


def test_granularity_accepts_tensor_for_deploy(tmp_path):
    config = tmp_path / "layers.json"
    config.write_text('{"quant_layers": ["model.layers.*.self_attn.o_proj"]}')
    bits = tmp_path / "bits.yaml"
    bits.write_text("w_bits: 8\na_bits: 8\n")
    argv = [
        "amct",
        "--model_name",
        "qwen3_6_moe",
        "--granularity",
        "tensor",
        "--quant_dtype",
        "int",
        "--quant_layers_config",
        str(config),
        "--bit_config",
        str(bits),
    ]
    with patch.object(sys, "argv", argv):
        args = parser_gen(command="deploy")

    assert args.granularity == "tensor"


@pytest.mark.parametrize("command", ["ptq", "eval", "extract_ptq_data", "deploy", None])
def test_model_name_required_for_all_commands(command):
    # Other required arguments are provided, so the failure is solely the
    # missing --model_name.
    argv = ["amct"] + _argv_without_model_name(command)
    with (
        patch.object(sys, "argv", argv),
        pytest.raises(SystemExit) as exc_info,
    ):
        parser_gen(command=command)

    assert exc_info.value.code == 2


def test_deploy_accepts_multiple_quant_targets():
    argv = [
        "amct",
        "--model_name",
        "qwen3_5",
        "--quant_target",
        "mlp",
        "attn-linear",
        "--quant_dtype",
        "int",
    ]
    with patch.object(sys, "argv", argv):
        args = parser_gen(command="deploy")

    assert args.quant_target == ["mlp", "attn-linear"]


def test_legacy_tensor_model_takes_no_quant_args():
    argv = [
        "amct",
        "--model_name",
        "deepseek_v4",
        "--trust_remote_code",
        "--granularity",
        "tensor",
        "--eval_mode",
        "bf16",
    ]
    with patch.object(sys, "argv", argv):
        args = parser_gen(command="deploy")

    assert args.granularity == "tensor"
    assert args.quant_dtype == ""
    assert args.quant_layers_config is None


@pytest.mark.parametrize("model_name", ["glm5_2", "qwen3_6_moe"])
def test_configured_tensor_model_requires_quant_dtype(model_name):
    argv = ["amct", "--model_name", model_name, "--granularity", "tensor"]
    with (
        patch.object(sys, "argv", argv),
        pytest.raises(ValueError, match="quant_dtype int"),
    ):
        parser_gen(command="deploy")


def test_optional_args_stay_optional_for_eval():
    argv = ["amct", "--model_name", "qwen3_5"]
    with patch.object(sys, "argv", argv):
        args = parser_gen(command="eval")

    assert args.quant_target == []
    assert args.data_dir == ""


def test_command_specific_args_not_enforced_without_command():
    # Command-specific values are validated by each workflow's __init__, not
    # by the parser; without a command the parser only enforces --model_name.
    argv = ["amct", "--model_name", "qwen3_5"]
    with patch.object(sys, "argv", argv):
        args = parser_gen()

    assert args.quant_target == []
    assert args.data_dir == ""
    assert args.quant_dtype == ""


def test_deploy_format_defaults_to_legacy(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["amct", *REQUIRED_ARGV["deploy"]])
    args = parser_gen(command="deploy")
    assert args.deploy_format == "legacy"
    assert args.quant_layers_config is None
    assert args.deploy_selection is None


@pytest.mark.parametrize("command", [None, "eval", "ptq", "extract_ptq_data"])
@pytest.mark.parametrize("option", ["--deploy_format", "--quant_layers_config"])
def test_deploy_options_unavailable_elsewhere(monkeypatch, command, option):
    monkeypatch.setattr(
        sys, "argv", ["amct", *REQUIRED_ARGV[command], option, "ascend"]
    )
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
    monkeypatch.setattr(
        sys,
        "argv",
        ["amct", *REQUIRED_ARGV["deploy"], "--quant_layers_config", "layers.json"],
    )
    with pytest.raises(ValueError, match="quant_layers_config"):
        parser_gen(command="deploy")


@pytest.mark.parametrize("deploy_format", ["legacy", "ascend"])
def test_tensor_cli_module_bits_and_json(monkeypatch, tmp_path, deploy_format):
    config = tmp_path / "layers.json"
    config.write_text('{"quant_layers": ["model.layers.*.self_attn.o_proj"]}')
    bits = tmp_path / "bits.yaml"
    bits.write_text("attn-linear:\n  w_bits: 8\n  a_bits: 8\n")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "amct",
            "--granularity",
            "tensor",
            "--quant_dtype",
            "int",
            "--model_name",
            "glm5_2",
            "--deploy_format",
            deploy_format,
            "--quant_layers_config",
            str(config),
            "--bit_config",
            str(bits),
        ],
    )
    args = parser_gen(command="deploy")
    assert args.deploy_selection.quant_layers == ("model.layers.*.self_attn.o_proj",)


@pytest.mark.parametrize(
    "bits_text, message",
    [
        ("moe:\n  w_bits: 4\n  a_bits: 8\n", "W8A8"),
        ("moe:\n  w_bit: 8\n  a_bits: 8\n", "w_bit"),
        ("moe:\n  w_bits: 16\n  a_bits: 16\n", "W8A8"),
    ],
)
def test_tensor_cli_rejects_invalid_bits_before_json_read(
    monkeypatch, tmp_path, bits_text, message
):
    bits = tmp_path / "bits.yaml"
    bits.write_text(bits_text)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "amct",
            "--model_name",
            "qwen3_6_moe",
            "--granularity",
            "tensor",
            "--quant_dtype",
            "int",
            "--quant_layers_config",
            "does-not-exist.json",
            "--bit_config",
            str(bits),
        ],
    )
    with pytest.raises(ValueError, match=message):
        parser_gen(command="deploy")
