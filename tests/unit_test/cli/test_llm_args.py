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


# Minimal argv per sub-command to satisfy required arguments.
REQUIRED_ARGV = {
    "ptq": ["--model_name", "qwen3_5", "--quant_target", "mlp", "--data_dir", "/tmp/d"],
    "eval": ["--model_name", "qwen3_5"],
    "extract_ptq_data": [
        "--model_name",
        "qwen3_5",
        "--quant_target",
        "mlp",
        "--data_dir",
        "/tmp/d",
    ],
    "deploy": ["--model_name", "qwen3_5", "--quant_dtype", "int"],
    None: ["--model_name", "qwen3_5"],
}


@pytest.mark.parametrize("command", ["ptq", "eval", "extract_ptq_data", "deploy", None])
def test_granularity_defaults_to_block(command):
    with patch.object(sys, "argv", ["amct"] + REQUIRED_ARGV[command]):
        args = parser_gen(command=command)

    assert args.granularity == "block"


@pytest.mark.parametrize("command", ["ptq", "extract_ptq_data", "deploy"])
def test_granularity_rejects_model_for_non_eval_commands(command):
    with (
        patch.object(sys, "argv", ["amct", "--granularity", "model"]),
        pytest.raises(SystemExit) as exc_info,
    ):
        parser_gen(command=command)

    assert exc_info.value.code == 2


def test_granularity_accepts_model_for_eval():
    argv = ["amct", "--model_name", "qwen3_5", "--granularity", "model"]
    with patch.object(sys, "argv", argv):
        args = parser_gen(command="eval")

    assert args.granularity == "model"


def test_granularity_accepts_tensor_for_deploy():
    argv = [
        "amct",
        "--model_name",
        "qwen3_5",
        "--quant_dtype",
        "int",
        "--granularity",
        "tensor",
    ]
    with patch.object(sys, "argv", argv):
        args = parser_gen(command="deploy")

    assert args.granularity == "tensor"


@pytest.mark.parametrize("command", ["ptq", "eval", "extract_ptq_data", "deploy", None])
def test_model_name_required_for_all_commands(command):
    with (
        patch.object(sys, "argv", ["amct"]),
        pytest.raises(SystemExit) as exc_info,
    ):
        parser_gen(command=command)

    assert exc_info.value.code == 2


@pytest.mark.parametrize("command", ["ptq", "extract_ptq_data"])
def test_quant_target_required_for_ptq_commands(command):
    argv = ["amct", "--model_name", "qwen3_5", "--data_dir", "/tmp/some_dir"]
    with (
        patch.object(sys, "argv", argv),
        pytest.raises(SystemExit) as exc_info,
    ):
        parser_gen(command=command)

    assert exc_info.value.code == 2


@pytest.mark.parametrize("command", ["ptq", "extract_ptq_data"])
def test_data_dir_required_for_ptq_commands(command):
    argv = ["amct", "--model_name", "qwen3_5", "--quant_target", "mlp"]
    with (
        patch.object(sys, "argv", argv),
        pytest.raises(SystemExit) as exc_info,
    ):
        parser_gen(command=command)

    assert exc_info.value.code == 2


def test_quant_dtype_required_for_deploy():
    argv = ["amct", "--model_name", "qwen3_5"]
    with (
        patch.object(sys, "argv", argv),
        pytest.raises(SystemExit) as exc_info,
    ):
        parser_gen(command="deploy")

    assert exc_info.value.code == 2


def test_optional_args_stay_optional_for_eval():
    argv = ["amct", "--model_name", "qwen3_5"]
    with patch.object(sys, "argv", argv):
        args = parser_gen(command="eval")

    assert args.quant_target == []
    assert args.data_dir == ""


def test_command_specific_args_not_enforced_without_command():
    # Without a known command, only the always-required --model_name applies;
    # the command-specific required arguments stay optional.
    argv = ["amct", "--model_name", "qwen3_5"]
    with patch.object(sys, "argv", argv):
        args = parser_gen()

    assert args.quant_target == []
    assert args.data_dir == ""
    assert args.quant_dtype == ""
