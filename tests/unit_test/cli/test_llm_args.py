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


def test_trust_remote_code_is_required(capsys):
    with patch.object(sys, "argv", ["amct"]), pytest.raises(SystemExit) as exc_info:
        parser_gen()

    assert exc_info.value.code == 2
    error = capsys.readouterr().err
    assert "trust_remote_code is required" in error
    assert "must be True or False" in error
    assert "recommend setting it to True" in error


@pytest.mark.parametrize(
    "command_line_value,expected",
    [("True", True), ("False", False)],
)
def test_trust_remote_code_accepts_boolean_values(command_line_value, expected):
    with patch.object(sys, "argv", ["amct", "--trust_remote_code", command_line_value]):
        if expected:
            args = parser_gen()
        else:
            with pytest.warns(UserWarning):
                args = parser_gen()

    assert args.trust_remote_code is expected


def test_trust_remote_code_false_warns_user():
    with (
        patch.object(sys, "argv", ["amct", "--trust_remote_code", "False"]),
        pytest.warns(
            UserWarning,
            match=(
                "may cause some models to fail to load.*recommend setting it to True"
            ),
        ),
    ):
        parser_gen()


def test_trust_remote_code_rejects_non_boolean_value(capsys):
    with (
        patch.object(sys, "argv", ["amct", "--trust_remote_code", "yes"]),
        pytest.raises(SystemExit) as exc_info,
    ):
        parser_gen()

    assert exc_info.value.code == 2
    error = capsys.readouterr().err
    assert "invalid choice: 'yes'" in error
    assert "True" in error
    assert "False" in error
