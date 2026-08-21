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
