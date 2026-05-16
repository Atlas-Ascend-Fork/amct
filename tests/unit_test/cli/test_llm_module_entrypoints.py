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

import importlib
import runpy
import sys

import pytest


ENTRYPOINTS = [
    ("amct_pytorch.deploy", "amct_pytorch.cli.llm.deploy"),
    ("amct_pytorch.eval", "amct_pytorch.cli.llm.eval"),
    ("amct_pytorch.extract_ptq_data", "amct_pytorch.cli.llm.extract_ptq_data"),
    ("amct_pytorch.ptq", "amct_pytorch.cli.llm.ptq"),
]


@pytest.mark.parametrize("module_name,target_name", ENTRYPOINTS)
def test_amct_pytorch_module_exports_main(module_name, target_name):
    module = importlib.import_module(module_name)

    assert callable(module.main)


@pytest.mark.parametrize("module_name,target_name", ENTRYPOINTS)
def test_amct_pytorch_module_executes_llm_cli_main(monkeypatch, module_name, target_name):
    calls = []
    target = importlib.import_module(target_name)
    monkeypatch.setattr(target, "main", lambda: calls.append(target_name))
    sys.modules.pop(module_name, None)

    runpy.run_module(module_name, run_name="__main__", alter_sys=True)

    assert calls == [target_name]
