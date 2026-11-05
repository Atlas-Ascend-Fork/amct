# -*- coding: UTF-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""CPU-only checks for deployment registration and fallback order."""

import ast
import importlib.util
from pathlib import Path
import sys
import types

import pytest


@pytest.fixture
def registry(monkeypatch):
    logger = types.ModuleType("amct_pytorch.common.utils.log")
    logger.LOGGER = types.SimpleNamespace(logd=lambda *args: None)
    monkeypatch.setitem(sys.modules, logger.__name__, logger)
    source = (
        Path(__file__).resolve().parents[3] / "amct_pytorch/algorithms/register_algo.py"
    )
    spec = importlib.util.spec_from_file_location("_deploy_registry_test", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Algorithm()


class StableHash(type):
    def __hash__(cls):
        return cls.hash_value


class FirstDeploy(metaclass=StableHash):
    hash_value = 1

    def __init__(self, source):
        self.source = source


class SecondDeploy(metaclass=StableHash):
    hash_value = 0

    def __init__(self, source):
        self.source = source


class ThirdDeploy(metaclass=StableHash):
    hash_value = 2


class Quantized:
    pass


def test_registration_preserves_caller_list_and_priority(registry):
    supplied = [FirstDeploy, SecondDeploy]
    registry.register("example", "Linear", Quantized, supplied)
    assert supplied == [FirstDeploy, SecondDeploy]
    assert registry.quant_to_deploy[Quantized] == [FirstDeploy, SecondDeploy]
    registry.register("example", "Linear", Quantized, FirstDeploy)
    assert registry.quant_to_deploy[Quantized] == [FirstDeploy, SecondDeploy]
    additional = [SecondDeploy, ThirdDeploy]
    registry.register("example", "Linear", Quantized, additional)
    assert additional == [SecondDeploy, ThirdDeploy]
    assert registry.quant_to_deploy[Quantized] == [
        FirstDeploy,
        SecondDeploy,
        ThirdDeploy,
    ]


def test_fallback_uses_first_registered_deploy_operator(registry):
    registry.register("example", "Linear", Quantized, [FirstDeploy, SecondDeploy])
    path = (
        Path(__file__).resolve().parents[3]
        / "amct_pytorch/classic/optimizer/replace_npu_quant_pass.py"
    )
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "ReplaceNpuQuantModulePass"
    )
    namespace = {"AlgorithmRegistry": registry, "BaseModuleFusionPass": object}
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])),
            str(path),
            "exec",
        ),
        namespace,
    )
    selector = namespace["ReplaceNpuQuantModulePass"]()
    selector._should_use_deploy_op = lambda obj, deploy: False
    instance = Quantized()
    result = selector._get_deploy_module(instance, Quantized)
    assert isinstance(result, FirstDeploy)
    assert result.source is instance
