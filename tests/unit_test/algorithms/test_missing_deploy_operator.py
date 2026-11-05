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
"""CPU-only checks for absent deployment operators during conversion."""

import importlib.util
from pathlib import Path
import sys
import types

import pytest


class Quantized:
    pass


class Deploy:
    def __init__(self, source):
        self.source = source


@pytest.fixture
def conversion_pass(monkeypatch):
    registry = types.SimpleNamespace(quant_to_deploy={})
    logger = types.SimpleNamespace(logd=lambda *args: None)

    class Helper:
        @staticmethod
        def replace_module_by_name(model, name, replacement):
            model.replacement = (name, replacement)

    modules = {
        "amct_pytorch.classic.optimizer.base_module_fusion_pass": {
            "BaseModuleFusionPass": object
        },
        "amct_pytorch.common.utils.log": {"LOGGER": logger},
        "amct_pytorch.common.utils.model_util": {"ModuleHelper": Helper},
        "amct_pytorch.classic.deploy_op": {
            "NpuQuantizationLinear": type("DynamicDeploy", (Deploy,), {}),
            "NpuWeightQuantizedLinear": type("WeightDeploy", (Deploy,), {}),
            "NpuQuantizationConv2d": type("ConvDeploy", (Deploy,), {}),
        },
        "amct_pytorch.algorithms": {"AlgorithmRegistry": registry},
    }
    for name, attrs in modules.items():
        module = types.ModuleType(name)
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, module)
    path = (
        Path(__file__).resolve().parents[3]
        / "amct_pytorch/classic/optimizer/replace_npu_quant_pass.py"
    )
    spec = importlib.util.spec_from_file_location("_missing_deploy_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.ReplaceNpuQuantModulePass(), registry


@pytest.mark.parametrize("registered", [None, [], [None]])
def test_missing_deploy_operator_has_actionable_error(conversion_pass, registered):
    processor, registry = conversion_pass
    if registered is not None:
        registry.quant_to_deploy[Quantized] = registered
    with pytest.raises(RuntimeError, match="deploy_op.*Quantized"):
        processor.do_pass(types.SimpleNamespace(), Quantized(), "layer")


@pytest.mark.parametrize("registered", [[None, Deploy], [Deploy, None], [Deploy]])
def test_valid_operator_is_used_when_placeholder_is_present(
    conversion_pass, registered
):
    processor, registry = conversion_pass
    registry.quant_to_deploy[Quantized] = registered
    model, quantized = types.SimpleNamespace(), Quantized()
    processor.do_pass(model, quantized, "layer")
    assert model.replacement[0] == "layer"
    assert isinstance(model.replacement[1], Deploy)
    assert model.replacement[1].source is quantized


def test_name_based_registration_uses_same_lookup_for_selection(conversion_pass):
    processor, registry = conversion_pass
    registry.quant_to_deploy["Quantized"] = [Deploy]
    model, quantized = types.SimpleNamespace(), Quantized()
    processor.do_pass(model, quantized, "layer")
    assert isinstance(model.replacement[1], Deploy)


def test_scalar_registration_is_used(conversion_pass):
    processor, registry = conversion_pass
    registry.quant_to_deploy[Quantized] = Deploy
    model, quantized = types.SimpleNamespace(), Quantized()
    processor.do_pass(model, quantized, "layer")
    assert isinstance(model.replacement[1], Deploy)
    assert model.replacement[1].source is quantized


@pytest.mark.parametrize("registered", [None, [], [None]])
def test_selection_rejects_missing_operator(conversion_pass, registered):
    processor, registry = conversion_pass
    if registered is not None:
        registry.quant_to_deploy[Quantized] = registered
    with pytest.raises(RuntimeError, match="deploy_op.*Quantized"):
        processor._get_deploy_module(Quantized(), Quantized)


@pytest.mark.parametrize("registration_key", [Quantized, "Quantized"])
@pytest.mark.parametrize(
    ("attributes", "operator_name"),
    [
        ({}, "NpuWeightQuantizedLinear"),
        ({"dynamic": True}, "NpuQuantizationLinear"),
        ({"scale_d": 1}, "NpuQuantizationLinear"),
        ({"ori_module_type": "Conv2d"}, "NpuQuantizationConv2d"),
    ],
)
def test_matching_registered_operator_wins_over_fallback(
    conversion_pass, registration_key, attributes, operator_name
):
    processor, registry = conversion_pass
    deployment = sys.modules["amct_pytorch.classic.deploy_op"]
    expected = getattr(deployment, operator_name)
    registry.quant_to_deploy[registration_key] = [None, Deploy, expected]
    model, quantized = types.SimpleNamespace(), Quantized()
    for name, value in attributes.items():
        setattr(quantized, name, value)

    processor.do_pass(model, quantized, "layer")

    assert model.replacement[0] == "layer"
    assert type(model.replacement[1]) is expected
    assert model.replacement[1].source is quantized
