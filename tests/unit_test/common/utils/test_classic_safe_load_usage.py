#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------
import importlib.util
import sys
import types
from pathlib import Path

import torch

LINEAR_IN_FEATURES = 2
LINEAR_OUT_FEATURES = 1
LAYER_NAME = "linear"
MODEL_STATE_DICT_NAME = "state_dict"
PARAM_SCALE_VALUE = 1.0
REPO_ROOT = Path(__file__).resolve().parents[4]
SAFE_LOAD_PATH = REPO_ROOT / "amct_pytorch/common/utils/safe_load.py"
MODULE_RECORD_PARSER_PATH = (
    REPO_ROOT / "amct_pytorch/classic/graph_based/amct_pytorch/parser/module_based_record_parser.py"
)
MODEL_UTIL_PATH = REPO_ROOT / "amct_pytorch/classic/graph_based/amct_pytorch/utils/model_util.py"


def _install_package(monkeypatch, module_name):
    module = types.ModuleType(module_name)
    module.__path__ = []
    monkeypatch.setitem(sys.modules, module_name, module)
    return module


def _install_module(monkeypatch, module_name, **attrs):
    module = types.ModuleType(module_name)
    for attr_name, attr_value in attrs.items():
        setattr(module, attr_name, attr_value)
    monkeypatch.setitem(sys.modules, module_name, module)
    return module


def _load_module(monkeypatch, module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module


def _prepare_safe_load_module(monkeypatch):
    _install_package(monkeypatch, "amct_pytorch")
    _install_package(monkeypatch, "amct_pytorch.common")
    _install_package(monkeypatch, "amct_pytorch.common.utils")
    return _load_module(
        monkeypatch,
        "amct_pytorch.common.utils.safe_load",
        SAFE_LOAD_PATH,
    )


def _prepare_module_based_record_parser(monkeypatch):
    _prepare_safe_load_module(monkeypatch)
    _install_package(monkeypatch, "amct_pytorch.classic")
    _install_package(monkeypatch, "amct_pytorch.classic.graph_based")
    _install_package(monkeypatch, "amct_pytorch.classic.graph_based.amct_pytorch")
    _install_package(monkeypatch, "amct_pytorch.classic.graph_based.amct_pytorch.parser")
    _install_package(monkeypatch, "amct_pytorch.classic.graph_based.amct_pytorch.common")
    _install_package(monkeypatch, "amct_pytorch.classic.graph_based.amct_pytorch.common.utils")
    _install_module(
        monkeypatch,
        "amct_pytorch.classic.graph_based.amct_pytorch.common.utils.util",
        version_higher_than=lambda *_: True,
    )
    return _load_module(
        monkeypatch,
        "amct_pytorch.classic.graph_based.amct_pytorch.parser.module_based_record_parser",
        MODULE_RECORD_PARSER_PATH,
    )


def _prepare_model_util(monkeypatch):
    _prepare_safe_load_module(monkeypatch)
    _install_package(monkeypatch, "amct_pytorch.classic")
    _install_package(monkeypatch, "amct_pytorch.classic.graph_based")
    _install_package(monkeypatch, "amct_pytorch.classic.graph_based.amct_pytorch")
    _install_package(monkeypatch, "amct_pytorch.classic.graph_based.amct_pytorch.configuration")
    _install_package(monkeypatch, "amct_pytorch.classic.graph_based.amct_pytorch.common")
    _install_package(monkeypatch, "amct_pytorch.classic.graph_based.amct_pytorch.common.utils")
    _install_package(monkeypatch, "amct_pytorch.classic.graph_based.amct_pytorch.utils")
    _install_module(
        monkeypatch,
        "amct_pytorch.classic.graph_based.amct_pytorch.configuration.configuration",
        Configuration=type("Configuration", (), {"get_quant_config": lambda self: {}}),
    )
    _install_module(
        monkeypatch,
        "amct_pytorch.classic.graph_based.amct_pytorch.utils.vars",
        AMCT_OPERATIONS=set(),
        AMCT_DISTILL_OPERATIONS=set(),
        AMCT_RETRAIN_OPERATIONS=set(),
    )
    return _load_module(
        monkeypatch,
        "amct_pytorch.classic.graph_based.amct_pytorch.utils.model_util",
        MODEL_UTIL_PATH,
    )


def test_get_layer_quant_params_uses_safe_load_with_mmap(monkeypatch, tmp_path):
    module_based_record_parser = _prepare_module_based_record_parser(monkeypatch)
    quant_result = tmp_path / "quant_result.pth"
    quant_result.touch()
    captured = {}

    def fake_safe_torch_load(file_path, **kwargs):
        captured["file_path"] = file_path
        captured["kwargs"] = kwargs
        return {LAYER_NAME: {"scale": torch.tensor([PARAM_SCALE_VALUE])}}

    monkeypatch.setattr(module_based_record_parser, "safe_torch_load", fake_safe_torch_load)
    monkeypatch.setattr(module_based_record_parser, "version_higher_than", lambda *_: True)

    quant_params = module_based_record_parser.get_layer_quant_params(
        {"quant_result_path": str(quant_result)}, LAYER_NAME
    )

    assert torch.equal(quant_params["scale"], torch.tensor([PARAM_SCALE_VALUE]))
    assert captured == {
        "file_path": str(quant_result),
        "kwargs": {"mmap": True},
    }


def test_load_pth_file_uses_safe_load_for_deserialization(monkeypatch, tmp_path):
    model_util = _prepare_model_util(monkeypatch)
    model = torch.nn.Linear(LINEAR_IN_FEATURES, LINEAR_OUT_FEATURES)
    checkpoint = {
        MODEL_STATE_DICT_NAME: {
            key: torch.ones_like(value)
            for key, value in model.state_dict().items()
        }
    }
    captured = {}

    def fake_safe_torch_load(file_path, **kwargs):
        captured["file_path"] = file_path
        captured["kwargs"] = kwargs
        return checkpoint

    monkeypatch.setattr(model_util, "safe_torch_load", fake_safe_torch_load)

    model_util.load_pth_file(model, str(tmp_path / "model.pth"), MODEL_STATE_DICT_NAME)

    # weights_only is handled inside safe_torch_load, so the caller only passes
    # map_location and lets the safe wrapper apply the secure default.
    assert captured == {
        "file_path": str(tmp_path / "model.pth"),
        "kwargs": {"map_location": torch.device("cpu")},
    }
    assert torch.equal(model.weight, checkpoint[MODEL_STATE_DICT_NAME]["weight"])
    assert torch.equal(model.bias, checkpoint[MODEL_STATE_DICT_NAME]["bias"])
