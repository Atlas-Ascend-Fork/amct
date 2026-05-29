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
import torch
from torch import nn

from amct_pytorch.classic.graph_based.amct_pytorch.parser import module_based_record_parser
from amct_pytorch.classic.graph_based.amct_pytorch.utils import model_util

LINEAR_IN_FEATURES = 2
LINEAR_OUT_FEATURES = 1
LAYER_NAME = "linear"
MODEL_STATE_DICT_NAME = "state_dict"
PARAM_SCALE_VALUE = 1.0


def test_get_layer_quant_params_uses_safe_load_with_mmap(monkeypatch, tmp_path):
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


def test_load_pth_file_uses_safe_load_with_weights_only(monkeypatch, tmp_path):
    model = nn.Linear(LINEAR_IN_FEATURES, LINEAR_OUT_FEATURES)
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
    monkeypatch.setattr(model_util, "version_higher_than", lambda *_: True)

    model_util.load_pth_file(model, str(tmp_path / "model.pth"), MODEL_STATE_DICT_NAME)

    assert captured == {
        "file_path": str(tmp_path / "model.pth"),
        "kwargs": {"map_location": torch.device("cpu"), "weights_only": True},
    }
    assert torch.equal(model.weight, checkpoint[MODEL_STATE_DICT_NAME]["weight"])
    assert torch.equal(model.bias, checkpoint[MODEL_STATE_DICT_NAME]["bias"])
