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
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------
import torch
from torch import nn

from amct_pytorch.common.utils.model_util import ModuleHelper


def _build_model():
    return nn.Sequential(
        nn.Linear(4, 4),
        nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
        ),
    )


def test_named_module_dict_collects_all_submodules():
    model = _build_model()
    helper = ModuleHelper(model)

    expected = {name for name, _ in model.named_modules()}
    assert set(helper.named_module_dict) == expected


def test_named_module_dict_holds_module_references():
    model = _build_model()
    helper = ModuleHelper(model)
    assert helper.named_module_dict["0"] is model[0]
    assert helper.named_module_dict["1.0"] is model[1][0]


def test_replace_module_by_name_top_level():
    model = nn.Sequential(nn.Linear(4, 4))
    new = nn.Linear(4, 8)
    ModuleHelper.replace_module_by_name(model, "0", new)
    assert model[0] is new


def test_replace_module_by_name_nested():
    model = _build_model()
    new = nn.Linear(4, 16)
    ModuleHelper.replace_module_by_name(model, "1.0", new)
    assert model[1][0] is new
    # Forward path should reflect the new shape.
    out = model(torch.randn(1, 4))
    assert out.shape == (1, 16)
