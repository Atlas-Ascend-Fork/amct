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

import importlib
from unittest import mock

import pytest


_MODULE_NAME = (
    "amct_pytorch.classic.graph_based.amct_pytorch.accuracy_based_auto_calibration"
)


def _load_module():
    return importlib.import_module(_MODULE_NAME)


def _build_adapter(module):
    adapter = object.__new__(module.AccuracyBasedAutoCalibration)
    adapter.original_model = object()
    return adapter


def test_get_original_model_returns_deep_copy_result():
    module = _load_module()
    adapter = _build_adapter(module)
    copied_model = object()

    with mock.patch.object(module.ModuleHelper, "deep_copy", return_value=copied_model):
        assert adapter.get_original_model() is copied_model


def test_get_original_model_reraises_runtime_error():
    module = _load_module()
    adapter = _build_adapter(module)

    with mock.patch.object(
        module.ModuleHelper, "deep_copy", side_effect=RuntimeError("deep copy failed")
    ):
        with pytest.raises(RuntimeError, match="deep copy failed"):
            adapter.get_original_model()
