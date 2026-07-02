#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
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
"""Unit tests for module_based_record_parser.get_layer_quant_params error branches."""
import unittest
from unittest.mock import patch

from amct_pytorch.classic.graph_based.amct_pytorch.parser import (
    module_based_record_parser as mbrp,
)

_MODULE = (
    "amct_pytorch.classic.graph_based.amct_pytorch.parser.module_based_record_parser"
)


class TestGetLayerQuantParams(unittest.TestCase):
    def test_raises_when_quant_result_path_missing(self):
        with self.assertRaises(RuntimeError):
            mbrp.get_layer_quant_params({}, "layer.0")

    def test_raises_when_path_not_exists(self):
        records = {"quant_result_path": "/no/such/file.pth"}
        with patch(f"{_MODULE}.os.path.exists", return_value=False):
            with self.assertRaises(RuntimeError):
                mbrp.get_layer_quant_params(records, "layer.0")

    def test_raises_when_layer_missing_in_params(self):
        records = {"quant_result_path": "/tmp/fake.pth"}
        with patch(f"{_MODULE}.os.path.exists", return_value=True), patch(
            f"{_MODULE}.safe_torch_load", return_value={"other.layer": {}}
        ):
            with self.assertRaises(RuntimeError):
                mbrp.get_layer_quant_params(records, "layer.0")

    def test_returns_layer_params_on_success(self):
        records = {"quant_result_path": "/tmp/fake.pth"}
        expected = {"scale": 1.0}
        with patch(f"{_MODULE}.os.path.exists", return_value=True), patch(
            f"{_MODULE}.safe_torch_load", return_value={"layer.0": expected}
        ):
            result = mbrp.get_layer_quant_params(records, "layer.0")
        self.assertEqual(result, expected)

    def test_raises_when_load_fails(self):
        records = {"quant_result_path": "/tmp/fake.pth"}
        with patch(f"{_MODULE}.os.path.exists", return_value=True), patch(
            f"{_MODULE}.safe_torch_load", side_effect=ValueError("bad file")
        ):
            with self.assertRaises(RuntimeError):
                mbrp.get_layer_quant_params(records, "layer.0")


if __name__ == "__main__":
    unittest.main()
