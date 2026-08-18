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

import importlib.util
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).parents[6]
MODULE_PATH = (
    REPO_ROOT
    / "amct_pytorch"
    / "common"
    / "models"
    / "llm"
    / "common"
    / "weight_path.py"
)


def _load_weight_path_module():
    if not MODULE_PATH.exists():
        raise AssertionError("weight_path.py has not been implemented")
    spec = importlib.util.spec_from_file_location("weight_path", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestResolveSafetensorsPath(unittest.TestCase):
    def test_rejects_unsafe_weight_file_names(self):
        module = _load_weight_path_module()
        unsafe_names = (
            "",
            "../outside.safetensors",
            "/outside.safetensors",
            r"C:\outside.safetensors",
            "C:outside.safetensors",
            "subdir/shard.safetensors",
            r"subdir\shard.safetensors",
            "config.json",
        )

        with tempfile.TemporaryDirectory() as model_dir:
            for file_name in unsafe_names:
                with self.subTest(file_name=file_name):
                    with self.assertRaises(ValueError):
                        module.resolve_safetensors_path(model_dir, file_name)

    def test_accepts_existing_safetensors_file_in_model_directory(self):
        module = _load_weight_path_module()

        with tempfile.TemporaryDirectory() as model_dir:
            weight_path = Path(model_dir) / "model-00001.safetensors"
            weight_path.touch()

            resolved = module.resolve_safetensors_path(
                model_dir, "model-00001.safetensors"
            )

            self.assertEqual(resolved, weight_path.resolve())


if __name__ == "__main__":
    unittest.main()
