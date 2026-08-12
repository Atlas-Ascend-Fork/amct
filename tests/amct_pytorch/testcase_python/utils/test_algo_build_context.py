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

import importlib.util
from pathlib import Path
import sys
from types import ModuleType
import unittest


MODULE_PATH = (
    Path(__file__).resolve().parents[4]
    / 'amct_pytorch'
    / 'algorithms'
    / 'quant'
    / '__init__.py'
)


def load_quant_module():
    module_name = 'amct_quant_for_test'
    spec = importlib.util.spec_from_file_location(
        module_name,
        MODULE_PATH,
        submodule_search_locations=[str(MODULE_PATH.parent)],
    )
    module = importlib.util.module_from_spec(spec)
    base_module = ModuleType(f'{module_name}.base')
    base_module.QuantAlgorithmBase = object
    sys.modules[base_module.__name__] = base_module
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TestAlgoBuildContext(unittest.TestCase):
    def test_optional_dimensions(self):
        algo_build_context = load_quant_module().AlgoBuildContext
        default_context = algo_build_context()
        sized_context = algo_build_context(matrix_size=128, dim_size=4096)

        self.assertIsNone(default_context.matrix_size)
        self.assertIsNone(default_context.dim_size)
        self.assertEqual(sized_context.matrix_size, 128)
        self.assertEqual(sized_context.dim_size, 4096)


if __name__ == '__main__':
    unittest.main()
