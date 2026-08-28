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
import unittest

import numpy as np
import torch

from amct_pytorch.classic.graph_based.amct_pytorch.custom_op.fake_quant import (
    FakeQuantizedLinear,
)
from amct_pytorch.classic.graph_based.amct_pytorch.optimizer.insert_fakequant_linear_pass import (
    InsertFakequantLinearPass,
)
from amct_pytorch.classic.graph_based.amct_pytorch.optimizer.weight_fakequant_module_pass import (
    WeightFakequantModulePass,
)


class LinearModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 4, bias=False)
        self.linear.weight.data = torch.tensor(
            [[1, 2], [3, 4], [5, 6], [7, 8]], dtype=torch.float32
        )

    def forward(self, inputs):
        return self.linear(inputs)


class TestWeightFakequantModulePass(unittest.TestCase):
    @staticmethod
    def _records():
        return {
            'linear': {
                'data_scale': np.array([1], dtype=np.float32),
                'data_offset': np.array([0], dtype=np.int8),
                'weight_scale': np.ones(4, dtype=np.float32),
                'weight_offset': np.array([1, 2, 3, 4], dtype=np.int8),
            }
        }

    def test_insert_linear_preserves_per_channel_setting(self):
        model = LinearModel()
        records = self._records()

        InsertFakequantLinearPass(records, 8).do_pass(model, model.linear, 'linear')

        self.assertIsInstance(model.linear, FakeQuantizedLinear)
        self.assertTrue(records['linear']['channel_wise'])

    def test_linear_offset_broadcasts_on_output_channel(self):
        model = LinearModel()
        records = self._records()
        original_weight = model.linear.weight.detach().clone()
        fake_linear = FakeQuantizedLinear(model.linear, records['linear'], 'linear', 8)

        WeightFakequantModulePass(records, 8).do_pass(model, fake_linear, 'linear')

        torch.testing.assert_close(fake_linear.sub_module.weight, original_weight)

    def test_linear_offset_broadcasts_for_multidimensional_weight(self):
        model = LinearModel()
        model.linear.weight = torch.nn.Parameter(
            torch.arange(24, dtype=torch.float32).reshape(4, 2, 3)
        )
        records = self._records()
        original_weight = model.linear.weight.detach().clone()
        fake_linear = FakeQuantizedLinear(model.linear, records['linear'], 'linear', 8)

        WeightFakequantModulePass(records, 8).do_pass(model, fake_linear, 'linear')

        torch.testing.assert_close(fake_linear.sub_module.weight, original_weight)


if __name__ == '__main__':
    unittest.main()
