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
import unittest
from unittest import mock
from amct_pytorch.amct_pytorch_inner.amct_pytorch.custom_op import utils


DEVICE = torch.device('cpu')

class CustomizedModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = torch.nn.Conv2d(3, 8, 3, bias=False)
        self.conv_2 = torch.nn.Conv2d(8, 8, 3)
        self.matmul_1 = torch.nn.Linear(8, 8)
        self.matmul_2 = torch.nn.Linear(8, 4)
        self.matmul_3 = torch.nn.Linear(4, 4)
        self.relu_0 = torch.nn.ReLU()

    def forward(self, inputs):
        y = self.conv_1(inputs)
        y = self.conv_2(y)
        y = y.transpose(1,3)
        y = self.matmul_1(y)
        y = self.matmul_2(y)
        y = self.matmul_3(y)
        y = self.relu_0(y)
        return y

class TestUtils(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pass

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_check_quant_data_float32(self):
        input_data = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32)
        utils.check_quant_data(input_data, "weight")

    # def test_check_quant_data_float16(self):
    #     input_data = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float16, device='cuda:0')
    #     utils.check_quant_data(input_data, "weight")

    def test_check_quant_data_float32(self):
        input_data = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float64)
        self.assertRaises(TypeError, utils.check_quant_data, input_data, "weight")

    def test_check_module_device(self):
        model = CustomizedModel().to('meta')
        self.assertRaises(RuntimeError, utils.check_module_device, model, ['matmul_1'])


if __name__ == "__main__":
    unittest.main()