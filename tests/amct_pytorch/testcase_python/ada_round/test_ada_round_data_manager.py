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
import torch

from amct_pytorch.amct_pytorch_inner.amct_pytorch.ada_round.ada_round_data_manager import AdaRoundDataManager


class TestModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 4)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.linear(x)
        x = self.relu(x)
        return x


class TestAdaRoundDataManager(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = TestModel()
        cls.input_data = torch.randn(4, 3)
        cls.data_manager = AdaRoundDataManager(cls.model)

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_get_input_data(self):
        data = self.data_manager.get_input_data(self.input_data, 'linear')
        self.assertTrue((data == self.input_data).all())

    def test_get_output_data(self):
        data = self.data_manager.get_output_data(self.input_data, 'linear')
        except_data = self.model.linear(self.input_data)
        self.assertTrue((data == except_data).all())