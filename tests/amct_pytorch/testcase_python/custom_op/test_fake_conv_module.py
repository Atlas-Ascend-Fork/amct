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
import os
import unittest
import torch
import numpy as np

from amct_pytorch.graph_based_compression.amct_pytorch.custom_op.fake_quant import FakeQuantizedLinear
from amct_pytorch.graph_based_compression.amct_pytorch.custom_op.fake_quant import FakeQuantizedConvTranspose
from amct_pytorch.graph_based_compression.amct_pytorch.custom_op.fake_quant import FakeQuantizedConv

np.random.seed(0)

CUR_DIR = os.path.split(os.path.realpath(__file__))[0]
DEVICE = torch.device('cpu')

class TestFakeConvModule(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_folder = os.path.join(CUR_DIR, 'test_fake_conv_module')
        if not os.path.isdir(cls.temp_folder):
            os.makedirs(cls.temp_folder)

    @classmethod
    def tearDownClass(cls):
        os.popen('rm -r ' + cls.temp_folder)

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_fake_conv_module(self):
        weight_scale = np.array([0.5, 0.5, 0.5])
        quant_params = {
            "data_scale": 1,
            "data_offset":-128,
            "weight_scale":weight_scale}
        sub_module = torch.nn.Conv2d(3,1,1)
        fake_conv = FakeQuantizedConv(
            sub_module, quant_params, "conv1")
        inputs = np.random.uniform(0, 1, (1,3,32,32))
        out = fake_conv(torch.tensor(inputs).to(torch.float32))
        self.assertIsNotNone(out)


    def test_fake_conv_transpose_module(self):
        weight_scale = np.array([0.5, 0.5, 0.5])
        quant_params = {
            "data_scale": 1,
            "data_offset":-128,
            "weight_scale":weight_scale}
        sub_module = torch.nn.ConvTranspose2d(3,3,3)
        fake_conv = FakeQuantizedConvTranspose(
            sub_module, quant_params, "conv_transpose1")
        inputs = np.random.uniform(0, 1, (1,3,32,32))
        out = fake_conv(torch.tensor(inputs).to(torch.float32))
        self.assertIsNotNone(out)


    def test_fake_linear_module(self):
        weight_scale = np.array([0.5])
        quant_params = {
            "data_scale": 1,
            "data_offset":-128,
            "weight_scale":weight_scale}
        sub_module = torch.nn.Identity()
        fake_linear = FakeQuantizedLinear(
            sub_module, quant_params, "linear1")
        inputs = np.random.uniform(0, 1, (32,256))
        out = fake_linear(torch.tensor(inputs))
        self.assertIsNotNone(out)


    def test_fake_conv3d_module(self):
        weight_scale = np.array([0.5, 0.5, 0.5])
        quant_params = {
            "data_scale": 1,
            "data_offset":-128,
            "weight_scale":weight_scale}
        sub_module = torch.nn.Conv3d(3,1,1)
        fake_conv = FakeQuantizedConv(
            sub_module, quant_params, "conv2")
        inputs = np.random.uniform(0, 1, (1,3,5,32,32))
        out = fake_conv(torch.tensor(inputs).to(torch.float32))
        self.assertIsNotNone(out)

    def test_fake_conv3d_transpose_module(self):
        weight_scale = np.array([0.5, 0.5, 0.5])
        quant_params = {
            "data_scale": 1,
            "data_offset":-128,
            "weight_scale":weight_scale}
        sub_module = torch.nn.ConvTranspose3d(3,3,3)
        fake_conv = FakeQuantizedConvTranspose(
            sub_module, quant_params, "conv_transpose2")
        inputs = np.random.uniform(0, 1, (1,3,5,32,32))
        out = fake_conv(torch.tensor(inputs).to(torch.float32))
        self.assertIsNotNone(out)

    def test_fake_conv1d_module(self):
        weight_scale = np.array([0.5, 0.5, 0.5])
        quant_params = {
            "data_scale": 1,
            "data_offset":-128,
            "weight_scale":weight_scale}
        sub_module = torch.nn.Conv1d(3, 3, 1)
        fake_conv = FakeQuantizedConv(
            sub_module, quant_params, "conv1")
        inputs = np.random.uniform(0, 1, (1,3,32))
        out = fake_conv(torch.tensor(inputs).to(torch.float))
        self.assertTrue(isinstance(out, torch.Tensor))
