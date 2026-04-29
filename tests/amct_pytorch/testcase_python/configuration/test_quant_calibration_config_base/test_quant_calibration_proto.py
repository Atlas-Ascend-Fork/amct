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
import sys
import unittest
import torch

from amct_pytorch.graph_based_compression.amct_pytorch.configuration.quant_calibration_config_base.quant_calibration_proto import QuantCalibrationProtoConfig

CUR_DIR = os.path.split(os.path.realpath(__file__))[0]

class CustomizedModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.matmul1 = torch.nn.Linear(8, 32)
        self.matmul2 = torch.nn.Linear(8, 4)
        self.matmul3 = torch.nn.Linear(8, 4)

    def forward(self, inputs):
        y = self.matmul1(inputs)
        y = self.matmul2(inputs)
        y = self.matmul3(inputs)
        return y

class TestQuantCalibrationProtoConfig(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print('TestQuantCalibrationProtoConfig start!')
        cls.proto_path = os.path.join(CUR_DIR, 'utils/test_case_config_00.cfg')
        cls.obj = QuantCalibrationProtoConfig(cls.proto_path, CustomizedModel())

    @classmethod
    def tearDownClass(cls):
        print('TestQuantCalibrationProtoConfig end!')

    def test_get_proto_global_config(self):
        ret = self.obj.get_proto_global_config()
        self.assertEqual(ret.get('batch_num'), 2)
        self.assertEqual(ret.get('activation_offset'), True)

    def test_get_quant_config(self):
        ret = self.obj.get_quant_config('kv_cache_quant_config')
        self.assertEqual(ret.get('act_algo'), 'hfmg')

    def test_get_quant_layers(self):
        ret = self.obj.get_quant_layers('kv_quant')
        self.assertIn('matmul1', ret)

    def test_get_override_layers(self):
        ret = self.obj.get_override_layers()
        self.assertIn('matmul3', ret.get('kv_cache_quant_layers'))

    def test_read_override_layer_config(self):
        ret = self.obj.read_override_layer_config('matmul3', 'kv_data_quant_config')
        self.assertIn('act_algo', ret)
        self.assertIn('quant_granularity', ret)

    def test_get_batch_num(self):
        ret = self.obj._get_batch_num()
        self.assertEqual(ret, 2)

    def test_get_activation_offset(self):
        ret = self.obj._get_activation_offset()
        self.assertEqual(ret, True)
