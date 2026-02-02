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
import shutil
import json

import torch

import amct_pytorch.amct_pytorch_inner.amct_pytorch
from amct_pytorch.amct_pytorch_inner.amct_pytorch.utils.model_util import ModuleHelper

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

class TestQuantCalibrationInterface(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print('TestQuantCalibrationInterface start!')
        cls.temp_dir = os.path.join(CUR_DIR, 'temp')
        if not os.path.exists(cls.temp_dir):
            os.mkdir(cls.temp_dir)

    @classmethod
    def tearDownClass(cls):
        print('TestQuantCalibrationInterface end!')
        if os.path.exists(cls.temp_dir):
            shutil.rmtree(cls.temp_dir)

    def test_create_default_config(self):
        config_file = os.path.join(self.temp_dir, 'config.json')
        model = CustomizedModel()
        quant_layers = {'kv_cache_quant_layers': ['matmul1', 'matmul2']}
        amct_pytorch.amct_pytorch_inner.amct_pytorch.create_quant_cali_config(config_file, model, quant_layers)

        with open(config_file) as f:
            config = json.load(f)
            self.assertIn('matmul1', config)
            self.assertIn('matmul2', config)

    def test_create_config_from_proto(self):
        config_file = os.path.join(self.temp_dir, 'config.json')
        model = CustomizedModel()
        quant_layers = {'kv_cache_quant_layers': ['matmul1', 'matmul2']}
        config_proto = os.path.join(CUR_DIR, 'utils/test_case_config_00.cfg')
        amct_pytorch.amct_pytorch_inner.amct_pytorch.create_quant_cali_config(config_file, model, quant_layers, config_proto)

        with open(config_file) as f:
            config = json.load(f)
            self.assertIn('matmul1', config)
            self.assertIn('matmul2', config)
            self.assertIn('matmul3', config)

            self.assertEqual('hfmg', config.get('matmul1').get('kv_data_quant_config').get('act_algo'))
            self.assertEqual('ifmr', config.get('matmul3').get('kv_data_quant_config').get('act_algo'))

    def test_create_quant_cali_model(self):
        config_file = os.path.join(self.temp_dir, 'config.json')
        model = CustomizedModel()
        quant_layers = {'kv_cache_quant_layers': ['matmul1', 'matmul2']}
        config_proto = os.path.join(CUR_DIR, 'utils/test_case_config_00.cfg')
        record_file = os.path.join(self.temp_dir, 'record.txt')
        amct_pytorch.amct_pytorch_inner.amct_pytorch.create_quant_cali_config(config_file, model, quant_layers, config_proto)
        amct_pytorch.amct_pytorch_inner.amct_pytorch.create_quant_cali_model(config_file, record_file, model)
        self.assertRaises(RuntimeError, ModuleHelper(model).check_amct_op)