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

from amct_pytorch.graph_based_compression.amct_pytorch.configuration.quant_calibration_config_base.quant_calibration_config_base import QuantCalibrationConfigBase

CUR_DIR = os.path.split(os.path.realpath(__file__))[0]

class CustomizedModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.matmul1 = torch.nn.Linear(8, 32)
        self.matmul2 = torch.nn.Linear(8, 4)
        self.matmul3 = torch.nn.Linear(8, 4)
        self.relu0 = torch.nn.ReLU()

    def forward(self, inputs):
        y = self.matmul1(inputs)
        y = self.matmul2(y)
        y = self.matmul3(y)
        y = self.relu(y)
        return y

class TestQuantCalibrationConfigBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print('TestQuantCalibrationConfigBase start!')
        cls.temp_dir = os.path.join(CUR_DIR, 'temp')
        if not os.path.exists(cls.temp_dir):
            os.mkdir(cls.temp_dir)

    @classmethod
    def tearDownClass(cls):
        print('TestQuantCalibrationConfigBase end!')
        if os.path.exists(cls.temp_dir):
            shutil.rmtree(cls.temp_dir)

    def test_del_reduant_config(self):
        config = {'matmul1': {'act_algo': 'ifmr', 'asymmetric': None}}
        QuantCalibrationConfigBase._del_reduant_config(config)
        self.assertIn('act_algo', config.get('matmul1'))
        self.assertNotIn('asymmetric', config.get('matmul1'))

    def test_check_quant_layers(self):
        QuantCalibrationConfigBase.check_quant_layers(CustomizedModel(), {'kv_cache_quant_layers': ('matmul1', 'matmul2')})
        self.assertRaises(ValueError, QuantCalibrationConfigBase.check_quant_layers, CustomizedModel(), {'kv_cache_quant_layers': ('matmul3', 'matmul4')})
        self.assertRaises(ValueError, QuantCalibrationConfigBase.check_quant_layers, CustomizedModel(), {'kv_cache_quant_layers': ('relu0')})
        self.assertRaises(ValueError, QuantCalibrationConfigBase.check_quant_layers, CustomizedModel(), {'kv_cache_quant_layers': ('matmul1', 'matmul1')})

    def test_add_global_to_layer(self):
        config = {'batch_num': 2, 'activation_offset': True, 'matmul1': {'kv_data_quant_config': {'act_algo': 'ifmr', 'search_range': [0.8, 1.2]}}}
        QuantCalibrationConfigBase.add_global_to_layer(config)
        self.assertEqual(2, config.get('matmul1').get('kv_data_quant_config').get('batch_num'))
        self.assertEqual(True, config.get('matmul1').get('kv_data_quant_config').get('with_offset'))
        self.assertEqual(0.8, config.get('matmul1').get('kv_data_quant_config').get('search_range_start'))
        self.assertEqual(1.2, config.get('matmul1').get('kv_data_quant_config').get('search_range_end'))

    def test_get_quant_layer_config(self):
        ret = QuantCalibrationConfigBase.get_quant_layer_config('matmul1', {})
        self.assertIsNone(ret)
        ret = QuantCalibrationConfigBase.get_quant_layer_config('matmul1', {'batch_num': 2, 'activation_offset': True, 'matmul1': {'kv_data_quant_config': {'act_algo': 'ifmr', 'search_range': [0.8, 1.2]}}})
        self.assertIn('kv_data_quant_config', ret)

    def test_create_default_config(self):
        config_file = os.path.join(self.temp_dir, 'config.json')
        model = CustomizedModel()
        quant_layers = {'kv_cache_quant_layers': ['matmul1', 'matmul2']}
        QuantCalibrationConfigBase().create_default_config(config_file, model, quant_layers)

        with open(config_file) as f:
            config = json.load(f)
            self.assertIn('matmul1', config)
            self.assertIn('matmul2', config)
        
    def test_create_default_config_no_quant_layers(self):
        config_file = os.path.join(self.temp_dir, 'config.json')
        model = CustomizedModel()
        quant_layers = {}
        self.assertRaises(RuntimeError, QuantCalibrationConfigBase().create_default_config, config_file, model, quant_layers)

    def test_create_config_from_proto(self):
        config_file = os.path.join(self.temp_dir, 'config.json')
        model = CustomizedModel()
        quant_layers = {'kv_cache_quant_layers': ['matmul1', 'matmul2']}
        config_proto = os.path.join(CUR_DIR, 'utils/test_case_config_00.cfg')
        QuantCalibrationConfigBase().create_config_from_proto(config_file, model, config_proto)

        with open(config_file) as f:
            config = json.load(f)
            self.assertIn('matmul1', config)
            self.assertIn('matmul2', config)
            self.assertIn('matmul3', config)

            self.assertEqual('hfmg', config.get('matmul1').get('kv_data_quant_config').get('act_algo'))
            self.assertEqual('ifmr', config.get('matmul3').get('kv_data_quant_config').get('act_algo'))

    def test_create_config_from_proto_no_quant_layer(self):
        config_file = os.path.join(self.temp_dir, 'config.json')
        model = CustomizedModel()
        quant_layers = {'kv_cache_quant_layers': ['matmul1', 'matmul2']}
        config_proto = os.path.join(CUR_DIR, 'utils/test_case_config_01.cfg')
        self.assertRaises(RuntimeError, QuantCalibrationConfigBase().create_config_from_proto, config_file, model, config_proto)

    def test_parse_quant_config(self):
        config_file = os.path.join(self.temp_dir, 'config.json')
        model = CustomizedModel()
        quant_layers = {'kv_cache_quant_layers': ['matmul1', 'matmul2']}
        config_proto = os.path.join(CUR_DIR, 'utils/test_case_config_00.cfg')
        QuantCalibrationConfigBase().create_config_from_proto(config_file, model, config_proto)
        config = QuantCalibrationConfigBase().parse_quant_config(config_file, model)
        self.assertIn('matmul1', config)
        self.assertIn('matmul2', config)
        self.assertIn('matmul3', config)

        self.assertEqual('hfmg', config.get('matmul1').get('kv_data_quant_config').get('act_algo'))
        self.assertEqual('ifmr', config.get('matmul3').get('kv_data_quant_config').get('act_algo'))