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
from unittest.mock import patch
import torch

from .utils import models

from amct_pytorch.amct_pytorch_inner.amct_pytorch.custom_op.recorder.recorder import Recorder
from amct_pytorch.amct_pytorch_inner.amct_pytorch.optimizer.insert_kv_cache_quant_pass import InsertKVCacheQuantPass

CUR_DIR = os.path.split(os.path.realpath(__file__))[0]

class TestInsertKVCacheQuantPass(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_folder = os.path.join(CUR_DIR, 'test_insert_kv_cache_quant_pass')
        if not os.path.isdir(cls.temp_folder):
            os.makedirs(cls.temp_folder)
        # recorder
        cls.record_file = os.path.join(cls.temp_folder, 'test_insert_kv_cache_quant_pass.txt')
        cls.record_module = Recorder(cls.record_file)

        cls.model = models.Conv2dLinear().to(torch.device("cpu"))

        cls.insert_kv_cache_quant_pass = InsertKVCacheQuantPass(cls.record_module, {})

    @classmethod
    def tearDownClass(cls):
        os.popen('rm -r ' + cls.temp_folder)

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_match_pattern_not_linear(self):
        module = torch.nn.Conv2d(2, 4, kernel_size=2)
        ret = self.insert_kv_cache_quant_pass.match_pattern(module, '')
        self.assertFalse(ret)

    def test_match_pattern_not_quant(self):
        module = torch.nn.Linear(4, 8)
        with patch('amct_pytorch.amct_pytorch_inner.amct_pytorch.configuration.quant_calibration_config.get_kv_cache_quant_layers', return_value=['linear1']):
            ret = self.insert_kv_cache_quant_pass.match_pattern(module, 'not_quant_layer')
            self.assertFalse(ret)

    def test_match_pattern_success(self):
        module = torch.nn.Linear(4, 8)
        with patch('amct_pytorch.amct_pytorch_inner.amct_pytorch.configuration.quant_calibration_config.get_kv_cache_quant_layers', return_value=['linear1']):
            insert_kv_cache_quant_pass = InsertKVCacheQuantPass(self.record_module, {})
            ret = insert_kv_cache_quant_pass.match_pattern(module, 'linear1')
            self.assertTrue(ret)

    def test_do_pass_ifmr(self):
        config = {
            'kv_data_quant_config':{
                'act_algo': 'ifmr',
                'num_bits': 8,
                'batch_num': 1,
                'with_offset': False,
                'max_percentile': 0.999999,
                'min_percentile': 0.999999,
                'search_range_start': 0.7,
                'search_range_end': 1.3,
                'search_step': 0.01,
                'quant_granularity': 1
            }
        }
        with patch('amct_pytorch.amct_pytorch_inner.amct_pytorch.configuration.quant_calibration_config.get_quant_layer_config', return_value=config):
            self.insert_kv_cache_quant_pass.do_pass(self.model, self.model.layer3, 'layer3')

    def test_do_pass_hfmg(self):
        config = {
            'kv_data_quant_config':{
                'act_algo': 'hfmg',
                'num_bits': 8,
                'batch_num': 1,
                'with_offset': False,
                'num_of_bins': 4096,
                'quant_granularity': 1
            }
        }
        with patch('amct_pytorch.amct_pytorch_inner.amct_pytorch.configuration.quant_calibration_config.get_quant_layer_config', return_value=config):
            self.insert_kv_cache_quant_pass.do_pass(self.model, self.model.layer3, 'layer3')
