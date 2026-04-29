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

from amct_pytorch.graph_based_compression.amct_pytorch.custom_op.recorder.recorder import Recorder

CUR_DIR = os.path.split(os.path.realpath(__file__))[0]

class TestRecorder(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_folder = os.path.join(CUR_DIR, 'test_recorder')
        if not os.path.isdir(cls.temp_folder):
            os.makedirs(cls.temp_folder)

    @classmethod
    def tearDownClass(cls):
        os.popen('rm -r ' + cls.temp_folder)

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_record_kv_cache_factors_success(self):
        # recorder
        record_file = os.path.join(self.temp_folder, 'test_record_kv_cache_factors_success.txt')
        if not os.path.exists(record_file):
            with open(record_file, 'w') as f:
                f.write('')
        record_module = Recorder(record_file, enable_kv_cache_quant=True)
        record_module.quant_layer_names = ['layer_name']
        record_module.record_quant_layer('layer_name')
        scale = [0.1, 0.2]
        offset = [1, 2]
        record_module.forward('layer_name', 'kv_cache', {'scale':scale, 'offset':offset})
        self.assertTrue(os.path.exists(record_file))

    def test_add_kv_cache_factors_num_error(self):
        # recorder
        record_file = os.path.join(self.temp_folder, 'test_add_kv_cache_factors_num_error.txt')
        record_module = Recorder(record_file, enable_kv_cache_quant=True)
        quant_factors = {
            'scale': [0.1, 0.2],
            'offset': [1, 2]
        }
        with self.assertRaises(RuntimeError):
            record_module._add_kv_cache_factors('layer_name', quant_factors)

    def test_add_kv_cache_factors_factor_error(self):
        # recorder
        record_file = os.path.join(self.temp_folder, 'test_add_kv_cache_factors_factor_error.txt')
        if not os.path.exists(record_file):
            with open(record_file, 'w') as f:
                f.write('')
        record_module = Recorder(record_file, enable_kv_cache_quant=True)
        record_module.record_quant_layer('layer_name')
        record_module._read_record_file()
        quant_factors = {
            'scale': None,
            'offset': None
        }
        with self.assertRaises(ValueError):
            record_module._add_kv_cache_factors('layer_name', quant_factors)