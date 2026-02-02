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

from amct_pytorch.amct_pytorch_inner.amct_pytorch.custom_op.recorder.recorder import Recorder
from amct_pytorch.amct_pytorch_inner.amct_pytorch.custom_op.ifmr.ifmr import IFMR
from amct_pytorch.amct_pytorch_inner.amct_pytorch.custom_op.kv_cache_quant import KVCacheQuant

CUR_DIR = os.path.split(os.path.realpath(__file__))[0]

class TestKVCacheQuant(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_folder = os.path.join(CUR_DIR, 'test_kv_cache_quant')
        if not os.path.isdir(cls.temp_folder):
            os.makedirs(cls.temp_folder)
        # linear
        cls.linear = torch.nn.Linear(4, 8)
        cls.args_shape = (16, 4)
        cls.input_data = (torch.randn(cls.args_shape))
        # cali_module
        cls.ifmr_args = {
            'layers_name': ['linear1'],
            'num_bits': 8,
            'batch_num': 1,
            'with_offset': False,
            'max_percentile': 0.999999,
            'min_percentile': 0.999999,
            'search_start': 0.7,
            'search_end': 1.3,
            'search_step': 0.01,
            'quant_granularity': 1
        }
        cls.cali_module = IFMR(**cls.ifmr_args)
        # recorder
        cls.record_file = os.path.join(cls.temp_folder, 'test_kv_cache_quant_record.txt')
        if not os.path.exists(cls.record_file):
            with open(cls.record_file, 'w') as f:
                f.write('')
        cls.record_module = Recorder(cls.record_file, enable_kv_cache_quant=True)

    @classmethod
    def tearDownClass(cls):
        os.popen('rm -r ' + cls.temp_folder)

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_kv_cache_quant_forward_success(self):
        self.record_module.record_quant_layer(['linear1'])
        kv_cache_quant_module = KVCacheQuant(self.linear,
                                             self.cali_module,
                                             self.record_module,
                                             ['linear1'],
                                             self.ifmr_args)
        kv_cache_quant_module.forward(self.input_data)
        self.assertTrue(os.path.exists(self.record_file))