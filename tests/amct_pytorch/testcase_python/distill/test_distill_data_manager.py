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
import sys
import os
import unittest

import numpy as np
import torch
import torch.nn as nn

from amct_pytorch.graph_based_compression.amct_pytorch.distill.distill_data_manager import DistillDataManager
from amct_pytorch.graph_based_compression.amct_pytorch.distill.distill_helper import DistillHelper
from amct_pytorch.graph_based_compression.amct_pytorch.distill.distill_sample import ModelSingleTensorInput

CUR_DIR = os.path.split(os.path.realpath(__file__))[0]

class DistillNet(nn.Module):
    def __init__(self):
        super(DistillNet, self).__init__()
        self.conv = nn.Conv2d(2, 16, kernel_size=3, bias=False)

    def forward(self, x):
        x = self.conv(x)
        return x

class TestDistillDataManager(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    @classmethod
    def setUpClass(cls):
        cls.temp_folder = os.path.join(CUR_DIR, 'test_distill_data_manager')
        if not os.path.isdir(cls.temp_folder):
            os.makedirs(cls.temp_folder)

        cls.data = torch.randn(1,2,4,4)
        cls.train_loader = torch.utils.data.DataLoader(cls.data)
        cls.groups = [['conv']]
        cls.torch_model = DistillNet().to(torch.device("cpu"))
        cls.sample_ins = ModelSingleTensorInput()

    @classmethod
    def tearDownClass(cls):
        os.system('rm -r ' + cls.temp_folder)
        print("[UNITTEST END test_distill_data_manager.py]")

    def test_get_norm_min_data(self):
        data_t = torch.tensor([1.0])
        data_s = torch.tensor([0.0])
        cmp_result = DistillDataManager.get_norm_min_data(data_t, data_s)
        self.assertEqual(cmp_result, data_t)

        data_t = torch.tensor([1.0])
        data_s = torch.tensor([1.0])
        cmp_result = DistillDataManager.get_norm_min_data(data_t, data_s)
        self.assertEqual(cmp_result, data_s)

    def test_dump_data(self):
        distill_manager = DistillDataManager(self.sample_ins)
        # dump data
        distill_manager.dump_data(self.torch_model, self.groups, epochs=1, data_loader=self.train_loader)
        self.assertTrue(os.path.exists('tmp/data_dump_input_model_input_0_0_cpu.npy'))

        # load model input data
        load_model_data = distill_manager.load_model_input_dump_data(0, 0)
        self.assertIsNotNone(load_model_data)
        self.assertTrue(load_model_data[0].equal(self.data))

        # load group input data
        load_in_data = distill_manager.load_input_dump_data(self.groups[0], 0, 0)
        self.assertIsNotNone(load_in_data)
        self.assertTrue(load_in_data.equal(self.data))

        # load group output data
        load_out_data = distill_manager.load_output_dump_data(self.groups[0], 0, 0)
        self.assertIsNotNone(load_out_data)
        infer_output = self.torch_model(self.data)
        self.assertTrue(load_out_data.equal(infer_output))

        # release
        distill_manager.release()
        self.assertFalse(os.path.exists('tmp/data_dump_input_model_input_0_0_cpu.npy'))
    
    def test_infer_data(self):
        distill_manager = DistillDataManager(self.sample_ins)

        # infer input data
        infer_in_data = distill_manager.get_input_data_by_inferring(self.torch_model, self.groups[0], self.data)
        self.assertIsNotNone(infer_in_data)
        
        # infer output data
        infer_out_data = distill_manager.get_output_data_by_inferring(self.torch_model, self.groups[0], self.data)
        self.assertIsNotNone(infer_out_data)

    def test_infer_data_invalid_group(self):
        distill_manager = DistillDataManager(self.sample_ins)

        # infer input data
        self.assertRaises(RuntimeError, distill_manager.get_input_data_by_inferring, self.torch_model, ['linear'], self.data)
        
        # infer output data
        self.assertRaises(RuntimeError, distill_manager.get_output_data_by_inferring, self.torch_model, ['linear'], self.data)
    


