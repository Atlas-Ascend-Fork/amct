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
from amct_pytorch.amct_pytorch_inner.amct_pytorch.custom_op.comp_module.comp_module_base import CompModuleBase

CUR_DIR = os.path.split(os.path.realpath(__file__))[0]

class TestCompModuleBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_folder = os.path.join(CUR_DIR, 'test_comp_module_base')
        if not os.path.isdir(cls.temp_folder):
            os.makedirs(cls.temp_folder)

        cls.conv_model = torch.nn.Conv2d(2, 4, kernel_size=2)
        cls.args_shape = [(1, 2, 8, 8)]
        cls.args = list()
        for input_shape in cls.args_shape:
            cls.args.append(torch.randn(input_shape))
        cls.onnx_file = os.path.join(cls.temp_folder, 'conv_model.onnx')
        # recorder
        cls.record_file = os.path.join(CUR_DIR, 'utils/conv_model.txt')
        cls.record_module = Recorder(cls.record_file)
        cls.gaussian_tensor = torch.tensor(
        [[[[-0.4534, -0.3931,  1.4165, -0.2845,  1.3123, -0.5861, -0.9216, -2.9547],
          [-1.1394, -0.5585,  1.0372,  0.8990, -0.4554,  0.0915,  0.1655, 0.7922],
          [ 0.4200,  0.0399,  0.2150, -0.4659,  1.0101, -0.0448,  1.6585, 0.3404],
          [-0.3245, -0.8681,  1.4481, -1.2872,  0.8476, -0.3390,  0.6998, 0.9337],
          [-0.2462,  0.0883,  0.7797,  0.2625, -0.5033,  0.3216, -0.5286, -1.1610],
          [-0.7261,  0.9066,  0.7493,  0.2082, -0.0145,  1.0608, -0.0193, 1.1523],
          [-0.8728,  0.7968,  0.5643, -0.9889, -1.5797,  1.3663,  0.3207, -0.5696],
          [-1.1566,  0.5097, -0.2754,  0.4404,  0.9737,  1.1980,  0.1219, 0.2367]],
         [[ 1.4429,  1.8090,  0.5159, -2.3921, -0.5751, -1.1347, -0.4845, -0.8817],
          [ 0.4223,  0.6367,  1.3398,  0.5552,  1.2147,  0.0621, -0.7931, -0.4430],
          [-0.6708,  0.0983, -0.3347, -1.1744,  0.0569, -0.7386,  0.5141, -0.6317],
          [ 1.6589,  0.3437, -0.2570,  0.4474, -0.3003, -1.4195,  0.1037, -0.4414],
          [-0.3674,  0.0741,  0.7980,  0.8538,  1.1619,  0.2918, -0.3035, 1.5540],
          [ 1.9373,  2.5682, -0.2986, -0.2450, -1.4277,  0.3202, -1.5792, 0.7967],
          [-1.1843, -2.2168, -0.3940,  0.7455,  1.1740, -0.7198, -0.0188, -0.4981],
          [-0.8536,  0.5208, -1.8724, -0.7368,  0.2126, -0.0707,  0.5122, 0.5675]]]])

        cls.act_config = {
            'num_bits': 8
        }

        cls.wts_config = {
            'num_bits': 8,
            'channel_wise': False
        }

        cls.common_config = {
            'device': 'cpu',
            'need_sync': False,
            'process_group': None,
            'world_size': 1,
            'layers_name': ['conv1'],
            'batch_num': 1
        }

        cls.comp_args = {
            'module': cls.conv_model,
            'act_config': cls.act_config,
            'wts_config': cls.wts_config,
            'common_config': cls.common_config,
            'acts_comp_reuse': True
        }

    @classmethod
    def tearDownClass(cls):
        os.popen('rm -r ' + cls.temp_folder)

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_ulq_config_single_batch_success(self):
        comp_module = CompModuleBase(**self.comp_args)
        in_data = self.gaussian_tensor
        # base class has no attribute 'cur_batch', call IFMR sucess
        with self.assertRaises(AttributeError) as cm:
            comp_module.forward(in_data)

    def test_ulq_config_multi_batch_success(self):
        self.common_config['batch_num'] = 2
        comp_module = CompModuleBase(**self.comp_args)
        in_data = self.gaussian_tensor
        # base class has no attribute 'cur_batch', call IFMR sucess
        with self.assertRaises(AttributeError) as cm:
            comp_module.forward(in_data)

