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
from amct_pytorch.graph_based_compression.amct_pytorch.custom_op.comp_module.comp_module_conv_transpose1d import CompModuleConvTranspose1d

CUR_DIR = os.path.split(os.path.realpath(__file__))[0]

class TestCompModuleConvTranspose1d(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_folder = os.path.join(CUR_DIR, 'test_comp_module_conv_transpose1d')
        if not os.path.isdir(cls.temp_folder):
            os.makedirs(cls.temp_folder)

        cls.module = torch.nn.ConvTranspose1d(3, 3, 3)
        cls.input = torch.randn(3, 3, 8)

        cls.act_config = {
            'num_bits': 8,
            'clip_max': 1.0,
            'clip_min': -1.0,
            'algo': 'ulq_quantize'
        }

        cls.wts_config = {
            'num_bits': 8,
            'channel_wise': False,
            'algo': 'arq_retrain'
        }

        cls.common_config = {
            'device': 'cpu',
            'need_sync': False,
            'process_group': None,
            'world_size': 1,
            'layers_name': ['conv_transpose1d'],
            'batch_num': 1
        }

        cls.comp_args = {
            'module': cls.module,
            'act_config': cls.act_config,
            'wts_config': cls.wts_config,
            'common_config': cls.common_config,
            'acts_comp_reuse': False
        }

    @classmethod
    def tearDownClass(cls):
        os.popen('rm -r ' + cls.temp_folder)

    def test_forward_success(self):
        comp_module = CompModuleConvTranspose1d(**self.comp_args)
        comp_module.comp_algs.append('quant')
        comp_module.forward(self.input)

        self.wts_config['channel_wise'] = True
        comp_module = CompModuleConvTranspose1d(**self.comp_args)
        comp_module.comp_algs.append('quant')
        comp_module.forward(self.input)