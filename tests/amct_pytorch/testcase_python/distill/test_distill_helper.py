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

from amct_pytorch.graph_based_compression.amct_pytorch.distill.distill_helper import DistillHelper
from amct_pytorch.graph_based_compression.amct_pytorch.nn.module.quantization.conv2d import Conv2dQAT
from amct_pytorch.graph_based_compression.amct_pytorch.distill.distill_sample import ModelSingleTensorInput
from amct_pytorch.graph_based_compression.amct_pytorch.distill.distill_sample import DistillSampleBase

CUR_DIR = os.path.split(os.path.realpath(__file__))[0]

class DistillNet(nn.Module):
    def __init__(self):
        super(DistillNet, self).__init__()
        self.conv = nn.Conv2d(2, 2, kernel_size=3, bias=False)

    def forward(self, x):
        x = self.conv(x)
        return x

class DistillQATNet(nn.Module):
    def __init__(self):
        super(DistillQATNet, self).__init__()
        self.conv = Conv2dQAT(2, 2, kernel_size=3, bias=False)

    def forward(self, x):
        x = self.conv(x)
        return x

class TestDistillHelper(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    @classmethod
    def setUpClass(cls):
        cls.temp_folder = os.path.join(CUR_DIR, 'test_distill_helper')
        if not os.path.isdir(cls.temp_folder):
            os.makedirs(cls.temp_folder)

        cls.data = torch.randn(1,2,4,4)
        cls.train_loader = torch.utils.data.DataLoader(cls.data)
        cls.groups = [['conv']]
        # torch model
        cls.torch_model = DistillNet()
        cls.qat_model = DistillQATNet()
        cls.cfg_file = os.path.join(CUR_DIR, 'cfgs/distill_cfg.json')
        cls.sample_ins = ModelSingleTensorInput()

    @classmethod
    def tearDownClass(cls):
        os.system('rm -r ' + cls.temp_folder)
        print("[UNITTEST END test_distill_helper.py]")


    def test_run_model_one_batch(self):
        DistillHelper.run_model_one_batch(self.torch_model, self.data)


    def test_get_distill_modules(self):
        distill_helper = DistillHelper(self.torch_model, self.qat_model, self.cfg_file, loss=None, sample_instance=None)
        layer_names = ['conv']
        modules = DistillHelper.get_distill_modules(self.torch_model, layer_names)
        self.assertIsNotNone(modules)

    def test_get_distill_modules_invalid_layer(self):
        distill_helper = DistillHelper(self.torch_model, self.qat_model, self.cfg_file, loss=None, sample_instance=None)
        layer_names = ['conv_invalid']
        self.assertRaises(RuntimeError, DistillHelper.get_distill_modules, self.torch_model, layer_names)

    def test_get_distill_modules_loss(self):
        distill_helper = DistillHelper(self.torch_model, self.qat_model, self.cfg_file, loss=None, sample_instance=None)
        input_data = torch.tensor(torch.randn(1,2,4,4))
        target = torch.tensor(torch.randn(1,2,2,2))

        modules = []
        for name, module in self.torch_model.named_modules():
            if name == 'conv':
                modules.append(module)

        loss_val = distill_helper.get_distill_modules_loss(modules, input_data, target)
        self.assertIsNotNone(loss_val)

    def test_get_distill_modules_loss_invalid_target(self):
        distill_helper = DistillHelper(self.torch_model, self.qat_model, self.cfg_file, loss=None, sample_instance=None)
        input_data = torch.tensor([1,2,4,4])
        target = torch.tensor([1,2,4,4])
        modules = []
        for name, module in self.torch_model.named_modules():
            if name == 'conv':
                modules.append(module)

        self.assertRaises(RuntimeError, distill_helper.get_distill_modules_loss, modules, input_data, target)

    def test_do_calibration(self):
        distill_helper = DistillHelper(self.torch_model, self.qat_model, self.cfg_file, loss=None, sample_instance=None)
        distill_helper.do_calibration(self.train_loader)

    def test_gen_optimizer(self):
        distill_helper = DistillHelper(self.torch_model, self.qat_model, self.cfg_file, loss=None, sample_instance=None)
        modules = []
        for name, module in self.qat_model.named_modules():
            if name == 'conv':
                modules.append(module)
        self.assertIsNotNone(distill_helper.gen_optimizer_per_group(modules, None))
    
    def test_user_define_gen_optimizer(self):
        distill_helper = DistillHelper(self.torch_model, self.qat_model, self.cfg_file, loss=None, sample_instance=None)

        modules = []
        for name, module in self.qat_model.named_modules():
            if name == 'conv':
                modules.append(module)
        optimizer = torch.optim.AdamW(self.qat_model.parameters(), lr=0.1)
        self.assertIsNotNone(distill_helper.gen_optimizer_per_group(modules, optimizer))
