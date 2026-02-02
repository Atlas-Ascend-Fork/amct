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
import torch.nn as nn
import numpy as np

from amct_pytorch.amct_pytorch_inner.amct_pytorch.utils.module_info import ModuleInfo


class TestModuleInfo(unittest.TestCase):
    """
    The UT for evaluator helper
    """
    @classmethod
    def setUpClass(cls):
        print("TestModuleInfo start!")

    @classmethod
    def tearDownClass(cls):
        print("TestModuleInfo end!")
        pass

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_get_wts_cout_cin(self):
        conv2d = torch.nn.Conv2d(1,1,1)
        cout_axis, cin_axis = ModuleInfo.get_wts_cout_cin(conv2d)
        self.assertEqual(cout_axis, 0)
        self.assertEqual(cin_axis, 1)

        conv3d = torch.nn.Conv3d(1,1,1)
        cout_axis, cin_axis = ModuleInfo.get_wts_cout_cin(conv3d)
        self.assertEqual(cout_axis, 0)
        self.assertEqual(cin_axis, 1)

        conv1d = torch.nn.Conv1d(1,1,1)
        cout_axis, cin_axis = ModuleInfo.get_wts_cout_cin(conv1d)
        self.assertEqual(cout_axis, 0)
        self.assertEqual(cin_axis, 1)

        deconv2d = torch.nn.ConvTranspose2d(1,1,1)
        cout_axis, cin_axis = ModuleInfo.get_wts_cout_cin(deconv2d)
        self.assertEqual(cout_axis, 1)
        self.assertEqual(cin_axis, 0)

        deconv3d = torch.nn.ConvTranspose3d(1,1,1)
        cout_axis, cin_axis = ModuleInfo.get_wts_cout_cin(deconv3d)
        self.assertEqual(cout_axis, 1)
        self.assertEqual(cin_axis, 0)

        linear = torch.nn.Linear(1,1)
        cout_axis, cin_axis = ModuleInfo.get_wts_cout_cin(linear)
        self.assertEqual(cout_axis, 0)
        self.assertEqual(cin_axis, 1)

        conv_transpose_1d = torch.nn.ConvTranspose1d(1,1,1)
        cout_axis, cin_axis = ModuleInfo.get_wts_cout_cin(conv_transpose_1d)
        self.assertEqual(cout_axis, 1)
        self.assertEqual(cin_axis, 0)