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
from io import BytesIO

import unittest
import torch
import torch.nn.functional as F

from amct_pytorch.amct_pytorch_inner.amct_pytorch.parser.parser import Parser
from amct_pytorch.amct_pytorch_inner.amct_pytorch.utils.onnx_conv_util import OnnxConvUtil


CUR_DIR = os.path.split(os.path.realpath(__file__))[0]

class TestOnnxConvUtil(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_folder = os.path.join(CUR_DIR, 'test_onnx_conv_util')
        if not os.path.isdir(cls.temp_folder):
            os.makedirs(cls.temp_folder)

    @classmethod
    def tearDownClass(cls):
        os.popen('rm -r ' + cls.temp_folder)

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_check_prune_reused_node(self):
        class ConvNet(torch.nn.Module):
            def __init__(self):
                super(ConvNet, self).__init__()
                self.conv1 = torch.nn.Conv2d(16, 32, 3, groups=16)
                self.conv1_1 = torch.nn.Conv2d(32, 32, 1, groups=1)
                self.conv2 = torch.nn.Conv2d(32, 64, 3, groups=2)

            # x represents our data
            def forward(self, x):
                # Pass data through conv1
                x = self.conv1(x)
                # Use the rectified-linear activation function over x
                x = F.relu(x)

                # repeated conv
                x = self.conv1_1(x)
                x = F.relu(x)

                x = self.conv1_1(x)
                x = F.relu(x)

                x = self.conv2(x)
                x = F.relu(x)
                return x

        model = ConvNet()
        dummy_input = torch.randn(1, 16, 28, 28)

        tmp_onnx = BytesIO()
        Parser.export_onnx(model, dummy_input, tmp_onnx)
        graph = Parser.parse_net_to_graph(tmp_onnx)
        graph.model = model

        conv_util = OnnxConvUtil(graph.get_node_by_name('conv1'))
        self.assertEqual(conv_util.is_depthwise_conv(), True)
        self.assertEqual(conv_util.get_depthwise_multiplier(), 2)

        conv_util = OnnxConvUtil(graph.get_node_by_name('conv2'))
        self.assertEqual(conv_util.is_group_conv(), True)