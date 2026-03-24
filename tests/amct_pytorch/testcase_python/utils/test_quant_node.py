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
from io import BytesIO
import torch
import torch.nn as nn
import numpy as np

from amct_pytorch.graph_based_compression.amct_pytorch.parser.parser import Parser
from amct_pytorch.graph_based_compression.amct_pytorch.utils.quant_node import QuantOpInfo


class TestQuantOpInfo(unittest.TestCase):
    """
    The UT for evaluator helper
    """
    @classmethod
    def setUpClass(cls):
        print("TestQuantOpInfo start!")

    @classmethod
    def tearDownClass(cls):
        print("TestQuantOpInfo end!")
        pass

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_get_dequant_shape(self):
        class Conv1dModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1d = torch.nn.Conv1d(1,1,1)
            def forward(self, x):
                return self.conv1d(x)
        conv1d_module = Conv1dModule()
        tmp_onnx = BytesIO()
        Parser.export_onnx(conv1d_module, torch.randn(1,1,1), tmp_onnx)
        graph = Parser.parse_net_to_graph(tmp_onnx)
        node = graph.get_node_by_name('conv1d')
        self.assertEqual(QuantOpInfo.get_dequant_shape(node), [1,-1,1])
