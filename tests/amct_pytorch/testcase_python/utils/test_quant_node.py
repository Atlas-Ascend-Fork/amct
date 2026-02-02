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

from amct_pytorch.amct_pytorch_inner.amct_pytorch.parser.parser import Parser
from amct_pytorch.amct_pytorch_inner.amct_pytorch.utils.quant_node import QuantOpInfo


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

    def test_get_scale_shape_rnn_per_channel(self):
        class RNNModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.lstm = torch.nn.LSTM(10, 20, 1)
                self.gru = torch.nn.GRU(10, 20, 1)
            def forward(self, input, hx):
                x = self.lstm(input, hx)
                y = self.gru(input, hx[0])
                return x, y
        model = RNNModule()
        tmp_onnx = BytesIO()
        Parser.export_onnx(model, (torch.randn(1, 1, 10), (torch.randn(1, 1, 20), torch.randn(1, 1, 20))), tmp_onnx)
        graph = Parser.parse_net_to_graph(tmp_onnx)
        node1 = graph.get_node_by_name('lstm')
        node2 = graph.get_node_by_name('gru')

        self.assertEqual(QuantOpInfo.get_scale_shape(node1, True), ([1, 80, 1], 80))
        self.assertEqual(QuantOpInfo.get_scale_shape(node2, True), ([1, 60, 1], 60))

    def test_get_scale_shape_rnn_per_tensor(self):
        class RNNModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.lstm = nn.LSTM(10, 20, 1)
                self.gru = nn.GRU(10, 20, 1)
            def forward(self, input, hx):
                x = self.lstm(input, hx)
                y = self.gru(input, hx[0])
                return x, y
        model = RNNModule()
        tmp_onnx = BytesIO()
        Parser.export_onnx(model, (torch.randn(1, 1, 10), (torch.randn(1, 1, 20), torch.randn(1, 1, 20))), tmp_onnx)
        graph = Parser.parse_net_to_graph(tmp_onnx)
        node1 = graph.get_node_by_name('lstm')
        node2 = graph.get_node_by_name('gru')

        self.assertEqual(QuantOpInfo.get_scale_shape(node1, False), ([4], 4))
        self.assertEqual(QuantOpInfo.get_scale_shape(node2, False), ([3], 3))

    def test_get_bias_for_matmul(self):
        class MatmulAddModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(3, 3, bias=False)
                self.register_buffer('add_tensor', torch.randn(3))
                self.linear1 = torch.nn.Linear(3, 1, bias=False)
                self.register_buffer('add_tensor1', torch.randn(1))

            def forward(self, x):
                return self.linear(x) + self.add_tensor, self.linear1(x) + self.add_tensor1
        model = MatmulAddModel().to(torch.device("cpu"))
        tmp_onnx = BytesIO()
        Parser.export_onnx(model, torch.randn(5, 3), tmp_onnx)
        graph = Parser.parse_net_to_graph(tmp_onnx)

        for node in graph.nodes:
            if node.type == 'MatMul':
                bias_node = QuantOpInfo.get_bias_for_matmul(node)
                self.assertIsNotNone(bias_node)

    def test_dual_input_matmul_add_shape_one(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer('add_tensor', torch.randn(1))

            def forward(self, x, y):
                z = torch.matmul(x, y)
                z = z + self.add_tensor
                return z
        model = Model().to(torch.device("cpu"))
        tmp_onnx = BytesIO()
        input_data = torch.randn(3, 3)
        Parser.export_onnx(model, (input_data, input_data), tmp_onnx)
        graph = Parser.parse_net_to_graph(tmp_onnx)

        for node in graph.nodes:
            if node.type == 'MatMul':
                bias_node = QuantOpInfo.get_bias_for_matmul(node)
                self.assertIsNone(bias_node)
