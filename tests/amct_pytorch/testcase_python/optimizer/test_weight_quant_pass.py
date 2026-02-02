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
from io import BytesIO
import unittest
from unittest.mock import patch
import json
import numpy as np
import torch

from .utils import models
from .utils import record_file_utils
from amct_pytorch.amct_pytorch_inner.amct_pytorch.parser.parser import Parser
from amct_pytorch.amct_pytorch_inner.amct_pytorch.optimizer.graph_optimizer import GraphOptimizer
from amct_pytorch.amct_pytorch_inner.amct_pytorch.configuration.configuration import Configuration
from amct_pytorch.amct_pytorch_inner.amct_pytorch.proto import scale_offset_record_pb2
from google.protobuf import text_format

from amct_pytorch.amct_pytorch_inner.amct_pytorch.optimizer.insert_weight_quant_pass import InsertWeightQuantPass
from amct_pytorch.amct_pytorch_inner.amct_pytorch.utils.vars import QUANTIZABLE_TYPES
from amct_pytorch.amct_pytorch_inner.amct_pytorch.utils.onnx_initializer_util import TensorProtoHelper


CUR_DIR = os.path.split(os.path.realpath(__file__))[0]

class TestWeightQuantPass(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        QUANTIZABLE_TYPES.extend(['ConvTranspose2d','AvgPool2d'])
        cls.temp_folder = os.path.join(CUR_DIR, 'test_weight_quant_pass')
        if not os.path.isdir(cls.temp_folder):
            os.makedirs(cls.temp_folder)

        cls.model_001 = models.Net001().to(torch.device("cpu"))
        cls.args_shape = [(1, 2, 28, 28)]
        cls.args = list()
        for input_shape in cls.args_shape:
            cls.args.append(torch.randn(input_shape))
        cls.args = tuple(cls.args)

        tmp_onnx = BytesIO()
        Parser.export_onnx(cls.model_001, cls.args, tmp_onnx)
        cls.graph = Parser.parse_net_to_graph(tmp_onnx)

        cls.records = record_file_utils.generate_records(
            layers_length={
                "layer1.0": 16,
                "layer2.0": 16,
                "fc.2": 1
            })

    @classmethod
    def tearDownClass(cls):
        QUANTIZABLE_TYPES.remove('ConvTranspose2d')
        QUANTIZABLE_TYPES.remove('AvgPool2d')
        os.popen('rm -r ' + cls.temp_folder)

    def test_quant_weight(self):
        passer = InsertWeightQuantPass(self.records)
        optimizer = GraphOptimizer()
        optimizer.add_pass(passer)
        optimizer.do_optimizer(self.graph, None)
        weight_dtype = TensorProtoHelper(
            self.graph.get_node_by_name('layer1.0.sub_module.weight').proto).get_data().dtype
        self.assertEqual(weight_dtype, 'int8')

    def test_quant_weight_int4(self):
        with patch('amct_pytorch.amct_pytorch_inner.amct_pytorch.utils.quant_node.QuantOpInfo.get_dst_num_bits', return_value=4):
            passer = InsertWeightQuantPass(self.records)
            before_nodes= len(self.graph.nodes)
            optimizer = GraphOptimizer()
            optimizer.add_pass(passer)
            optimizer.do_optimizer(self.graph, None)
            after_nodes = len(self.graph.nodes)
            self.assertEqual(after_nodes - before_nodes, 6)

    def test_rnn_weight_quant_success(self):
        class RNNModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.lstm = torch.nn.LSTM(10, 20, 1)
            def forward(self, input, hx):
                x = self.lstm(input, hx)
                return x
        model = RNNModule()
        tmp_onnx = BytesIO()
        Parser.export_onnx(model, (torch.randn(1, 1, 10), (torch.randn(1, 1, 20), torch.randn(1, 1, 20))), tmp_onnx)
        graph = Parser.parse_net_to_graph(tmp_onnx)
        node_name = 'lstm'
        node = graph.get_node_by_name(node_name)

        records = {
            node_name: {
                'weight_scale': np.array([1.0]*4, dtype=np.float32),
                'weight_offset': np.array([0]*4, dtype=np.int8),
                'recurrence_weight_scale': np.array([1.0]*4, dtype=np.float32),
                'recurrence_weight_offset': np.array([0]*4, dtype=np.int8),
            }
        }

        passer = InsertWeightQuantPass(records)
        passer.quant_recurrence_weight(node)
