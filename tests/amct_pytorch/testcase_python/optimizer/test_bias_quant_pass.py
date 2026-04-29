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
import json
import numpy as np
import torch
from unittest.mock import patch

from .utils import models
from .utils import record_file_utils
from amct_pytorch.graph_based_compression.amct_pytorch.parser.parser import Parser
from amct_pytorch.graph_based_compression.amct_pytorch.optimizer.graph_optimizer import GraphOptimizer
from amct_pytorch.graph_based_compression.amct_pytorch.configuration.configuration import Configuration
from amct_pytorch.graph_based_compression.amct_pytorch.proto import scale_offset_record_pb2
from amct_pytorch.graph_based_compression.amct_pytorch.utils.onnx_initializer_util import TensorProtoHelper
from google.protobuf import text_format

from amct_pytorch.graph_based_compression.amct_pytorch.optimizer.insert_bias_quant_pass import InsertBiasQuantPass
from amct_pytorch.graph_based_compression.amct_pytorch.optimizer.replace_bias_quant_pass import ReplaceBiasQuantPass


CUR_DIR = os.path.split(os.path.realpath(__file__))[0]

class TestInsertBiasQuantPass(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_folder = os.path.join(CUR_DIR, 'test_bias_quant_pass')
        if not os.path.isdir(cls.temp_folder):
            os.makedirs(cls.temp_folder)

        cls.model_001 = models.Net001().to(torch.device("cpu"))
        cls.args_shape = [(1, 2, 28, 28)]
        cls.args = list()
        for input_shape in cls.args_shape:
            cls.args.append(torch.randn(input_shape))
        cls.args = tuple(cls.args)

        cls.onnx_file = os.path.join(cls.temp_folder, 'net_001.onnx')
        Parser.export_onnx(cls.model_001, cls.args, cls.onnx_file)
        cls.graph = Parser.parse_net_to_graph(cls.onnx_file)

        cls.records = record_file_utils.generate_records(
            layers_length={
                "layer1.0": 16,
                "layer2.0": 16,
                "fc.2": 1,
            })

    @classmethod
    def tearDownClass(cls):
        os.popen('rm -r ' + cls.temp_folder)
        pass

    def test_quant_bias(self):
        optimizer = GraphOptimizer()
        optimizer.add_pass(InsertBiasQuantPass(self.records))
        optimizer.do_optimizer(self.graph, None)
        bias_dtype = TensorProtoHelper(self.graph.get_node_by_name('layer2.0.sub_module.bias').proto).get_data().dtype
        self.assertEqual(bias_dtype, 'int32')

    def test_quant_bias_int4(self):
        optimizer = GraphOptimizer()
        before_nodes = len(self.graph.nodes)
        with patch('amct_pytorch.graph_based_compression.amct_pytorch.utils.quant_node.QuantOpInfo.get_dst_num_bits', return_value=4):
            optimizer.add_pass(InsertBiasQuantPass(self.records))
            optimizer.do_optimizer(self.graph, None)
            after_nodes = len(self.graph.nodes)
            self.assertEqual(after_nodes - before_nodes, 1)

    def test_replace_bias_quant_int4(self):
        optimizer = GraphOptimizer()
        before_nodes = len(self.graph.nodes)
        with patch('amct_pytorch.graph_based_compression.amct_pytorch.utils.quant_node.QuantOpInfo.get_dst_num_bits', return_value=4):
            optimizer.add_pass(ReplaceBiasQuantPass(self.records))
            optimizer.do_optimizer(self.graph, None)
            after_nodes = len(self.graph.nodes)
            self.assertEqual(before_nodes - after_nodes, 1)

    def test_bias_exceed_int32(self):
        bias = np.array([[0.0, 1.0, 2.0**31, -2.0**31], [0.0, 1.0, 2**31-1, -2**31-1]], dtype=np.float32)
        scale_w = np.array([0.1])
        scale_d = np.array(1.0)
        # InsertBiasQuantPass.quant_bias(bias, scale_w, scale_d, 'conv1')
        self.assertRaises(RuntimeError, InsertBiasQuantPass.quant_bias, bias, scale_w, scale_d, 'conv1')

    def test_rnn_bias_quant_success(self):
        layer_name = 'lstm'
        records = {
            layer_name: {
                'data_scale': np.array(1.0, dtype=np.float32),
                'h_scale': np.array(1.0, dtype=np.float32),
                'weight_scale': np.array([1.0]*4, dtype=np.float32),
                'weight_offset': np.array([0]*4, dtype=np.int8),
                'recurrence_weight_scale': np.array([1.0]*4, dtype=np.float32),
                'recurrence_weight_offset': np.array([0]*4, dtype=np.int8),
            }
        }
        bias = np.random.random([1, 160]).astype(np.float32)
        passer = InsertBiasQuantPass(records)
        quant_bias = passer.bias_quant_rnn(bias, layer_name)
        self.assertEqual(quant_bias.dtype, np.int32)