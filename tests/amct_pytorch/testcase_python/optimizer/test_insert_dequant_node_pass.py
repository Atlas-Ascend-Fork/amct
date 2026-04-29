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
from copy import deepcopy

import json
import numpy as np
import torch

from onnx import onnx_pb
from amct_pytorch.graph_based_compression.amct_pytorch.graph.graph import Graph

from amct_pytorch.graph_based_compression.amct_pytorch.optimizer.insert_dequant_pass import InsertDequantPass

class TestInsertDequantPass(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pass

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):
        # set basic info
        self.model_proto = onnx_pb.ModelProto()
        self.model_proto.producer_name = 'model'
        self.graph = onnx_pb.GraphProto()
        # Add graph input 0
        graph_input0 = self.graph.input.add()
        graph_input0.name = 'data0'
        graph_input0.type.tensor_type.shape.dim.add().dim_value = 3
        graph_input0.type.tensor_type.shape.dim.add().dim_value = 3
        graph_input0.type.tensor_type.shape.dim.add().dim_value = 3
        graph_input0.type.tensor_type.shape.dim.add().dim_value = 3
        # Add graph input 1
        graph_input1 = self.graph.input.add()
        graph_input1.name = 'data1'
        graph_input1.type.tensor_type.shape.dim.add().dim_value = 1
        # Add conv1
        conv1 = self.graph.node.add()
        conv1.name = 'conv1'
        conv1.op_type = 'Conv'
        conv1.input[:] = ['data0', 'conv1.weights', 'conv1.bias']
        conv1.output[:] = ['output']
        # add attribute "kernel_shape"
        kernel_shape = conv1.attribute.add()
        kernel_shape.name = 'kernel_shape'
        kernel_shape.type = onnx_pb.AttributeProto.AttributeType.INTS
        kernel_shape.ints[:] = [64, 3, 3, 3]
        # add attribute "pads"
        pads = conv1.attribute.add()
        pads.name = 'pads'
        pads.type = onnx_pb.AttributeProto.AttributeType.INTS
        pads.ints[:] = [0, 0, 0, 0]
        # Add weights
        weights = self.graph.initializer.add()
        weights.name = 'conv1.weights'
        weights.data_type = 3
        weights.int32_data[:] = [1, 2, 3, 4, 5, 6]
        weights.dims[:] = [1, 1, 2, 3]
        # Add bias
        bias = self.graph.initializer.add()
        bias.name = 'conv1.bias'
        bias.data_type = 6
        bias.int32_data[:] = [0]
        bias.dims[:] = [1]
        # add output
        graph_output = self.graph.output.add()
        graph_output.name = 'output'
        graph_output.type.tensor_type.shape.dim.add().dim_value = 1
        self.model_proto.graph.CopyFrom(self.graph)

    def tearDown(self):
        pass

    def test_match_pattern_success(self):
        records = {'conv1': {
                'data_scale': 1,
                'data_offset': 0
            }
        }
        test_model = deepcopy(self.model_proto)
        conv_node = Graph(test_model).get_node_by_name('conv1')
        self.assertTrue(InsertDequantPass(records).match_pattern(conv_node))

    def test_match_pattern_not_in_quantizable_types(self):
        records = {'conv1': {
                'data_scale': 1,
                'data_offset': 0
            }
        }
        test_model = deepcopy(self.model_proto)
        relu_node = Graph(test_model).get_node_by_name('conv1.weights')
        self.assertFalse(InsertDequantPass(records).match_pattern(relu_node))

    def test_match_pattern_not_in_records(self):
        records = {'conv2': {
                'data_scale': 1,
                'data_offset': 0
            }
        }
        test_model = deepcopy(self.model_proto)
        conv_node = Graph(test_model).get_node_by_name('conv1')
        self.assertFalse(InsertDequantPass(records).match_pattern(conv_node))

    def test_do_pass_success(self):
        records = {'conv1': {
                'data_scale': 1,
                'data_offset': 0,
                'weight_scale': np.array([1]),
                'weight_offset': np.array([0]),
            }
        }
        test_model = deepcopy(self.model_proto)
        graph = Graph(test_model)
        conv_node = graph.get_node_by_name('conv1')
        before_nodes_num = len(graph.nodes)
        InsertDequantPass(records).do_pass(graph, conv_node)
        after_nodes_num = len(graph.nodes)
        self.assertEqual(after_nodes_num - before_nodes_num, 2)
        
