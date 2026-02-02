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
from amct_pytorch.amct_pytorch_inner.amct_pytorch.graph.graph import Graph
from amct_pytorch.amct_pytorch_inner.amct_pytorch.optimizer.insert_quant_pass import construct_quant_node

from amct_pytorch.amct_pytorch_inner.amct_pytorch.optimizer.quant_fusion_pass import QuantFusionPass

class TestQuantFusionPass(unittest.TestCase):
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
        def conv_sub(graph, conv_name, inputs, outputs, quant_attrs):
            # Add Ascend Quant
            quant_node = graph.node.add()
            quant_node.CopyFrom(construct_quant_node(inputs, ['%s_quant' % (conv_name)],
                quant_attrs, conv_name))
            # Add conv
            conv = graph.node.add()
            conv.name = conv_name
            conv.op_type = 'Conv'
            conv.input[:] = ['%s_quant' % (conv_name), '%s.weights' % (conv_name), '%s.bias' % (conv_name)]
            conv.output[:] = [conv_name]
            # add attribute "kernel_shape"
            kernel_shape = conv.attribute.add()
            kernel_shape.name = 'kernel_shape'
            kernel_shape.type = onnx_pb.AttributeProto.AttributeType.INTS
            kernel_shape.ints[:] = [64, 3, 3, 3]
            # add attribute "pads"
            pads = conv.attribute.add()
            pads.name = 'pads'
            pads.type = onnx_pb.AttributeProto.AttributeType.INTS
            pads.ints[:] = [0, 0, 0, 0]
            # Add weights
            weights = graph.initializer.add()
            weights.name = '%s.weights' % (conv_name)
            weights.data_type = 3
            weights.int32_data[:] = [1, 2, 3, 4, 5, 6]
            weights.dims[:] = [1, 1, 2, 3]
            # Add bias
            bias = graph.initializer.add()
            bias.name = '%s.bias' % (conv_name)
            bias.data_type = 6
            bias.int32_data[:] = [0]
            bias.dims[:] = [1]
            # Add relu
            relu1 = graph.node.add()
            relu1.name = '%s.relu' % (conv_name)
            relu1.op_type = 'Relu'
            relu1.input[:] = [conv_name]
            relu1.output[:] = outputs
        conv_sub(self.graph, 'conv1', ['data0'], ['conv1_output'], {'scale': 1, 'offset': 0, 'quant_bit': 8, 'dst_type': 'INT8'})
        conv_sub(self.graph, 'conv2', ['data0'], ['conv2_output'], {'scale': 0.99999, 'offset': 0, 'quant_bit': 8, 'dst_type': 'INT8'})
        conv_sub(self.graph, 'conv3', ['data0'], ['conv3_output'], {'scale': 1, 'offset': 0, 'quant_bit': 8, 'dst_type': 'INT8'})
        # Add add
        add1 = self.graph.node.add()
        add1.name = 'add1'
        add1.op_type = 'Add'
        add1.input[:] = ['conv1_output', 'conv2_output', 'conv3_output']
        add1.output[:] = ['output']
        # add output
        graph_output = self.graph.output.add()
        graph_output.name = 'output'
        graph_output.type.tensor_type.shape.dim.add().dim_value = 1
        self.model_proto.graph.CopyFrom(self.graph)

    def tearDown(self):
        pass

    def test_match_pattern_success(self):
        records = {
            'conv1': {'data_scale': 1, 'data_offset': 0},
            'conv2': {'data_scale': 0.99999, 'data_offset': 0},
            'conv3': {'data_scale': 1, 'data_offset': 0},
        }
        test_model = deepcopy(self.model_proto)
        graph = Graph(test_model)
        data_node = graph._in_out_nodes[0]
        self.assertTrue(QuantFusionPass(records).match_pattern(data_node))

    def test_match_pattern_failed(self):
        records = {
            'conv1': {'data_scale': 1, 'data_offset': 0},
            'conv2': {'data_scale': 0.99999, 'data_offset': 0},
            'conv3': {'data_scale': 1, 'data_offset': 0},
        }
        test_model = deepcopy(self.model_proto)
        graph = Graph(test_model)
        add_node = graph.get_node_by_name('add1')
        self.assertFalse(QuantFusionPass(records).match_pattern(add_node))

    def test_do_pass_success(self):
        records = {
            'conv1': {'data_scale': 1, 'data_offset': 0},
            'conv2': {'data_scale': 0.99999, 'data_offset': 0},
            'conv3': {'data_scale': 1, 'data_offset': 0},
        }
        test_model = deepcopy(self.model_proto)
        graph = Graph(test_model)
        data_node = graph._in_out_nodes[0]
        graph.get_node_by_name('conv1.quant').set_attr('object_node', 'conv1')
        graph.get_node_by_name('conv2.quant').set_attr('object_node', 'conv2')
        graph.get_node_by_name('conv3.quant').set_attr('object_node', 'conv3')

        before_length = len(graph.nodes)
        QuantFusionPass(records).do_pass(graph, data_node)
        after_length = len(graph.nodes)
        self.assertEqual(before_length -  after_length, 1)
        self.assertRaises(
            RuntimeError,
            graph.get_node_by_name,
            'conv3_quant')
