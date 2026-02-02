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
from onnx import helper, TensorProto, ValueInfoProto
from amct_pytorch.amct_pytorch_inner.amct_pytorch.graph.graph import Graph
from amct_pytorch.amct_pytorch_inner.amct_pytorch.graph.node import Node

class TestGraph(unittest.TestCase):
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
        conv1.output[:] = ['conv1']
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
        # Add relu
        relu1 = self.graph.node.add()
        relu1.name = 'relu1'
        relu1.op_type = 'Relu'
        relu1.input[:] = ['conv1']
        relu1.output[:] = ['relu1']
        # Add add
        add1 = self.graph.node.add()
        add1.name = 'add1'
        add1.op_type = 'Add'
        add1.input[:] = ['relu1', 'data1']
        add1.output[:] = ['add1']
        # Add average_pool
        avg_pool1 = self.graph.node.add()
        avg_pool1.name = 'avg_pool1'
        avg_pool1.op_type = 'AveragePool'
        avg_pool1.input[:] = ['add1']
        avg_pool1.output[:] = ['output']
        # add output
        graph_output = self.graph.output.add()
        graph_output.name = 'output'
        graph_output.type.tensor_type.shape.dim.add().dim_value = 1
        self.model_proto.graph.CopyFrom(self.graph)

    def tearDown(self):
        pass

    def test_graph_init(self):
        test_model = deepcopy(self.model_proto)
        graph = Graph(test_model)
        self.assertEqual(len(graph.nodes), 6)
        self.assertEqual(graph.get_node(0).name, 'conv1.weights')
        self.assertEqual(graph.get_node(1).name, 'conv1.bias')
        self.assertEqual(graph.get_node(2).name, 'conv1')
        self.assertEqual(graph.get_node(3).name, 'relu1')
        self.assertEqual(graph.get_node(4).name, 'add1')
        self.assertEqual(graph.get_node(5).name, 'avg_pool1')
        self.assertEqual(graph.net.producer_name, 'model')
        model_proto = graph.dump_proto()
        self.assertEqual(type(model_proto), onnx_pb.ModelProto)

    def test_graph_init_without_name(self):
        test_model = deepcopy(self.model_proto)
        for node in test_model.graph.node:
            node.ClearField('name')
        graph = Graph(test_model)
        self.assertEqual(len(graph.nodes), 6)
        self.assertEqual(graph.get_node(0).name, 'conv1.weights')
        self.assertEqual(graph.get_node(1).name, 'conv1.bias')
        self.assertEqual(graph.get_node(2).ori_name, 'node_0')
        self.assertEqual(graph.get_node(3).ori_name, 'node_1')
        self.assertEqual(graph.get_node(4).ori_name, 'node_2')
        self.assertEqual(graph.get_node(5).ori_name, 'node_3')
        self.assertEqual(graph.net.producer_name, 'model')

    def test_graph_init_with_no_input_output(self):
        test_model = deepcopy(self.model_proto)
        output1 = test_model.graph.output.add()
        output1.name = 'output1'
        output1.type.tensor_type.shape.dim.add().dim_value = 1
        self.assertRaises(
            ReferenceError,
            Graph,
            test_model)

    def test_graph_init_node_input_not_exist(self):
        test_model = deepcopy(self.model_proto)
        empty_node = test_model.graph.node.add()
        empty_node.name = 'empty'
        empty_node.op_type = 'empty'
        empty_node.input[:] = ['not_exist_in']
        empty_node.output[:] = ['not_exist_out']
        self.assertRaises(
            ReferenceError,
            Graph,
            test_model)

    def test_linear_with_transpose(self):
        model_proto = onnx_pb.ModelProto()

        graph_input0 = model_proto.graph.input.add()
        graph_input0.name = 'data'
        graph_input0.type.tensor_type.shape.dim.add().dim_value = 3
        graph_input0.type.tensor_type.shape.dim.add().dim_value = 3
        graph_input0.type.tensor_type.shape.dim.add().dim_value = 3
        graph_input0.type.tensor_type.shape.dim.add().dim_value = 3

        transpose = model_proto.graph.node.add()
        transpose.name = 'transpose'
        transpose.op_type = 'Transpose'
        transpose.input[:] = ['fc.weights']
        transpose.output[:] = ['transpose']

        linear = model_proto.graph.node.add()
        linear.name = 'linear'
        linear.op_type = 'MatMul'
        linear.input[:] = ['data', 'transpose']
        linear.output[:] = ['fc']

        weights = model_proto.graph.initializer.add()
        weights.name = 'fc.weights'
        weights.data_type = 3
        weights.int32_data[:] = [1, 2, 3, 4, 5, 6]
        weights.dims[:] = [1, 1, 2, 3]

        graph = Graph(model_proto)
        self.assertEqual(graph.get_node(2).name, 'linear')

    def test_linear_with_transpose_mult_output(self):
        model_proto = onnx_pb.ModelProto()

        graph_input0 = model_proto.graph.input.add()
        graph_input0.name = 'data'
        graph_input0.type.tensor_type.shape.dim.add().dim_value = 3
        graph_input0.type.tensor_type.shape.dim.add().dim_value = 3
        graph_input0.type.tensor_type.shape.dim.add().dim_value = 3
        graph_input0.type.tensor_type.shape.dim.add().dim_value = 3

        transpose = model_proto.graph.node.add()
        transpose.name = 'transpose'
        transpose.op_type = 'Transpose'
        transpose.input[:] = ['fc.weights']
        transpose.output[:] = ['transpose']

        linear = model_proto.graph.node.add()
        linear.name = 'linear'
        linear.op_type = 'MatMul'
        linear.input[:] = ['data', 'transpose']
        linear.output[:] = ['fc']

        linear2 = model_proto.graph.node.add()
        linear2.name = 'linear2'
        linear2.op_type = 'MatMul'
        linear2.input[:] = ['fc', 'transpose']
        linear2.output[:] = ['fc2']

        weights = model_proto.graph.initializer.add()
        weights.name = 'fc.weights'
        weights.data_type = 3
        weights.int32_data[:] = [1, 2, 3, 4, 5, 6]
        weights.dims[:] = [1, 1, 2, 3]

        graph = Graph(model_proto)
        self.assertEqual(graph.get_node(3).name, 'linear2')


    def test_linear_with_transpose_not_matmul(self):
        model_proto = onnx_pb.ModelProto()

        graph_input0 = model_proto.graph.input.add()
        graph_input0.name = 'data'
        graph_input0.type.tensor_type.shape.dim.add().dim_value = 3
        graph_input0.type.tensor_type.shape.dim.add().dim_value = 3
        graph_input0.type.tensor_type.shape.dim.add().dim_value = 3
        graph_input0.type.tensor_type.shape.dim.add().dim_value = 3

        transpose = model_proto.graph.node.add()
        transpose.name = 'transpose'
        transpose.op_type = 'Transpose'
        transpose.input[:] = ['fc.weights']
        transpose.output[:] = ['transpose']

        linear = model_proto.graph.node.add()
        linear.name = 'linear'
        linear.op_type = 'Gemm'
        linear.input[:] = ['data', 'transpose']
        linear.output[:] = ['fc']

        weights = model_proto.graph.initializer.add()
        weights.name = 'fc.weights'
        weights.data_type = 3
        weights.int32_data[:] = [1, 2, 3, 4, 5, 6]
        weights.dims[:] = [1, 1, 2, 3]

        graph = Graph(model_proto)
        self.assertEqual(graph.get_node(2).type, 'Gemm')

    def test_modified_model_remove_node(self):
        test_model = deepcopy(self.model_proto)
        graph = Graph(test_model)

        relu2 = onnx_pb.NodeProto()
        relu2.name = 'relu2'
        relu2.op_type = 'Relu'
        relu2.input[:] = ['relu2_input']
        relu2.output[:] = ['relu2_output']

        avg_pool1_node = graph.get_node_by_name('avg_pool1')
        output_node = avg_pool1_node.get_output_anchor(0).get_peer_input_anchor()[0].node
        relu2_node = graph.add_node(relu2)
        graph.remove_edge(avg_pool1_node, 0, output_node, 0)
        graph.add_edge(avg_pool1_node, 0, relu2_node, 0)
        graph.add_edge(relu2_node, 0, output_node, 0)
        graph.topologic_sort()

        relu3 = onnx_pb.NodeProto()
        relu3.name = 'relu3'
        relu3.op_type = 'Relu'
        relu3.input[:] = ['relu3_input']
        relu3.output[:] = ['relu3_output']

        relu3_node = graph.add_node(relu3, 0)
        graph.remove_node(relu3_node)

        model_proto = graph.dump_proto()
        self.assertEqual(len(graph.nodes), 7)
        self.assertEqual(graph.get_node(0).name, 'conv1.weights')
        self.assertEqual(graph.get_node(1).name, 'conv1.bias')
        self.assertEqual(graph.get_node(2).name, 'conv1')
        self.assertEqual(graph.get_node(3).name, 'relu1')
        self.assertEqual(graph.get_node(4).name, 'add1')
        self.assertEqual(graph.get_node(5).name, 'avg_pool1')
        self.assertEqual(graph.get_node(6).name, 'relu2')
        model_proto = graph.dump_proto()
        model_proto.graph.node[0]


    def test_remove_node_not_found(self):
        test_model = deepcopy(self.model_proto)

        relu2 = onnx_pb.NodeProto()
        relu2.name = 'relu2'
        relu2.op_type = 'Relu'
        relu2.input[:] = ['relu2_input']
        relu2.output[:] = ['relu2_output']
        relu2_node = Node(0, relu2)

        graph = Graph(test_model)
        self.assertRaises(
            RuntimeError,
            graph.remove_node,
            relu2_node)

    def test_remove_input_anchor_failed(self):
        test_model = deepcopy(self.model_proto)
        graph = Graph(test_model)
        data_node = graph._in_out_nodes[0]
        self.assertRaises(
            RuntimeError,
            graph.remove_node,
            data_node)

    def test_dump_node_not_support(self):
        test_model = deepcopy(self.model_proto)
        graph = Graph(test_model)

        graph.get_node(1)._node_proto = onnx_pb.GraphProto()
        self.assertRaises(
            TypeError,
            graph.dump_proto)

    def test_remove_initializer_not_initializer(self):
        test_model = deepcopy(self.model_proto)
        graph = Graph(test_model)

        node = graph.get_node_by_name('conv1')
        # graph.remove_initializer(node)
        self.assertRaises(RuntimeError, graph.remove_initializer, node)

    def test_remove_initializer_linked(self):
        test_model = deepcopy(self.model_proto)
        graph = Graph(test_model)

        weight = graph.get_node_by_name('conv1.weights')
        graph.remove_initializer(weight)
        self.assertEqual(graph._nodes[0].name, 'conv1.weights')

    def test_delete_initializer_from_input(self):
        weight = np.random.randn(3,3,3,3).astype(np.float32).flatten()
        bias = np.random.randn(3).astype(np.float32)
        w = helper.make_tensor('conv1.weight', TensorProto.FLOAT, [3,3,3,3], weight)
        b = helper.make_tensor('conv1.bias', TensorProto.FLOAT, (3,), bias)
        initializer = [w, b]
        x = helper.make_tensor_value_info('input.1', TensorProto.FLOAT, (16,3,224,224))
        w_input = helper.make_tensor_value_info('conv1.weight', TensorProto.FLOAT, (3,3,3,3))
        b_input = helper.make_tensor_value_info('conv1.bias', TensorProto.FLOAT, (3,))
 
        identity_0_node = helper.make_node('Identity', inputs=['conv1.weight'], outputs=['conv2.weight'], name='identity_0')
        identity_1_node = helper.make_node('Identity', inputs=['conv1.bias'], outputs=['conv2.bias'], name='identity_1')
 
        conv1 = helper.make_node('Conv', inputs=['input.1', 'conv1.weight', 'conv1.bias'], outputs=['conv1.output'], name='conv1')
        conv2 = helper.make_node('Conv', inputs=['conv1.output', 'conv2.weight', 'conv2.bias'], outputs=['conv2.output'], name='conv2')
        y = helper.make_tensor_value_info('conv2.output', TensorProto.FLOAT, [3,3,3,3])
        graph_def = helper.make_graph((identity_0_node, identity_1_node, conv1, conv2),
                                  'model',
                                  [x, w_input, b_input],
                                  [y,],
                                  initializer=initializer)
        mode_def = helper.make_model(graph_def, opset_imports=[
                                     helper.make_opsetid("", 12)])
        graph = Graph(mode_def)
        err_flag = False
        for graph_input in graph._net.graph.input:
            if 'conv1' in graph_input.name:
                err_flag = True
        self.assertFalse(err_flag)

    def test_parse_unsqueeze_nodes(self):
        weight = np.random.randn(3,3,3,3).astype(np.float32).flatten()
        bias = np.random.randn(3).astype(np.float32)
        w = helper.make_tensor('conv1.weight', TensorProto.FLOAT, [3,3,3,3], weight)
        b = helper.make_tensor('conv1.bias', TensorProto.FLOAT, (3,), bias)
        initializer = [w, b]
        x = helper.make_tensor_value_info('input.1', TensorProto.FLOAT, (16,3,224,224))

        identity_node = helper.make_node('Identity', inputs=['input.1'], outputs=['identity_0'], name='identity_0')

        node_unsquezee_1 = helper.make_node(op_type="Unsqueeze", name="unsquezee_1", inputs=['identity_0'], outputs=['unsquezee_1.output'])
        conv1 = helper.make_node('Conv', inputs=["unsquezee_1.output", 'conv1.weight', 'conv1.bias'], outputs=['conv1.output'], name='conv1')
        y = helper.make_tensor_value_info('conv1.output', TensorProto.FLOAT, [3,3,3,3])
        graph_def = helper.make_graph((identity_node, node_unsquezee_1, conv1),
                                  'model',
                                  [x,],
                                  [y,],
                                  initializer=initializer)
        mode_def = helper.make_model(graph_def, opset_imports=[
                                     helper.make_opsetid("", 12)])
        graph = Graph(mode_def)
        for node in graph._nodes:
            if node.type == 'Identity':
                conv_node = Graph._parse_unsqueeze_nodes(node)
                self.assertTrue(conv_node.get_attr('input_dimension_reduction'))
                break
