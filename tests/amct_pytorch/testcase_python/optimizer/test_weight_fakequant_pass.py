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
import sys
import unittest
import torch
from unittest.mock import patch
import numpy as np

import amct_pytorch.graph_based_compression.amct_pytorch as amct
from .util import models
from .util import record_file
from onnx import onnx_pb
from copy import deepcopy
from amct_pytorch.graph_based_compression.amct_pytorch.graph.graph import Graph

from amct_pytorch.graph_based_compression.amct_pytorch.optimizer.weight_fakequant_pass import WeightFakequantPass
from amct_pytorch.graph_based_compression.amct_pytorch.optimizer.insert_weight_quant_pass import InsertWeightQuantPass
from amct_pytorch.graph_based_compression.amct_pytorch.optimizer.replace_weight_quant_pass import ReplaceWeightQuantPass
from amct_pytorch.graph_based_compression.amct_pytorch.optimizer.insert_quant_pass import construct_quant_node
from amct_pytorch.graph_based_compression.amct_pytorch.optimizer.graph_optimizer import GraphOptimizer
from amct_pytorch.graph_based_compression.amct_pytorch.parser.parser import Parser
from amct_pytorch.graph_based_compression.amct_pytorch.parser.parse_record_file import RecordFileParser
from amct_pytorch.graph_based_compression.amct_pytorch.configuration.configuration import Configuration
from amct_pytorch.graph_based_compression.amct_pytorch.common.utils import files as files_util
from amct_pytorch.graph_based_compression.amct_pytorch.proto import scale_offset_record_pb2
from amct_pytorch.graph_based_compression.amct_pytorch.utils.onnx_initializer_util import TensorProtoHelper
from google.protobuf import text_format

from amct_pytorch.graph_based_compression.amct_pytorch.utils.vars import QUANTIZABLE_TYPES

def conv_sub_with_trans(graph, conv_name, inputs, outputs, quant_attrs):
    # Add Ascend Quant
    quant_node = graph.node.add()
    quant_node.CopyFrom(construct_quant_node(inputs, ['%s_quant' % (conv_name)],
        quant_attrs, conv_name))
    # Add conv
    conv = graph.node.add()
    conv.name = conv_name
    conv.op_type = 'Conv'
    conv.input[:] = ['%s_quant' % (conv_name), '%s.transpose' % (conv_name), '%s.bias' % (conv_name)]
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
    weights.data_type = 6
    weights.int32_data[:] = [1, 2, 3, 4, 5, 6]
    weights.dims[:] = [6]
    # Add Transpose
    transpose = graph.node.add()
    transpose.name = '%s.transpose' % (conv_name)
    transpose.op_type = 'Transpose'
    transpose.input[:] = ['%s.weights' % (conv_name)]
    transpose.output[:] = ['%s.transpose' % (conv_name)]
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
    relu1.input[:] = ['%s' % (conv_name)]
    relu1.output[:] = outputs

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
    weights.data_type = 6
    weights.int32_data[:] = [1, 2, 3, 4, 5, 6]
    weights.dims[:] = [6]
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
    relu1.input[:] = ['%s' % (conv_name)]
    relu1.output[:] = outputs

def deconv_sub(graph, deconv_name, inputs, outputs, quant_attrs):
    # Add Ascend Quant
    quant_node = graph.node.add()
    quant_node.CopyFrom(construct_quant_node(inputs, ['%s_quant' % (deconv_name)],
        quant_attrs, deconv_name))
    # Add deconv
    deconv = graph.node.add()
    deconv.name = deconv_name
    deconv.op_type = 'ConvTranspose'
    deconv.input[:] = ['%s_quant' % (deconv_name), '%s.weights' % (deconv_name), '%s.bias' % (deconv_name)]
    deconv.output[:] = [deconv_name]
    # add attribute "kernel_shape"
    kernel_shape = deconv.attribute.add()
    kernel_shape.name = 'kernel_shape'
    kernel_shape.type = onnx_pb.AttributeProto.AttributeType.INTS
    kernel_shape.ints[:] = [3, 3]
    # add attribute "pads"
    pads = deconv.attribute.add()
    pads.name = 'pads'
    pads.type = onnx_pb.AttributeProto.AttributeType.INTS
    pads.ints[:] = [0, 0, 0, 0]
    # add attribute "group"
    group = deconv.attribute.add()
    group.name = 'group'
    group.type = onnx_pb.AttributeProto.AttributeType.INT
    group.i = 3
    # Add weights
    weights = graph.initializer.add()
    weights.name = '%s.weights' % (deconv_name)
    weights.data_type = 6
    weights.int32_data[:] = [1] * 81
    weights.dims[:] = [3,3,3,3]
    # Add bias
    bias = graph.initializer.add()
    bias.name = '%s.bias' % (deconv_name)
    bias.data_type = 6
    bias.int32_data[:] = [0]
    bias.dims[:] = [1]
    # Add relu
    relu1 = graph.node.add()
    relu1.name = '%s.relu' % (deconv_name)
    relu1.op_type = 'Relu'
    relu1.input[:] = ['%s' % (deconv_name)]
    relu1.output[:] = outputs

class TestReplaceWeightQuantPass(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass
    @classmethod
    def setUpClass(self):
        QUANTIZABLE_TYPES.extend(['ConvTranspose2d','AvgPool2d'])
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
        # Add concat node
        concat_node = self.graph.node.add()
        concat_node.name = 'concat0'
        concat_node.op_type = 'Concat'
        concat_node.input[:] = ['data0']
        concat_node.output[:] = ['concat0']

        conv_sub(self.graph, 'conv1', ['concat0'], ['conv1_output'], {'scale': 1, 'offset': 0, 'quant_bit': 8, 'dst_type': 'INT8'})
        conv_sub(self.graph, 'conv2', ['concat0'], ['conv2_output'], {'scale': 1, 'offset': 0, 'quant_bit': 8, 'dst_type': 'INT8'})
        conv_sub_with_trans(self.graph, 'conv3', ['concat0'], ['conv3_output'], {'scale': 1, 'offset': 0, 'quant_bit': 8, 'dst_type': 'INT8'})
        deconv_sub(self.graph, 'deconv1', ['concat0'], ['deconv1_output'], {'scale': 1, 'offset': 0, 'quant_bit': 8, 'dst_type': 'INT8'})

        # Add max_pooling
        pool0 = self.graph.node.add()
        pool0.name = 'pool0'
        pool0.op_type = 'MaxPool'
        pool0.input[:] = ['concat0']
        pool0.output[:] = ['pool0']
        # Add add
        add1 = self.graph.node.add()
        add1.name = 'add1'
        add1.op_type = 'Add'
        add1.input[:] = ['conv1_output', 'conv2_output', 'conv3_output', 'deconv1_output', 'pool0']
        add1.output[:] = ['add1']
        # Add average_pool
        pad0 = self.graph.node.add()
        pad0.name = 'avg_pool1_pad'
        pad0.op_type = 'Pad'
        pad0.input[:] = ['add1']
        pad0.output[:] = ['pad0']
        # Add average_pool
        avg_pool1 = self.graph.node.add()
        avg_pool1.name = 'avg_pool1'
        avg_pool1.op_type = 'AveragePool'
        avg_pool1.input[:] = ['pad0']
        avg_pool1.output[:] = ['output']
        # add output
        graph_output = self.graph.output.add()
        graph_output.name = 'output'
        graph_output.type.tensor_type.shape.dim.add().dim_value = 1
        self.model_proto.graph.CopyFrom(self.graph)

    @classmethod
    def tearDownClass(self):
        QUANTIZABLE_TYPES.remove('ConvTranspose2d')
        QUANTIZABLE_TYPES.remove('AvgPool2d')
        print("[UNITTEST END replace_quant_pass.py]")

    def test_do_pass_int4_success(self):
        with patch('amct_pytorch.graph_based_compression.amct_pytorch.utils.quant_node.QuantOpInfo.get_dst_num_bits', return_value=4):
            with patch('amct_pytorch.graph_based_compression.amct_pytorch.utils.quant_node.QuantOpInfo.get_dst_num_bits', return_value=4):
                records = {'conv1': {
                        'data_scale': 1.0,
                        'data_offset': 0,
                        'weight_scale': np.array([1.0], dtype=np.float32),
                        'weight_offset': np.array([0], dtype=np.int8),
                    }
                }
                test_model = deepcopy(self.model_proto)
                graph = Graph(test_model)
                optimizer = GraphOptimizer()
                optimizer.add_pass(InsertWeightQuantPass(records))
                optimizer.add_pass(ReplaceWeightQuantPass(records))
                optimizer.do_optimizer(graph, test_model)
                self.assertEqual(TensorProtoHelper(graph.nodes[0].proto).get_data().dtype, 'float32')

    def test_do_pass_int4_trans_success(self):
        with patch('amct_pytorch.graph_based_compression.amct_pytorch.utils.quant_node.QuantOpInfo.get_dst_num_bits', return_value=4):
            with patch('amct_pytorch.graph_based_compression.amct_pytorch.utils.quant_node.QuantOpInfo.get_dst_num_bits', return_value=4):
                records = {'conv3': {
                        'data_scale': 1.0,
                        'data_offset': 0,
                        'weight_scale': np.array([1.0], dtype=np.float32),
                        'weight_offset': np.array([0], dtype=np.int8),
                    }
                }
                test_model = deepcopy(self.model_proto)
                graph = Graph(test_model)
                optimizer = GraphOptimizer()
                optimizer.add_pass(InsertWeightQuantPass(records))
                optimizer.add_pass(ReplaceWeightQuantPass(records))
                optimizer.do_optimizer(graph, test_model)
                self.assertEqual(TensorProtoHelper(graph.get_node_by_name('conv3.weights').proto).get_data().dtype, 'float32')
                

    def test_do_pass_int4_deconv_success(self):
        with patch('amct_pytorch.graph_based_compression.amct_pytorch.utils.quant_node.QuantOpInfo.get_dst_num_bits', return_value=4):
            with patch('amct_pytorch.graph_based_compression.amct_pytorch.utils.quant_node.QuantOpInfo.get_dst_num_bits', return_value=4):
                records = {'deconv1': {
                        'data_scale': 1.0,
                        'data_offset': 0,
                        'weight_scale': np.array([1.0], dtype=np.float32),
                        'weight_offset': np.array([0], dtype=np.int8),
                    }
                }
                test_model = deepcopy(self.model_proto)
                graph = Graph(test_model)
                optimizer = GraphOptimizer()
                optimizer.add_pass(InsertWeightQuantPass(records))
                optimizer.add_pass(ReplaceWeightQuantPass(records))
                optimizer.do_optimizer(graph, test_model)
                self.assertEqual(TensorProtoHelper(graph.get_node_by_name(
                    'deconv1.weights').proto).get_data().dtype, 'float32')

    def test_match_pattern_success(self):
        records = {
            'conv1': {'data_scale': 1, 'data_offset': 0},
            'conv2': {'data_scale': 1, 'data_offset': 0},
            'conv3': {'data_scale': 1, 'data_offset': 0},
        }
        test_model = deepcopy(self.model_proto)
        graph = Graph(test_model)
        conv_node = graph.get_node_by_name('conv1')
        self.assertTrue(InsertWeightQuantPass(records).match_pattern(conv_node))

    def test_match_pattern_false0(self):
        records = {
            'conv1': {'data_scale': 1, 'data_offset': 0},
            'conv2': {'data_scale': 1, 'data_offset': 0},
        }
        test_model = deepcopy(self.model_proto)
        graph = Graph(test_model)
        conv_node = graph.get_node_by_name('conv3')
        self.assertFalse(InsertWeightQuantPass(records).match_pattern(conv_node))

    def test_match_pattern_false1(self):
        records = {
            'conv1': {'data_scale': 1, 'data_offset': 0},
            'conv2': {'data_scale': 1, 'data_offset': 0},
        }
        test_model = deepcopy(self.model_proto)
        graph = Graph(test_model)
        conv_node = graph.get_node_by_name('concat0')
        self.assertFalse(InsertWeightQuantPass(records).match_pattern(conv_node))

    def test_match_pattern_false2(self):
        records = {
            'conv1': {'data_scale': 1, 'data_offset': 0},
            'conv2': {'data_scale': 1, 'data_offset': 0},
        }
        test_model = deepcopy(self.model_proto)
        graph = Graph(test_model)
        avg_pool1_node = graph.get_node_by_name('avg_pool1')
        self.assertFalse(InsertWeightQuantPass(records).match_pattern(avg_pool1_node))

    def test_rnn_weight_fake_quant_success(self):
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
        passer = WeightFakequantPass(records)
        passer.dequant_weight(node)
        passer.dequant_weight(node, is_recurrence_weight=True)

    def test_conv_1d_weight_fake_quant_success(self):
        class Conv1dModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = torch.nn.Conv1d(3, 3, 3)
                self.deconv1 = torch.nn.ConvTranspose1d(3, 3, 3)
            def forward(self, input):
                x = self.conv1(input)
                x = self.deconv1(x)
                return x
        model = Conv1dModule()
        tmp_onnx = BytesIO()
        Parser.export_onnx(model, torch.randn(3, 3, 10), tmp_onnx)
        graph = Parser.parse_net_to_graph(tmp_onnx)

        records = {
            'conv1': {
                'weight_scale': np.array([1.0]*3, dtype=np.float32),
                'weight_offset': np.array([0]*3, dtype=np.int8)
            },
            'deconv1': {
                'weight_scale': np.array([1.0]*3, dtype=np.float32),
                'weight_offset': np.array([0]*3, dtype=np.int8)
            }
        }

        node_name = 'deconv1'
        node = graph.get_node_by_name(node_name)

        passer = WeightFakequantPass(records)
        passer.do_pass(graph, node)

        node_name = 'conv1'
        node = graph.get_node_by_name(node_name)
        passer.do_pass(graph, node)
