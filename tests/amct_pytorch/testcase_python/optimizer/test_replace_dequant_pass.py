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

from amct_pytorch.graph_based_compression.amct_pytorch.optimizer.insert_dequant_pass import construct_dequant_node
from amct_pytorch.graph_based_compression.amct_pytorch.optimizer.insert_quant_pass import construct_quant_node
from amct_pytorch.graph_based_compression.amct_pytorch.optimizer.replace_dequant_pass import ReplaceDequantPass
from amct_pytorch.graph_based_compression.amct_pytorch.optimizer.insert_quant_pass import InsertQuantPass
from amct_pytorch.graph_based_compression.amct_pytorch.optimizer.insert_dequant_pass import InsertDequantPass
from amct_pytorch.graph_based_compression.amct_pytorch.optimizer.graph_optimizer import GraphOptimizer
from amct_pytorch.graph_based_compression.amct_pytorch.parser.parser import Parser
from amct_pytorch.graph_based_compression.amct_pytorch.parser.parse_record_file import RecordFileParser
from amct_pytorch.graph_based_compression.amct_pytorch.configuration.configuration import Configuration
from amct_pytorch.graph_based_compression.amct_pytorch.common.utils import files as files_util
from amct_pytorch.graph_based_compression.amct_pytorch.proto import scale_offset_record_pb2
from google.protobuf import text_format

CUR_DIR = os.path.split(os.path.realpath(__file__))[0]

class TestReplaceDequantPass(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass
    @classmethod
    def setUpClass(self):
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
            # Add Ascend DeQuant
            dequant_node = graph.node.add()
            dequant_node.CopyFrom(construct_dequant_node([conv_name] + ['%s.dequant.param' % (conv_name)], ['%s_dequant' % (conv_name)],
                conv_name, 0))
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
            # Add conv1.dequant.param
            dequant_param = graph.initializer.add()
            dequant_param.name = '%s.dequant.param' % (conv_name)
            dequant_param.data_type = 13
            dequant_param.uint64_data[:] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
            dequant_param.dims[:] = [16]
            # Add relu
            relu1 = graph.node.add()
            relu1.name = '%s.relu' % (conv_name)
            relu1.op_type = 'Relu'
            relu1.input[:] = ['%s_dequant' % (conv_name)]
            relu1.output[:] = outputs
        conv_sub(self.graph, 'conv1', ['concat0'], ['conv1_output'], {'scale': 1, 'offset': 0, 'quant_bit': 8, 'dst_type': 'INT8'})
        conv_sub(self.graph, 'conv2', ['concat0'], ['conv2_output'], {'scale': 1, 'offset': 0, 'quant_bit': 8, 'dst_type': 'INT8'})
        conv_sub(self.graph, 'conv3', ['concat0'], ['conv3_output'], {'scale': 1, 'offset': 0, 'quant_bit': 8, 'dst_type': 'INT8'})
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
        add1.input[:] = ['conv1_output', 'conv2_output', 'conv3_output', 'pool0']
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
        kernel_shape = avg_pool1.attribute.add()
        kernel_shape.name = 'kernel_shape'
        kernel_shape.type = onnx_pb.AttributeProto.AttributeType.INTS
        kernel_shape.ints[:] = [3, 3]
        avg_pool1.op_type = 'AveragePool'
        avg_pool1.input[:] = ['pad0']
        avg_pool1.output[:] = ['avg_pool1']
        # Add Ascend DeQuant
        dequant_node = self.graph.node.add()
        dequant_node.CopyFrom(construct_dequant_node(['avg_pool1'] + ['avg_pool1.dequant.param'], ['output'],
                                                     'avg_pool1', 0))
        # Add conv1.dequant.param
        dequant_param = self.graph.initializer.add()
        dequant_param.name = 'avg_pool1.dequant.param'
        dequant_param.data_type = 13
        dequant_param.uint64_data[:] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        dequant_param.dims[:] = [16]
        # add output
        graph_output = self.graph.output.add()
        graph_output.name = 'output'
        graph_output.type.tensor_type.shape.dim.add().dim_value = 1
        self.model_proto.graph.CopyFrom(self.graph)

    @classmethod
    def tearDownClass(self):
        print("[UNITTEST END replace_dequant_pass.py]")

    def test_do_pass_success(self):
        records = {'conv1': {'data_scale': 1,
                             'data_offset': 0,
                             'weight_scale': np.array([1]),
                             'weight_offset': np.array([0])}}
        test_model = deepcopy(self.model_proto)
        graph = Graph(test_model)
        conv_node = graph.get_node_by_name('conv1.dequant')
        before_nodes = len(graph.nodes)
        ReplaceDequantPass(records).do_pass(graph, conv_node)
        after_nodes = len(graph.nodes)
        self.assertEqual(after_nodes - before_nodes, 0)

    def test_do_pass_average_pool_success(self):
        records = {'avg_pool1': {'data_scale': 1,
                             'data_offset': 0,
                             'weight_scale': np.array([1]),
                             'weight_offset': np.array([0])}}
        test_model = deepcopy(self.model_proto)
        graph = Graph(test_model)
        before_nodes = len(graph.nodes)
        conv_node = graph.get_node_by_name('avg_pool1.dequant')
        ReplaceDequantPass(records).do_pass(graph, conv_node)
        after_nodes = len(graph.nodes)
        self.assertEqual(after_nodes - before_nodes, 0)

    def test_match_pattern_success(self):
        records = {'conv1': {'data_scale': 1, 'data_offset': 0},
                   'conv2': {'data_scale': 1, 'data_offset': 0},
                   'conv3': {'data_scale': 1, 'data_offset': 0}}
        test_model = deepcopy(self.model_proto)
        graph = Graph(test_model)
        antiquant_node = graph.get_node_by_name('conv1.dequant')
        self.assertTrue(ReplaceDequantPass(records).match_pattern(antiquant_node))

    def test_match_pattern_false(self):
        records = {'conv1': {'data_scale': 1, 'data_offset': 0},
                   'conv2': {'data_scale': 1, 'data_offset': 0},
                   'conv3': {'data_scale': 1, 'data_offset': 0}}
        test_model = deepcopy(self.model_proto)
        graph = Graph(test_model)
        antiquant_node = graph.get_node_by_name('concat0')
        self.assertFalse(ReplaceDequantPass(records).match_pattern(antiquant_node))
