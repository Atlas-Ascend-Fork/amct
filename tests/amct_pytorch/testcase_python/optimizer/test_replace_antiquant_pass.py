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
from copy import deepcopy

import amct_pytorch.amct_pytorch_inner.amct_pytorch as amct
from .util import models
from .util import record_file

from amct_pytorch.amct_pytorch_inner.amct_pytorch.graph.graph import Graph
from onnx import onnx_pb
from amct_pytorch.amct_pytorch_inner.amct_pytorch.optimizer.replace_anti_quant_pass import ReplaceAntiQuantPass
from amct_pytorch.amct_pytorch_inner.amct_pytorch.optimizer.mult_output_with_quant_optimizer import construct_anti_quant_node
from amct_pytorch.amct_pytorch_inner.amct_pytorch.optimizer.insert_quant_pass import construct_quant_node
from amct_pytorch.amct_pytorch_inner.amct_pytorch.optimizer.mult_output_with_quant_optimizer import \
    MultQuantOptimizerPass
from amct_pytorch.amct_pytorch_inner.amct_pytorch.optimizer.quant_fusion_pass import QuantFusionPass
from amct_pytorch.amct_pytorch_inner.amct_pytorch.optimizer.graph_optimizer import GraphOptimizer
from amct_pytorch.amct_pytorch_inner.amct_pytorch.parser.parser import Parser
from amct_pytorch.amct_pytorch_inner.amct_pytorch.parser.parse_record_file import RecordFileParser
from amct_pytorch.amct_pytorch_inner.amct_pytorch.configuration.configuration import Configuration
from amct_pytorch.amct_pytorch_inner.amct_pytorch.common.utils import files as files_util
from amct_pytorch.amct_pytorch_inner.amct_pytorch.proto import scale_offset_record_pb2
from google.protobuf import text_format

CUR_DIR = os.path.split(os.path.realpath(__file__))[0]

class TestReplaceAntiQuantPass(unittest.TestCase):
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
        conv_sub(self.graph, 'conv1', ['concat0'], ['conv1_output'], {'scale': 1, 'offset': 0, 'quant_bit': 8, 'dst_type': 'INT8'})
        conv_sub(self.graph, 'conv2', ['concat0'], ['conv2_output'], {'scale': 1, 'offset': 0, 'quant_bit': 8, 'dst_type': 'INT8'})
        conv_sub(self.graph, 'conv3', ['concat0'], ['conv3_output'], {'scale': 1, 'offset': 0, 'quant_bit': 8, 'dst_type': 'INT8'})
        # Add AntiAscend Quant
        antiquant0 = self.graph.node.add()
        antiquant0.CopyFrom(construct_anti_quant_node(['concat0'], ['antiquant0'],
            {'scale': 1, 'offset': 0, 'quant_bit': 8, 'dst_type': 'INT8'}, 'antiquant0'))
        # Add max_pooling
        pool0 = self.graph.node.add()
        pool0.name = 'pool0'
        pool0.op_type = 'MaxPool'
        pool0.input[:] = ['antiquant0']
        pool0.output[:] = ['pool0']
        # Add add
        add1 = self.graph.node.add()
        add1.name = 'add1'
        add1.op_type = 'Add'
        add1.input[:] = ['conv1_output', 'conv2_output', 'conv3_output', 'pool0']
        add1.output[:] = ['output']
        # add output
        graph_output = self.graph.output.add()
        graph_output.name = 'output'
        graph_output.type.tensor_type.shape.dim.add().dim_value = 1
        self.model_proto.graph.CopyFrom(self.graph)

    @classmethod
    def tearDownClass(self):
        print("[UNITTEST END replace_antiquant_pass.py]")

    def test_do_pass_success(self):
        records = {
            'conv1': {'data_scale': 1, 'data_offset': 0},
            'conv2': {'data_scale': 1, 'data_offset': 0},
            'conv3': {'data_scale': 1, 'data_offset': 0},
        }
        numbits = 8
        test_model = deepcopy(self.model_proto)
        graph = Graph(test_model)
        data_node = graph.get_node_by_name('antiquant0.anti_quant')
        graph.get_node_by_name('conv1.quant').set_attr('object_node', 'conv1')
        graph.get_node_by_name('conv2.quant').set_attr('object_node', 'conv2')
        graph.get_node_by_name('conv3.quant').set_attr('object_node', 'conv3')

        ReplaceAntiQuantPass(records).do_pass(graph, data_node)

    def test_match_pattern_success(self):
        records = {
            'conv1': {'data_scale': 1, 'data_offset': 0},
            'conv2': {'data_scale': 1, 'data_offset': 0},
            'conv3': {'data_scale': 1, 'data_offset': 0},
        }
        test_model = deepcopy(self.model_proto)
        graph = Graph(test_model)
        antiquant_node = graph.get_node_by_name('antiquant0.anti_quant')
        self.assertTrue(ReplaceAntiQuantPass(records).match_pattern(antiquant_node))

    def test_match_pattern_false(self):
        records = {
            'conv1': {'data_scale': 1, 'data_offset': 0},
            'conv2': {'data_scale': 1, 'data_offset': 0},
            'conv3': {'data_scale': 1, 'data_offset': 0},
        }
        test_model = deepcopy(self.model_proto)
        graph = Graph(test_model)
        antiquant_node = graph.get_node_by_name('concat0')
        self.assertFalse(ReplaceAntiQuantPass(records).match_pattern(antiquant_node))
