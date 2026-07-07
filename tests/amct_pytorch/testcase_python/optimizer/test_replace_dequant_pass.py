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
import logging
import os
import sys
import unittest
from copy import deepcopy
from unittest.mock import patch

import numpy as np
import torch
from google.protobuf import text_format
from onnx import onnx_pb

import amct_pytorch.classic.graph_based.amct_pytorch as amct
from amct_pytorch.classic.graph_based.amct_pytorch.common.utils import (
    files as files_util,
)
from amct_pytorch.classic.graph_based.amct_pytorch.configuration.configuration import (
    Configuration,
)
from amct_pytorch.classic.graph_based.amct_pytorch.graph.graph import Graph
from amct_pytorch.classic.graph_based.amct_pytorch.optimizer.graph_optimizer import (
    GraphOptimizer,
)
from amct_pytorch.classic.graph_based.amct_pytorch.optimizer.insert_dequant_pass import (
    InsertDequantPass,
    construct_dequant_node,
)
from amct_pytorch.classic.graph_based.amct_pytorch.optimizer.insert_quant_pass import (
    InsertQuantPass,
    construct_quant_node,
)
from amct_pytorch.classic.graph_based.amct_pytorch.optimizer.replace_dequant_pass import (
    ReplaceDequantPass,
)
from amct_pytorch.classic.graph_based.amct_pytorch.parser.parse_record_file import (
    RecordFileParser,
)
from amct_pytorch.classic.graph_based.amct_pytorch.parser.parser import Parser
from amct_pytorch.classic.graph_based.amct_pytorch.proto import (
    scale_offset_record_pb2,
)

from .util import models, record_file

logger = logging.getLogger(__name__)

DATA_SCALE = 'data_scale'
DATA_OFFSET = 'data_offset'

CUR_DIR = os.path.split(os.path.realpath(__file__))[0]

CONCAT0 = 'concat0'

SCALE = 'scale'

OFFSET = 'offset'

QUANT_BIT = 'quant_bit'

DST_TYPE = 'dst_type'

INT8 = 'INT8'

POOL0 = 'pool0'

ADD1 = 'add1'

AVG_POOL1 = 'avg_pool1'


class TestReplaceDequantPass(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # set basic info
        cls.model_proto = onnx_pb.ModelProto()
        cls.model_proto.producer_name = 'model'
        cls.graph = onnx_pb.GraphProto()
        # Add graph input 0
        graph_input0 = cls.graph.input.add()
        graph_input0.name = 'data0'
        graph_input0.type.tensor_type.shape.dim.add().dim_value = 3
        graph_input0.type.tensor_type.shape.dim.add().dim_value = 3
        graph_input0.type.tensor_type.shape.dim.add().dim_value = 3
        graph_input0.type.tensor_type.shape.dim.add().dim_value = 3
        # Add concat node
        concat_node = cls.graph.node.add()
        concat_node.name = CONCAT0
        concat_node.op_type = 'Concat'
        concat_node.input[:] = ['data0']
        concat_node.output[:] = [CONCAT0]

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
            dequant_inputs = [conv_name] + ['%s.dequant.param' % (conv_name)]
            dequant_outputs = ['%s_dequant' % (conv_name)]
            dequant_node.CopyFrom(
                construct_dequant_node(dequant_inputs, dequant_outputs, conv_name, 0))
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
        conv_sub(cls.graph, 'conv1', [CONCAT0], ['conv1_output'], {SCALE: 1, OFFSET: 0, QUANT_BIT: 8, DST_TYPE: INT8})
        conv_sub(cls.graph, 'conv2', [CONCAT0], ['conv2_output'], {SCALE: 1, OFFSET: 0, QUANT_BIT: 8, DST_TYPE: INT8})
        conv_sub(cls.graph, 'conv3', [CONCAT0], ['conv3_output'], {SCALE: 1, OFFSET: 0, QUANT_BIT: 8, DST_TYPE: INT8})
        # Add max_pooling
        pool0 = cls.graph.node.add()
        pool0.name = POOL0
        pool0.op_type = 'MaxPool'
        pool0.input[:] = [CONCAT0]
        pool0.output[:] = [POOL0]
        # Add add
        add1 = cls.graph.node.add()
        add1.name = ADD1
        add1.op_type = 'Add'
        add1.input[:] = ['conv1_output', 'conv2_output', 'conv3_output', POOL0]
        add1.output[:] = [ADD1]
        # Add average_pool
        pad0 = cls.graph.node.add()
        pad0.name = 'avg_pool1_pad'
        pad0.op_type = 'Pad'
        pad0.input[:] = [ADD1]
        pad0.output[:] = ['pad0']
        cls.add_data()

    @classmethod
    def add_data(cls):
        # Add average_pool
        avg_pool1 = cls.graph.node.add()
        avg_pool1.name = AVG_POOL1
        kernel_shape = avg_pool1.attribute.add()
        kernel_shape.name = 'kernel_shape'
        kernel_shape.type = onnx_pb.AttributeProto.AttributeType.INTS
        kernel_shape.ints[:] = [3, 3]
        avg_pool1.op_type = 'AveragePool'
        avg_pool1.input[:] = ['pad0']
        avg_pool1.output[:] = [AVG_POOL1]
        # Add Ascend DeQuant
        dequant_node = cls.graph.node.add()
        dequant_node.CopyFrom(construct_dequant_node([AVG_POOL1] + ['avg_pool1.dequant.param'], ['output'],
                                                      AVG_POOL1, 0))
        # Add conv1.dequant.param
        dequant_param = cls.graph.initializer.add()
        dequant_param.name = 'avg_pool1.dequant.param'
        dequant_param.data_type = 13
        dequant_param.uint64_data[:] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        dequant_param.dims[:] = [16]
        # add output
        graph_output = cls.graph.output.add()
        graph_output.name = 'output'
        graph_output.type.tensor_type.shape.dim.add().dim_value = 1
        cls.model_proto.graph.CopyFrom(cls.graph)

    @classmethod
    def tearDownClass(cls):
        logger.info("[UNITTEST END replace_dequant_pass.py]")

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_do_pass_success(self):
        records = {'conv1': {DATA_SCALE: 1,
                             DATA_OFFSET: 0,
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
        records = {AVG_POOL1: {DATA_SCALE: 1,
                             DATA_OFFSET: 0,
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
        records = {'conv1': {DATA_SCALE: 1, DATA_OFFSET: 0},
                   'conv2': {DATA_SCALE: 1, DATA_OFFSET: 0},
                   'conv3': {DATA_SCALE: 1, DATA_OFFSET: 0}}
        test_model = deepcopy(self.model_proto)
        graph = Graph(test_model)
        antiquant_node = graph.get_node_by_name('conv1.dequant')
        self.assertTrue(ReplaceDequantPass(records).match_pattern(antiquant_node))

    def test_match_pattern_false(self):
        records = {'conv1': {DATA_SCALE: 1, DATA_OFFSET: 0},
                   'conv2': {DATA_SCALE: 1, DATA_OFFSET: 0},
                   'conv3': {DATA_SCALE: 1, DATA_OFFSET: 0}}
        test_model = deepcopy(self.model_proto)
        graph = Graph(test_model)
        antiquant_node = graph.get_node_by_name(CONCAT0)
        self.assertFalse(ReplaceDequantPass(records).match_pattern(antiquant_node))

