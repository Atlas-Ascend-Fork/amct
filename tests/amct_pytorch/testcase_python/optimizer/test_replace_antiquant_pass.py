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
from amct_pytorch.classic.graph_based.amct_pytorch.optimizer.insert_quant_pass import (
    construct_quant_node,
)
from amct_pytorch.classic.graph_based.amct_pytorch.optimizer.mult_output_with_quant_optimizer import (
    MultQuantOptimizerPass,
    construct_anti_quant_node,
)
from amct_pytorch.classic.graph_based.amct_pytorch.optimizer.quant_fusion_pass import (
    QuantFusionPass,
)
from amct_pytorch.classic.graph_based.amct_pytorch.optimizer.replace_anti_quant_pass import (
    ReplaceAntiQuantPass,
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
OBJECT_NODE = 'object_node'

SCALE = 'scale'

OFFSET = 'offset'

QUANT_BIT = 'quant_bit'

DST_TYPE = 'dst_type'

INT8 = 'INT8'

ANTIQUANT0 = 'antiquant0'

POOL0 = 'pool0'


class TestReplaceAntiQuantPass(unittest.TestCase):
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
        conv_sub(cls.graph, 'conv1', [CONCAT0], ['conv1_output'], {SCALE: 1, OFFSET: 0, QUANT_BIT: 8, DST_TYPE: INT8})
        conv_sub(cls.graph, 'conv2', [CONCAT0], ['conv2_output'], {SCALE: 1, OFFSET: 0, QUANT_BIT: 8, DST_TYPE: INT8})
        conv_sub(cls.graph, 'conv3', [CONCAT0], ['conv3_output'], {SCALE: 1, OFFSET: 0, QUANT_BIT: 8, DST_TYPE: INT8})
        # Add AntiAscend Quant
        antiquant0 = cls.graph.node.add()
        antiquant0.CopyFrom(construct_anti_quant_node([CONCAT0], [ANTIQUANT0],
            {SCALE: 1, OFFSET: 0, QUANT_BIT: 8, DST_TYPE: INT8}, ANTIQUANT0))
        # Add max_pooling
        pool0 = cls.graph.node.add()
        pool0.name = POOL0
        pool0.op_type = 'MaxPool'
        pool0.input[:] = [ANTIQUANT0]
        pool0.output[:] = [POOL0]
        # Add add
        add1 = cls.graph.node.add()
        add1.name = 'add1'
        add1.op_type = 'Add'
        add1.input[:] = ['conv1_output', 'conv2_output', 'conv3_output', POOL0]
        add1.output[:] = ['output']
        # add output
        graph_output = cls.graph.output.add()
        graph_output.name = 'output'
        graph_output.type.tensor_type.shape.dim.add().dim_value = 1
        cls.model_proto.graph.CopyFrom(cls.graph)

    @classmethod
    def tearDownClass(cls):
        logger.info("[UNITTEST END replace_antiquant_pass.py]")

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_do_pass_success(self):
        records = {
            'conv1': {DATA_SCALE: 1, DATA_OFFSET: 0},
            'conv2': {DATA_SCALE: 1, DATA_OFFSET: 0},
            'conv3': {DATA_SCALE: 1, DATA_OFFSET: 0},
        }
        numbits = 8
        test_model = deepcopy(self.model_proto)
        graph = Graph(test_model)
        data_node = graph.get_node_by_name('antiquant0.anti_quant')
        graph.get_node_by_name('conv1.quant').set_attr(OBJECT_NODE, 'conv1')
        graph.get_node_by_name('conv2.quant').set_attr(OBJECT_NODE, 'conv2')
        graph.get_node_by_name('conv3.quant').set_attr(OBJECT_NODE, 'conv3')

        ReplaceAntiQuantPass(records).do_pass(graph, data_node)

    def test_match_pattern_success(self):
        records = {
            'conv1': {DATA_SCALE: 1, DATA_OFFSET: 0},
            'conv2': {DATA_SCALE: 1, DATA_OFFSET: 0},
            'conv3': {DATA_SCALE: 1, DATA_OFFSET: 0},
        }
        test_model = deepcopy(self.model_proto)
        graph = Graph(test_model)
        antiquant_node = graph.get_node_by_name('antiquant0.anti_quant')
        self.assertTrue(ReplaceAntiQuantPass(records).match_pattern(antiquant_node))

    def test_match_pattern_false(self):
        records = {
            'conv1': {DATA_SCALE: 1, DATA_OFFSET: 0},
            'conv2': {DATA_SCALE: 1, DATA_OFFSET: 0},
            'conv3': {DATA_SCALE: 1, DATA_OFFSET: 0},
        }
        test_model = deepcopy(self.model_proto)
        graph = Graph(test_model)
        antiquant_node = graph.get_node_by_name(CONCAT0)
        self.assertFalse(ReplaceAntiQuantPass(records).match_pattern(antiquant_node))

