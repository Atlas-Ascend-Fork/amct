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

from amct_pytorch.graph_based_compression.amct_pytorch.optimizer import ReplaceAvgpoolFlattenPass


class TestReplaceAvgpoolFlattenPass(unittest.TestCase):
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
        # Add fc0
        fc0 = cls.graph.node.add()
        fc0.name = 'avg_pool'
        fc0.op_type = 'GlobalAveragePool'
        fc0.input[:] = ['data0', ]
        fc0.output[:] = ['avg_pool_output']

        # Add Flatten
        flatten0 = cls.graph.node.add()
        flatten0.name = 'flatten'
        flatten0.op_type = 'Flatten'
        flatten0.input[:] = ['avg_pool_output']
        flatten0.output[:] = ['output']
        # add attribute "kernel_shape"
        axis = flatten0.attribute.add()
        axis.name = 'axis'
        axis.type = onnx_pb.AttributeProto.AttributeType.INT
        axis.i = 1

        # add output
        graph_output = cls.graph.output.add()
        graph_output.name = 'output'
        graph_output.type.tensor_type.shape.dim.add().dim_value = 1
        cls.model_proto.graph.CopyFrom(cls.graph)

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_match_pattern_success(self):
        test_model = deepcopy(self.model_proto)
        faltten_node = Graph(test_model).get_node_by_name('avg_pool')
        is_matched = ReplaceAvgpoolFlattenPass.match_pattern(faltten_node)
        self.assertTrue(is_matched)

    def test_do_pass(self):
        test_model = deepcopy(self.model_proto)
        graph = Graph(test_model)
        faltten_node = graph.get_node_by_name('avg_pool')
        ReplaceAvgpoolFlattenPass.do_pass(graph, faltten_node)
        self.assertEqual(1, len(graph.nodes))

    def test_match_pattern_fail_01(self):
        test_model = deepcopy(self.model_proto)
        faltten_node = Graph(test_model).get_node_by_name('flatten')
        is_matched = ReplaceAvgpoolFlattenPass.match_pattern(faltten_node)
        self.assertTrue(not is_matched)

    def test_match_pattern_fail_02(self):
        test_model = deepcopy(self.model_proto)
        faltten_node = Graph(test_model).get_node_by_name('flatten')
        faltten_node.proto.attribute[0].i = 2
        is_matched = ReplaceAvgpoolFlattenPass.match_pattern(faltten_node)
        self.assertTrue(not is_matched)


