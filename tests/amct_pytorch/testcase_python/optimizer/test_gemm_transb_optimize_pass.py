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

from amct_pytorch.amct_pytorch_inner.amct_pytorch.optimizer.gemm_transb_optimize_pass import \
    GemmTransBOptimizePass


class TestInsertQuantPass(unittest.TestCase):
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
        # Add fc0
        fc0 = self.graph.node.add()
        fc0.name = 'fc0'
        fc0.op_type = 'Gemm'
        fc0.input[:] = ['data0', 'fc0.weights', 'fc0.bias']
        fc0.output[:] = ['fc0']
        # add attribute "kernel_shape"
        attr_transb = fc0.attribute.add()
        attr_transb.name = 'transB'
        attr_transb.type = onnx_pb.AttributeProto.AttributeType.INT
        attr_transb.i = 1
        # Add weights
        weights = self.graph.initializer.add()
        weights.name = 'fc0.weights'
        weights.data_type = onnx_pb.TensorProto.DataType.FLOAT
        weights.float_data[:] = [1, 2, 3, 4, 5, 6]
        weights.dims[:] = [2, 3]
        # Add bias
        bias = self.graph.initializer.add()
        bias.name = 'fc0.bias'
        bias.data_type = 6
        bias.int32_data[:] = [0]
        bias.dims[:] = [1]
        # Add relu
        relu1 = self.graph.node.add()
        relu1.name = 'relu1'
        relu1.op_type = 'Relu'
        relu1.input[:] = ['fc0']
        relu1.output[:] = ['output']
        # add output
        graph_output = self.graph.output.add()
        graph_output.name = 'output'
        graph_output.type.tensor_type.shape.dim.add().dim_value = 1
        self.model_proto.graph.CopyFrom(self.graph)

    def tearDown(self):
        pass


    def test_match_pattern_success(self):
        records = {
            'fc0': {
                'scale': 1,
                'offset': 0
            }
        }
        test_model = deepcopy(self.model_proto)
        fc_node = Graph(test_model).get_node_by_name('fc0')
        self.assertTrue(GemmTransBOptimizePass(records).match_pattern(fc_node))

    def test_match_pattern_not_in_records(self):
        records = {
            'fc1': {
                'scale': 1,
                'offset': 0
            }
        }
        test_model = deepcopy(self.model_proto)
        fc_node = Graph(test_model).get_node_by_name('fc0')
        self.assertFalse(GemmTransBOptimizePass(records).match_pattern(fc_node))

    def test_match_pattern_transb_false(self):
        records = {
            'fc0': {
                'scale': 1,
                'offset': 0
            }
        }
        test_model = deepcopy(self.model_proto)
        test_model.graph.node[0].attribute[0].i = 0
        fc_node = Graph(test_model).get_node_by_name('fc0')
        self.assertFalse(GemmTransBOptimizePass(records).match_pattern(fc_node))

    def test_do_pass_success(self):
        records = {
            'fc0': {
                'scale': 1,
                'offset': 0
            }
        }
        test_model = deepcopy(self.model_proto)
        graph = Graph(test_model)
        fc_node = graph.get_node_by_name('fc0')
        GemmTransBOptimizePass(records).do_pass(graph, fc_node)
        self.assertEqual(fc_node.proto.attribute[0].i, 0)
        weights = graph.get_node_by_name('fc0.weights')
        self.assertEqual(list(weights.proto.float_data), [1.0, 4.0, 2.0, 5.0, 3.0, 6.0])
        self.assertEqual(list(weights.proto.dims), [3, 2])

    def test_do_pass_without_weights(self):
        records = {
            'fc0': {
                'scale': 1,
                'offset': 0
            }
        }
        test_model = deepcopy(self.model_proto)
        test_model.graph.node[0].input[:] = ['data0']
        graph = Graph(test_model)
        fc_node = graph.get_node_by_name('fc0')
        GemmTransBOptimizePass(records).do_pass(graph, fc_node)
        self.assertEqual(fc_node.proto.attribute[0].i, 1)
        weights = graph.get_node_by_name('fc0.weights')
        self.assertEqual(list(weights.proto.float_data), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        self.assertEqual(list(weights.proto.dims), [2, 3])

    def test_do_pass_with_illegal_weights(self):
        records = {
            'fc0': {
                'scale': 1,
                'offset': 0
            }
        }
        test_model = deepcopy(self.model_proto)
        test_model.graph.initializer[0].dims[:] = [1, 2, 3]
        graph = Graph(test_model)
        fc_node = graph.get_node_by_name('fc0')
        self.assertRaises(
            RuntimeError,
            GemmTransBOptimizePass.do_pass,
            GemmTransBOptimizePass(records),
            graph,
            fc_node)
