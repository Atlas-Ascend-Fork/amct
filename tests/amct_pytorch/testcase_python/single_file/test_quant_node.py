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
import json
from unittest.mock import MagicMock
from copy import deepcopy

import numpy as np
import torch
from onnx import onnx_pb

from amct_pytorch.graph_based_compression.amct_pytorch.utils.quant_node import QuantOpInfo
from amct_pytorch.graph_based_compression.amct_pytorch.graph.node import Node
from amct_pytorch.graph_based_compression.amct_pytorch.graph.graph import Graph


class TestModelHelper(unittest.TestCase):
    """
    The UT for QuantOpInfo
    """
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
        # Add weights
        weights = self.graph.initializer.add()
        weights.name = 'conv1.pre_weights'
        weights.data_type = 3
        weights.int32_data[:] = [1, 2, 3, 4, 5, 6]
        weights.dims[:] = [1, 1, 2, 3]
        # trans wts
        wts_trans = self.graph.node.add()
        wts_trans.name = 'wts_trans_0'
        wts_trans.op_type = 'Transpose'
        wts_trans.input[:] = ['conv1.pre_weights',]
        wts_trans.output[:] = ['conv1.weights']
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

    def test_get_parent_module_trans_node(self):
        test_model = deepcopy(self.model_proto)
        graph = Graph(test_model)
        node_Trans = graph.get_node_by_name('conv1')
        graph = QuantOpInfo.get_weight_node(node_Trans)
        self.assertIsNotNone(graph)

    def test_get_cout_length(self):
        test_model = deepcopy(self.model_proto)
        graph = Graph(test_model)
        node_add = graph.get_node_by_name('add1')
        self.assertRaises(RuntimeError, QuantOpInfo.get_cout_length, node_add)

    def test_get_cout_length(self):
        test_model = deepcopy(self.model_proto)
        graph = Graph(test_model)
        node_add = graph.get_node_by_name('add1')
        self.assertRaises(RuntimeError, QuantOpInfo.get_cout_length, node_add)

    def test_get_dst_num_bits_none_records(self):
        records = None
        with self.assertRaises(RuntimeError):
            num_bits = QuantOpInfo.get_dst_num_bits(records, "")

    def test_get_dst_num_bits_none_op(self):
        records = {
            'conv1': {
                'data_scale': 1,
                'data_offset': 0,
                'act_type': 'INT8'
            }
        }
        with self.assertRaises(RuntimeError):
            num_bits = QuantOpInfo.get_dst_num_bits(records, "conv2")

    def test_get_dst_num_bits_act_type(self):
        records = {
            'conv1': {
                'data_scale': 1,
                'data_offset': 0,
                'act_type': 'INT16'
            }
        }

        num_bits = QuantOpInfo.get_dst_num_bits(records, "conv1", "act")
        self.assertEqual(num_bits, 16)

    def test_get_dst_num_bits_unset_act_type(self):
        records = {
            'conv1': {
                'data_scale': 1,
                'data_offset': 0,
                'act_type': 'UNSET',
            }
        }
        num_bits = QuantOpInfo.get_dst_num_bits(records, "conv1", "act")
        self.assertEqual(num_bits, 8)

    def test_get_dst_num_bits_invalid_act_type(self):
        records = {
            'conv1': {
                'data_scale': 1,
                'data_offset': 0,
                'act_type': 'xxx'
            }
        }
        with self.assertRaises(RuntimeError):
            num_bits = QuantOpInfo.get_dst_num_bits(records, "conv1", "act")

    def test_get_dst_num_bits_wts_type(self):
        records = {
            'conv1': {
                'data_scale': 1,
                'data_offset': 0,
                'wts_type': 'INT8'
            }
        }
        num_bits = QuantOpInfo.get_dst_num_bits(records, "conv1", "wts")
        self.assertEqual(num_bits, 8)

    def test_get_dst_num_bits_unset_wts_type(self):
        records = {
            'conv1': {
                'data_scale': 1,
                'data_offset': 0,
                'wts_type': 'UNSET',
            }
        }
        num_bits = QuantOpInfo.get_dst_num_bits(records, "conv1", "wts")
        self.assertEqual(num_bits, 8)

    def test_get_dst_num_bits_invalid_wts_type(self):
        records = {
            'conv1': {
                'data_scale': 1,
                'data_offset': 0,
                'wts_type': 'xxx'
            }
        }
        with self.assertRaises(RuntimeError):
            num_bits = QuantOpInfo.get_dst_num_bits(records, "conv1", "wts")

    def test_get_dst_num_bits_none_data_type(self):
        records = {
            'conv1': {
                'data_scale': 1,
                'data_offset': 0,
                'dst_type': 'INT8'
            }
        }
        num_bits = QuantOpInfo.get_dst_num_bits(records, "conv1")
        self.assertEqual(num_bits, 8)

    def test_get_dst_num_bits_invalid_data_type(self):
        records = {
            'conv1': {
                'data_scale': 1,
                'data_offset': 0,
                'dst_type': 'INT16'
            }
        }
        with self.assertRaises(RuntimeError):
            num_bits = QuantOpInfo.get_dst_num_bits(records, "conv1")

    def test_get_dst_num_bits_none_data_type_none_dst_type(self):
        records = {
            'conv1': {
                'data_scale': 1,
                'data_offset': 0,
            }
        }
        num_bits = QuantOpInfo.get_dst_num_bits(records, "conv1")
        self.assertEqual(num_bits, 8)