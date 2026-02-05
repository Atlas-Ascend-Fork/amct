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
from amct_pytorch.graph_based_compression.amct_pytorch.graph.node import Node

CUR_DIR = os.path.split(os.path.realpath(__file__))[0]

class TestNode(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pass

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):
        # set basic info
        self.node_proto = onnx_pb.NodeProto()
        self.node_proto.name = 'conv'
        self.node_proto.op_type = 'Conv'
        self.node_proto.input[:] = ['data', 'weights', 'bias']
        self.node_proto.output[:] = ['conv']
        # add attribute "dilations"
        dilations = self.node_proto.attribute.add()
        dilations.name = 'dilations'
        dilations.type = onnx_pb.AttributeProto.AttributeType.INTS
        dilations.ints[:] = [1, 1, 1, 1]
        # add attribute "kernel_shape"
        kernel_shape = self.node_proto.attribute.add()
        kernel_shape.name = 'kernel_shape'
        kernel_shape.type = onnx_pb.AttributeProto.AttributeType.INTS
        kernel_shape.ints[:] = [64, 3, 3, 3]
        # add attribute "pads"
        pads = self.node_proto.attribute.add()
        pads.name = 'pads'
        pads.type = onnx_pb.AttributeProto.AttributeType.INTS
        pads.ints[:] = [0, 0, 0, 0]
        # add attribute "group"
        group = self.node_proto.attribute.add()
        group.name = 'group'
        group.type = onnx_pb.AttributeProto.AttributeType.INT
        group.i = 1

    def tearDown(self):
        pass

    def test_init_success(self):
        test_proto = deepcopy(self.node_proto)
        node_conv = Node(0, test_proto)
        print(node_conv.get_input_anchor(0))
        print(node_conv.get_input_anchor(1))
        print(node_conv.get_input_anchor(2))
        print(node_conv.get_output_anchor(0))
        print(node_conv)
        self.assertEqual(node_conv.name, 'conv')
        self.assertEqual(node_conv.type, 'Conv')
        self.assertEqual(node_conv.index, 0)
        self.assertEqual(len(node_conv.input_anchors), 3)
        self.assertEqual(len(node_conv.output_anchors), 1)
        self.assertEqual(node_conv.get_input_anchor(0).name, 'data')
        self.assertEqual(node_conv.get_input_anchor(1).name, 'weights')
        self.assertEqual(node_conv.get_input_anchor(2).name, 'bias')
        self.assertEqual(node_conv.get_output_anchor(0).name, 'conv')
        node_conv.set_name('conv0')
        self.assertEqual(node_conv.name, 'conv0')
        node_conv_proto = node_conv.dump_proto()
        self.assertEqual(type(node_conv_proto), onnx_pb.NodeProto)
        self.assertEqual(node_conv_proto.name, 'conv0')
        self.assertEqual(node_conv_proto.op_type, 'Conv')
        self.assertEqual(list(node_conv_proto.input), ['data', 'weights', 'bias'])
        self.assertEqual(list(node_conv_proto.output), ['conv'])
        self.assertEqual(node_conv_proto.attribute[0].name, 'dilations')
        self.assertEqual(node_conv_proto.attribute[1].name, 'kernel_shape')
        self.assertEqual(node_conv_proto.attribute[2].name, 'pads')
        self.assertEqual(node_conv_proto.attribute[3].name, 'group')

    def test_unsupport_type(self):
        graph_proto = onnx_pb.GraphProto()
        graph_proto.name = 'conv'
        self.assertRaises(
            TypeError,
            Node,
            0,
            graph_proto)

    def test_sparse_initializer_without_data(self):
        sparse_initializer = onnx_pb.SparseTensorProto()
        sparse_initializer.dims[:] = [3, 3, 3, 3]
        self.assertRaises(
            RuntimeError,
            Node,
            0,
            sparse_initializer)

    def test_sparse_initializer_with_values(self):
        sparse_initializer = onnx_pb.SparseTensorProto()
        sparse_initializer.dims[:] = [3, 3, 3, 3]
        sparse_initializer.values.name = 'sparse_initializer'
        sparse_node = Node(0, sparse_initializer)
        self.assertEqual(sparse_node.name, 'sparse_initializer')
        self.assertEqual(sparse_node.get_output_anchor(0).name, 'sparse_initializer')
        sparse_proto = sparse_node.dump_proto()
        self.assertEqual(type(sparse_proto), onnx_pb.SparseTensorProto)
        self.assertEqual(sparse_proto.values.name, 'sparse_initializer')
        self.assertEqual(list(sparse_proto.dims), [3, 3, 3, 3])

    def test_sparse_initializer_with_indices(self):
        sparse_initializer = onnx_pb.SparseTensorProto()
        sparse_initializer.dims[:] = [3, 3, 3, 3]
        sparse_initializer.indices.name = 'sparse_initializer'
        sparse_node = Node(0, sparse_initializer)
        self.assertEqual(sparse_node.name, 'sparse_initializer')
        self.assertEqual(sparse_node.get_output_anchor(0).name, 'sparse_initializer')
        sparse_proto = sparse_node.dump_proto()
        self.assertEqual(type(sparse_proto), onnx_pb.SparseTensorProto)
        self.assertEqual(sparse_proto.indices.name, 'sparse_initializer')
        self.assertEqual(list(sparse_proto.dims), [3, 3, 3, 3])

    def test_graph_anchor(self):
        graph_anchor = onnx_pb.ValueInfoProto()
        graph_anchor.name = 'graph_anchor'
        graph_anchor_node = Node(0, graph_anchor)
        self.assertEqual(graph_anchor_node.name, 'graph_anchor')
        self.assertEqual(type(graph_anchor_node.dump_proto()), onnx_pb.ValueInfoProto)

    def test_tensor_node(self):
        constant = onnx_pb.TensorProto()
        constant.name = 'constant'
        constant_node = Node(0, constant)
        self.assertEqual(constant_node.name, 'constant')

    def test_add_input_anchor_to_unsupport_node(self):
        sparse_initializer = onnx_pb.SparseTensorProto()
        sparse_initializer.dims[:] = [3, 3, 3, 3]
        sparse_initializer.values.name = 'sparse_initializer'
        sparse_node = Node(0, sparse_initializer)
        self.assertRaises(
            RuntimeError,
            sparse_node.add_input_anchor,
            'input')

    def test_add_input_anchor_to_graph_anchor(self):
        graph_anchor = onnx_pb.ValueInfoProto()
        graph_anchor.name = 'graph_anchor'
        graph_anchor_node = Node(0, graph_anchor)
        graph_anchor_node.add_input_anchor('input0')
        self.assertRaises(
            RuntimeError,
            graph_anchor_node.add_input_anchor,
            'input1')

    def test_get_output_anchor_index_success(self):
        test_proto = deepcopy(self.node_proto)
        node_conv = Node(0, test_proto)
        self.assertEqual(node_conv.get_output_anchor_index('conv'), 0)

    def test_get_output_anchor_index_failed(self):
        test_proto = deepcopy(self.node_proto)
        node_conv = Node(0, test_proto)
        self.assertRaises(
            ValueError,
            node_conv.get_output_anchor_index,
            'not_exist')

    def test_add_output_anchor_failed(self):
        graph_anchor = onnx_pb.ValueInfoProto()
        graph_anchor.name = 'graph_anchor'
        graph_anchor_node = Node(0, graph_anchor)
        graph_anchor_node.add_output_anchor('output0')
        self.assertRaises(
            RuntimeError,
            graph_anchor_node.add_output_anchor,
            'output1')

    def test_get_output_anchor_by_name_success(self):
        test_proto = deepcopy(self.node_proto)
        node_conv = Node(0, test_proto)
        self.assertEqual(node_conv.get_output_anchor_by_name('conv').index, 0)

    def test_get_output_anchor_by_name_failed(self):
        test_proto = deepcopy(self.node_proto)
        node_conv = Node(0, test_proto)
        self.assertRaises(
            ValueError,
            node_conv.get_output_anchor_by_name,
            'not_exist')

    def test_dump_proto_exceed_input_index(self):
        test_proto0 = deepcopy(self.node_proto)
        test_proto1 = deepcopy(self.node_proto)
        node_conv0 = Node(0, test_proto0)
        node_conv1 = Node(1, test_proto1)
        src_anchor = node_conv0.get_output_anchor(0)
        dst_anchor = node_conv1.get_input_anchor(0)
        src_anchor.add_link(dst_anchor)
        dst_anchor.add_link(src_anchor)
        src_anchor._index = 1

        self.assertRaises(
            RuntimeError,
            node_conv1.dump_proto)

    def test_dump_graph_anchor_failed(self):
        graph_anchor = onnx_pb.ValueInfoProto()
        graph_anchor.name = 'graph_anchor'
        graph_anchor_node = Node(0, graph_anchor)
        graph_anchor_node.add_input_anchor('output0')
        graph_anchor_node.add_output_anchor('output0')

        self.assertRaises(
            RuntimeError,
            graph_anchor_node.dump_proto)

    def test_dump_graph_anchor_failed_002(self):
        test_proto = deepcopy(self.node_proto)
        node_conv = Node(0, test_proto)

        graph_anchor = onnx_pb.ValueInfoProto()
        graph_anchor.name = 'graph_anchor'
        graph_anchor_node = Node(0, graph_anchor)
        graph_anchor_node.add_input_anchor('output0')

        src_anchor = node_conv.get_output_anchor(0)
        dst_anchor = graph_anchor_node.get_input_anchor(0)
        src_anchor.add_link(dst_anchor)
        dst_anchor.add_link(src_anchor)

        self.assertRaises(
            RuntimeError,
            graph_anchor_node.dump_proto)


    def test_dump_node_failed(self):
        test_proto = deepcopy(self.node_proto)
        node_conv = Node(0, test_proto)

        node_conv._node_proto = onnx_pb.GraphProto()
        self.assertRaises(
            TypeError,
            node_conv.dump_proto)
