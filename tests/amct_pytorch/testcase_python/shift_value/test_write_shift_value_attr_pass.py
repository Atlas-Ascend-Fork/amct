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
"""Unit tests of WriteShiftValueAttrPass."""

import unittest

import pytest
from copy import deepcopy

from onnx import onnx_pb

from amct_pytorch.classic.graph_based.amct_pytorch.graph.graph import Graph
from amct_pytorch.classic.graph_based.amct_pytorch.optimizer.write_shift_value_attr_pass import (
    WriteShiftValueAttrPass,
)

pytestmark = pytest.mark.cpu


def _make_model_proto():
    """A minimal graph: Conv -> Relu -> Gemm."""
    model_proto = onnx_pb.ModelProto()
    model_proto.producer_name = 'model'
    graph_proto = onnx_pb.GraphProto()

    graph_input = graph_proto.input.add()
    graph_input.name = 'data'
    graph_input.type.tensor_type.shape.dim.add().dim_value = 2

    conv = graph_proto.node.add()
    conv.name = 'conv'
    conv.op_type = 'Conv'
    conv.input[:] = ['data', 'conv.weight', 'conv.bias']
    conv.output[:] = ['conv_out']

    relu = graph_proto.node.add()
    relu.name = 'relu'
    relu.op_type = 'Relu'
    relu.input[:] = ['conv_out']
    relu.output[:] = ['relu_out']

    gemm = graph_proto.node.add()
    gemm.name = 'fc'
    gemm.op_type = 'Gemm'
    gemm.input[:] = ['relu_out', 'fc.weight', 'fc.bias']
    gemm.output[:] = ['output']

    for name, dims in (
        ('conv.weight', [8, 3, 3, 3]),
        ('conv.bias', [8]),
        ('fc.weight', [4, 8]),
        ('fc.bias', [4]),
    ):
        initializer = graph_proto.initializer.add()
        initializer.name = name
        initializer.data_type = 1
        initializer.dims[:] = dims
        initializer.float_data[:] = [0.0]

    graph_output = graph_proto.output.add()
    graph_output.name = 'output'
    graph_output.type.tensor_type.shape.dim.add().dim_value = 4

    model_proto.graph.CopyFrom(graph_proto)
    return model_proto


def _get_fixed_shift_value(node):
    for attr in node.proto.attribute:
        if attr.name == 'fixed_shift_value':
            return attr.i
    return None


class TestWriteShiftValueAttrPass(unittest.TestCase):
    def setUp(self):
        self.model_proto = _make_model_proto()

    def test_match_pattern_success(self):
        records = {'conv': {'fixed_shift_value': 37}}
        graph = Graph(deepcopy(self.model_proto))
        conv_node = graph.get_node_by_name('conv')
        self.assertTrue(WriteShiftValueAttrPass(records).match_pattern(conv_node))

    def test_match_pattern_node_type_not_target(self):
        """Relu is not a FIXED_SHIFT_VALUE_ONNX_TYPES node."""
        records = {'relu': {'fixed_shift_value': 37}}
        graph = Graph(deepcopy(self.model_proto))
        relu_node = graph.get_node_by_name('relu')
        self.assertFalse(WriteShiftValueAttrPass(records).match_pattern(relu_node))

    def test_match_pattern_layer_not_in_records(self):
        records = {'fc': {'fixed_shift_value': 37}}
        graph = Graph(deepcopy(self.model_proto))
        conv_node = graph.get_node_by_name('conv')
        self.assertFalse(WriteShiftValueAttrPass(records).match_pattern(conv_node))

    def test_match_pattern_no_fixed_shift_value_field(self):
        """A record without the fixed_shift_value field does not match."""
        records = {'conv': {'scale_w': [0.001]}}
        graph = Graph(deepcopy(self.model_proto))
        conv_node = graph.get_node_by_name('conv')
        self.assertFalse(WriteShiftValueAttrPass(records).match_pattern(conv_node))

    def test_do_pass_writes_int_attribute(self):
        records = {'conv': {'fixed_shift_value': 37}}
        graph = Graph(deepcopy(self.model_proto))
        conv_node = graph.get_node_by_name('conv')
        pass_obj = WriteShiftValueAttrPass(records)
        pass_obj.do_pass(graph, conv_node)
        self.assertEqual(_get_fixed_shift_value(conv_node), 37)
        # attribute type is INT
        for attr in conv_node.proto.attribute:
            if attr.name == 'fixed_shift_value':
                self.assertEqual(attr.type, onnx_pb.AttributeProto.AttributeType.INT)

    def test_do_pass_keeps_other_nodes_clean(self):
        """Nodes without the field stay untouched."""
        records = {'conv': {'fixed_shift_value': 37}}
        graph = Graph(deepcopy(self.model_proto))
        conv_node = graph.get_node_by_name('conv')
        gemm_node = graph.get_node_by_name('fc')
        pass_obj = WriteShiftValueAttrPass(records)
        pass_obj.do_pass(graph, conv_node)
        self.assertIsNone(_get_fixed_shift_value(gemm_node))

    def test_none_records(self):
        graph = Graph(deepcopy(self.model_proto))
        conv_node = graph.get_node_by_name('conv')
        self.assertFalse(WriteShiftValueAttrPass(None).match_pattern(conv_node))


if __name__ == '__main__':
    unittest.main()
