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

from onnx import onnx_pb, AttributeProto # pylint: disable=E0401
from ...amct_pytorch.common.utils.onnx_node_util import AttributeProtoHelper
from ...amct_pytorch.utils.onnx_initializer_util import TensorProtoHelper

SUB = 'sub'
CAST = 'cast'
MUL = 'mul'
OUTPUT_0 = 'output0'


def add_fakequant(graph, layer_name, scale, offset, quant_bit):
    """
    Function: Add fakequant "quant" pattern in graph
    """
    quantized_layer_name = '.'.join(layer_name.split('.')[0:-1])
    node = graph.get_node_by_name(quantized_layer_name)
    if node.has_attr('op_data_type'):
        dtype = node.get_attr('op_data_type')
    else:
        dtype = 'float32'
    layer_name = '.'.join([layer_name, 'fakequant'])
    # add scale, offset, clip_min, clip_max
    scale_node = graph.add_node(_construct_fakequant_scale(layer_name, scale))
    offset_node = graph.add_node(
        _construct_fakequant_offset(layer_name, offset))
    min_node = graph.add_node(
        _construct_fakequant_clip_min(layer_name, -2**(quant_bit - 1)))
    max_node = graph.add_node(
        _construct_fakequant_clip_max(layer_name, 2**(quant_bit - 1) - 1))
    # add mul
    mul_node = graph.add_node(_construct_fakequant_mul(layer_name))
    # add round
    round_node = graph.add_node(_construct_fakequant_round(layer_name))
    # add add
    add_node = graph.add_node(_construct_fakequant_add(layer_name))
    # add clip
    clip_node = graph.add_node(_construct_fakequant_clip(layer_name))
    # add sub
    sub_node = graph.add_node(_construct_fakequant_sub(layer_name))

    # add links
    graph.add_edge(scale_node, 0, mul_node, 1)
    graph.add_edge(mul_node, 0, round_node, 0)
    graph.add_edge(round_node, 0, add_node, 0)
    graph.add_edge(offset_node, 0, add_node, 1)
    graph.add_edge(add_node, 0, clip_node, 0)
    graph.add_edge(min_node, 0, clip_node, 1)
    graph.add_edge(max_node, 0, clip_node, 2)
    graph.add_edge(clip_node, 0, sub_node, 0)
    graph.add_edge(offset_node, 0, sub_node, 1)
    enter_node = mul_node
    out_node = sub_node

    if dtype == 'float16':
        cast_node = graph.add_node(construct_fake_quant_dequant_cast_op(layer_name, "float32"))
        graph.add_edge(cast_node, 0, mul_node, 0)
        enter_node = cast_node

    return enter_node, out_node


def add_fake_antiquant(graph, layer_name, scale):
    """
    Function: Add fakequant "antiquant" pattern in graph
    """
    layer_name = '.'.join([layer_name, 'fake_antiquant'])
    # add scale
    scale_node = graph.add_node(_construct_fakequant_scale(layer_name, scale))
    # add mul
    mul_node = graph.add_node(_construct_fakequant_mul(layer_name))

    # add links
    graph.add_edge(scale_node, 0, mul_node, 1)

    return mul_node


def _construct_fakequant_scale(layer_name, scale):
    ''' construct scale op'''
    node_proto = onnx_pb.NodeProto()
    node_proto.name = '.'.join([layer_name, 'scale'])
    node_proto.op_type = 'Constant'
    node_proto.doc_string = 'scale of the Fakequant'
    node_proto.output.extend( # pylint: disable=E1101
        ['.'.join([layer_name, 'scale', OUTPUT_0])])

    AttributeProtoHelper(node_proto).set_attr_value('value', 'TENSOR', None)
    attr = node_proto.attribute[0] # pylint: disable=E1101
    TensorProtoHelper(attr.t).set_data([scale], 'FLOAT')

    return node_proto


def _construct_fakequant_offset(layer_name, offset):
    ''' construct offset op'''
    node_proto = onnx_pb.NodeProto()
    node_proto.name = '.'.join([layer_name, 'offset'])
    node_proto.op_type = 'Constant'
    node_proto.doc_string = 'offset of the Fakequant'
    node_proto.output.extend( # pylint: disable=E1101
        ['.'.join([layer_name, 'offset', OUTPUT_0])])

    AttributeProtoHelper(node_proto).set_attr_value('value', 'TENSOR', None)
    attr = node_proto.attribute[0] # pylint: disable=E1101
    TensorProtoHelper(attr.t).set_data([offset], 'FLOAT')

    return node_proto


def _construct_fakequant_clip_min(layer_name, clip_min):
    ''' construct clip's min op'''
    node_proto = onnx_pb.NodeProto()
    node_proto.name = '.'.join([layer_name, 'min_val'])
    node_proto.op_type = 'Constant'
    node_proto.doc_string = 'clip min of the Fakequant'
    node_proto.output.extend( # pylint: disable=E1101
        ['.'.join([layer_name, 'min_val', OUTPUT_0])])

    AttributeProtoHelper(node_proto).set_attr_value('value', 'TENSOR', None)
    attr = node_proto.attribute[0] # pylint: disable=E1101
    TensorProtoHelper(attr.t).set_data([clip_min], 'FLOAT')

    return node_proto


def _construct_fakequant_clip_max(layer_name, clip_max):
    ''' construct clip's max op'''
    node_proto = onnx_pb.NodeProto()
    node_proto.name = '.'.join([layer_name, 'max_val'])
    node_proto.op_type = 'Constant'
    node_proto.doc_string = 'clip max of the Fakequant'
    node_proto.output.extend( # pylint: disable=E1101
        ['.'.join([layer_name, 'max_val', OUTPUT_0])])

    AttributeProtoHelper(node_proto).set_attr_value('value', 'TENSOR', None)
    attr = node_proto.attribute[0] # pylint: disable=E1101
    TensorProtoHelper(attr.t).set_data([clip_max], 'FLOAT')

    return node_proto


def _construct_fakequant_mul(layer_name):
    ''' construct mul op'''
    node_proto = onnx_pb.NodeProto()
    node_proto.name = '.'.join([layer_name, MUL])
    node_proto.op_type = 'Mul'
    node_proto.doc_string = 'mul of the Fakequant'
    node_proto.input.extend([ # pylint: disable=E1101
        '.'.join([layer_name, MUL, 'input0']),
        '.'.join([layer_name, 'scale', OUTPUT_0])
    ])
    node_proto.output.extend( # pylint: disable=E1101
        ['.'.join([layer_name, MUL, OUTPUT_0])])

    return node_proto


def _construct_fakequant_round(layer_name):
    ''' construct round op'''
    node_proto = onnx_pb.NodeProto()
    node_proto.name = '.'.join([layer_name, 'round'])
    node_proto.op_type = 'Round'
    node_proto.doc_string = 'round of the Fakequant'
    node_proto.input.extend( # pylint: disable=E1101
        ['.'.join([layer_name, MUL, OUTPUT_0])])
    node_proto.output.extend( # pylint: disable=E1101
        ['.'.join([layer_name, 'round', OUTPUT_0])])

    return node_proto


def _construct_fakequant_add(layer_name):
    ''' construct add op'''
    node_proto = onnx_pb.NodeProto()
    node_proto.name = '.'.join([layer_name, 'add'])
    node_proto.op_type = 'Add'
    node_proto.doc_string = 'add of the Fakequant'
    node_proto.input.extend([ # pylint: disable=E1101
        '.'.join([layer_name, 'round', OUTPUT_0]),
        '.'.join([layer_name, 'offset', OUTPUT_0])
    ])
    node_proto.output.extend( # pylint: disable=E1101
        ['.'.join([layer_name, 'add', OUTPUT_0])])

    return node_proto


def _construct_fakequant_clip(layer_name):
    ''' construct clip op'''
    node_proto = onnx_pb.NodeProto()
    node_proto.name = '.'.join([layer_name, 'clip'])
    node_proto.op_type = 'Clip'
    node_proto.doc_string = 'clip of the Fakequant'
    node_proto.input.extend([ # pylint: disable=E1101
        '.'.join([layer_name, 'add',
                  OUTPUT_0]), '.'.join([layer_name, 'min_val', OUTPUT_0]),
        '.'.join([layer_name, 'max_val', OUTPUT_0])
    ])
    node_proto.output.extend( # pylint: disable=E1101
        ['.'.join([layer_name, 'clip', OUTPUT_0])])

    return node_proto


def _construct_fakequant_sub(layer_name):
    ''' construct sub op'''
    node_proto = onnx_pb.NodeProto()
    node_proto.name = '.'.join([layer_name, SUB])
    node_proto.op_type = 'Sub'
    node_proto.doc_string = 'sub of the Fakequant'
    node_proto.input.extend([ # pylint: disable=E1101
        '.'.join([layer_name, SUB, 'input0']),
        '.'.join([layer_name, SUB, 'input1'])
    ])
    node_proto.output.extend( # pylint: disable=E1101
        ['.'.join([layer_name, SUB, OUTPUT_0])])

    return node_proto


def construct_fake_quant_dequant_cast_op(layer_name, to_dtype):
    node_proto = onnx_pb.NodeProto()
    node_proto.name = '.'.join([layer_name, CAST])
    node_proto.op_type = 'Cast'
    node_proto.doc_string = 'cast of the Fakequant'
    node_proto.input.extend([
        '.'.join([layer_name, CAST, 'input0'])
    ])
    node_proto.output.extend([
        '.'.join([layer_name, CAST, OUTPUT_0])
    ])
    node_proto.attribute.add().name = 'to'
    node_proto.attribute[0].type = AttributeProto.INT
    if to_dtype == "float32":
        node_proto.attribute[0].i = onnx_pb.TensorProto.FLOAT
    else:
        node_proto.attribute[0].i = onnx_pb.TensorProto.FLOAT16
    return node_proto