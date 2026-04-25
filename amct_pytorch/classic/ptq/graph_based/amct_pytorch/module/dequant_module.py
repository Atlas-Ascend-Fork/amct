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
from ...amct_pytorch.module.quant_module import construct_fake_quant_dequant_cast_op

OUTPUT_0 = 'output0'


# def add_fake_dequant(graph, layer_name, dqscale, shift_bit, clip_mode):
def add_fake_dequant(graph, layer_name, dqscale):
    """
    Function: Add fakequant "dequant" pattern in graph
    """
    node = graph.get_node_by_name(layer_name)
    if node.has_attr('op_data_type'):
        dtype = node.get_attr('op_data_type')
    else:
        dtype = 'float32'
    layer_name = '.'.join([layer_name, 'fakedequant'])
    # add scale
    dqscale_node = graph.add_node(
        _construct_fake_dequant_dqscale(layer_name, dqscale))
    # add mul
    mul_node = graph.add_node(_construct_fake_dequant_mul(layer_name))

    # add links
    graph.add_edge(dqscale_node, 0, mul_node, 1)

    enter_node = mul_node
    out_node = mul_node

    if dtype == 'float16':
        cast_node = graph.add_node(construct_fake_quant_dequant_cast_op(layer_name, "float16"))
        graph.add_edge(mul_node, 0, cast_node, 0)
        out_node = cast_node

    return enter_node, out_node


def _construct_fake_dequant_pow(layer_name):
    '''construct pow op '''
    node_proto = onnx_pb.NodeProto()
    node_proto.name = '.'.join([layer_name, 'pow'])
    node_proto.op_type = 'Pow'
    node_proto.doc_string = 'pow of the Fake dequant'
    node_proto.input.extend([ # pylint: disable=E1101
        '.'.join([layer_name, 'pow_base.output0']),
        '.'.join([layer_name, 'shifit_bit.output0'])
    ])
    node_proto.output.extend( # pylint: disable=E1101
        ['.'.join([layer_name, 'pow.output0'])])

    return node_proto


def _construct_fake_dequant_div(layer_name):
    '''construct div op '''
    node_proto = onnx_pb.NodeProto()
    node_proto.name = '.'.join([layer_name, 'div'])
    node_proto.op_type = 'Div'
    node_proto.doc_string = 'div of the Fake dequant'
    node_proto.input.extend([ # pylint: disable=E1101
        '.'.join([layer_name, 'div.input0']),
        '.'.join([layer_name, 'pow.output0'])
    ])
    node_proto.output.extend( # pylint: disable=E1101
        ['.'.join([layer_name, 'div.output0'])])

    return node_proto


def _construct_fake_dequant_floor(layer_name):
    '''construct floor op '''
    node_proto = onnx_pb.NodeProto()
    node_proto.name = '.'.join([layer_name, 'floor'])
    node_proto.op_type = 'Floor'
    node_proto.doc_string = 'floor of the Fake dequant'
    node_proto.input.extend( # pylint: disable=E1101
        ['.'.join([layer_name, 'div.output0'])])
    node_proto.output.extend( # pylint: disable=E1101
        ['.'.join([layer_name, 'floor.output0'])])

    return node_proto


def _construct_fake_dequant_clip(layer_name):
    '''construct clip op '''
    node_proto = onnx_pb.NodeProto()
    node_proto.name = '.'.join([layer_name, 'clip'])
    node_proto.op_type = 'Clip'
    node_proto.doc_string = 'clip of the Fake dequant'
    node_proto.input.extend([ # pylint: disable=E1101
        '.'.join([layer_name, 'div.output0',
                  OUTPUT_0]), '.'.join([layer_name, 'min_val', OUTPUT_0]),
        '.'.join([layer_name, 'max_val', OUTPUT_0])
    ])
    node_proto.output.extend( # pylint: disable=E1101
        ['.'.join([layer_name, 'clip', OUTPUT_0])])

    return node_proto


def _construct_fake_dequant_mul(layer_name):
    '''construct mul op '''
    node_proto = onnx_pb.NodeProto()
    node_proto.name = '.'.join([layer_name, 'mul'])
    node_proto.op_type = 'Mul'
    node_proto.doc_string = 'mul of the Fake dequant'
    node_proto.input.extend([ # pylint: disable=E1101
        '.'.join([layer_name, 'clip', OUTPUT_0]),
        '.'.join([layer_name, 'dqscale', OUTPUT_0])
    ])
    node_proto.output.extend( # pylint: disable=E1101
        ['.'.join([layer_name, 'mul', OUTPUT_0])])

    return node_proto


def _construct_fake_dequant_dqscale(layer_name, dqscale):
    '''construct dqscale op '''
    node_proto = onnx_pb.NodeProto()
    node_proto.name = '.'.join([layer_name, 'dqscale'])
    node_proto.op_type = 'Constant'
    node_proto.doc_string = 'scale of the Fake dequant'
    node_proto.output.extend( # pylint: disable=E1101
        ['.'.join([layer_name, 'dqscale', OUTPUT_0])])

    AttributeProtoHelper(node_proto).set_attr_value('value', 'TENSOR', None)
    attr = node_proto.attribute[0] # pylint: disable=E1101
    TensorProtoHelper(attr.t).set_data(
        dqscale.reshape([-1]).tolist(), 'FLOAT', list(dqscale.shape))

    return node_proto


def _construct_fake_dequant_clip_min(layer_name, clip_min):
    '''construct clip's min op '''
    node_proto = onnx_pb.NodeProto()
    node_proto.doc_string = 'clip min of the Fake dequant'
    node_proto.name = '.'.join([layer_name, 'min_val'])
    node_proto.op_type = 'Constant'
    node_proto.output.extend( # pylint: disable=E1101
        ['.'.join([layer_name, 'min_val', OUTPUT_0])])

    AttributeProtoHelper(node_proto).set_attr_value('value', 'TENSOR', None)
    attr = node_proto.attribute[0] # pylint: disable=E1101
    TensorProtoHelper(attr.t).set_data([clip_min], 'FLOAT')

    return node_proto


def _construct_fake_dequant_clip_max(layer_name, clip_max):
    '''construct clip's max op '''
    node_proto = onnx_pb.NodeProto()
    node_proto.doc_string = 'clip max of the Fakequant'
    node_proto.name = '.'.join([layer_name, 'max_val'])
    node_proto.op_type = 'Constant'
    node_proto.output.extend( # pylint: disable=E1101
        ['.'.join([layer_name, 'max_val', OUTPUT_0])])

    AttributeProtoHelper(node_proto).set_attr_value('value', 'TENSOR', None)
    attr = node_proto.attribute[0] # pylint: disable=E1101
    TensorProtoHelper(attr.t).set_data([clip_max], 'FLOAT')

    return node_proto


def _construct_fake_dequant_pow_base(layer_name, pow_base):
    '''construct pow's base op '''
    node_proto = onnx_pb.NodeProto()
    node_proto.name = '.'.join([layer_name, 'pow_base'])
    node_proto.op_type = 'Constant'
    node_proto.doc_string = 'pow_base of the Fake dequant'
    node_proto.output.extend( # pylint: disable=E1101
        ['.'.join([layer_name, 'pow_base', OUTPUT_0])])

    AttributeProtoHelper(node_proto).set_attr_value('value', 'TENSOR', None)
    attr = node_proto.attribute[0] # pylint: disable=E1101
    TensorProtoHelper(attr.t).set_data([pow_base], 'FLOAT')

    return node_proto
