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
import torch
import numpy as np

from ...amct_pytorch.common.utils.onnx_node_util import AttributeProtoHelper
from ...amct_pytorch.utils.quant_node import QuantOpInfo

MUL = 'Mul'
ACTIVATION_MAP = {
    'Relu': torch.nn.ReLU(),
    'Sigmoid': torch.nn.Sigmoid(),
    'Tanh': torch.nn.Tanh()
}


def _is_weight_ada_round(quant_config, layer_name):
    '''
    Function: determine whether weight do ada round
    Parameters:
        quant_config: config which parsed from json file
        layer_name: name of quant layer
    Return:
        True: do ada round
        False: don't do ada round
    '''
    if layer_name not in quant_config.keys():
        return False
    layer_config = quant_config.get(layer_name)
    if layer_config.get('weight_quant_params').get('wts_algo') != 'ada_quantize':
        return False
    return True


def _match_gelu_subgraph(input_nodes):
    '''
    Function: judge whether match gelu() subgraph
        / \
        | |
        | Div
        | |
        | Erf
        | |
        | Add
        \ /
        Mul
         |
        Mul
    Parameters:
        input_nodes: input_nodes of subgraph
    Return:
        True: match
        False: mismatch
    '''
    if len(input_nodes) != 2:
        return False

    div_node = input_nodes[0]
    if div_node.type != 'Div':
        return False
    consumers, _ = div_node.get_consumers(0)
    if len(consumers) != 1:
        return False
    div_value = QuantOpInfo.get_node_input_value(div_node, 1)
    except_value = np.array(1.4142135381698608, dtype=np.float32)
    if div_value != except_value and div_value != except_value.astype(np.float16):
        return False

    erf_node = consumers[0]
    if erf_node.type != 'Erf':
        return False
    consumers, _ = erf_node.get_consumers(0)
    if len(consumers) != 1:
        return False

    add_node = consumers[0]
    if add_node.type != 'Add':
        return False
    consumers, _ = add_node.get_consumers(0)
    if len(consumers) != 1:
        return False
    add_value = QuantOpInfo.get_node_input_value(add_node, 1)
    if add_value != 1:
        return False

    mul0_node = consumers[0]
    if mul0_node.type != MUL or mul0_node is not input_nodes[1]:
        return False
    consumers, _ = mul0_node.get_consumers(0)
    if len(consumers) != 1:
        return False

    mul1_node = consumers[0]
    if mul1_node.type != MUL:
        return False
    mul1_value = QuantOpInfo.get_node_input_value(mul1_node, 1)
    if mul1_value != 0.5:
        return False

    return True


def _match_gelu_tanh_subgraph_front_part(input_nodes):
    '''
    Function: judge whether match gelu('tanh') subgraph front part
     / / | |\
    | |  | Mul0
    | |  \ /
    | |  Mul1
    | |   |
    | |  Mul2
    |  \ /
    |  Add0
    |   |
    Parameters:
        input_nodes: input_nodes of subgraph
    Return:
        True: match
        False: mismatch
    '''
    mul0_node = input_nodes[0]
    if mul0_node.type != MUL or mul0_node is not input_nodes[1]:
        return False
    consumers, _ = mul0_node.get_consumers(0)
    if len(consumers) != 1:
        return False

    mul1_node = consumers[0]
    if mul1_node.type != MUL or mul1_node is not input_nodes[2]:
        return False
    consumers, _ = mul1_node.get_consumers(0)
    if len(consumers) != 1:
        return False

    mul2_node = consumers[0]
    if mul2_node.type != MUL:
        return False
    consumers, _ = mul2_node.get_consumers(0)
    if len(consumers) != 1:
        return False
    mul2_value = QuantOpInfo.get_node_input_value(mul2_node, 0)
    except_value = np.array(0.044714998453855515, dtype=np.float32)
    if mul2_value != except_value and mul2_value != except_value.astype(np.float16):
        return False

    add0_node = consumers[0]
    if add0_node.type != 'Add' or add0_node is not input_nodes[3]:
        return False
    consumers, _ = add0_node.get_consumers(0)
    if len(consumers) != 1:
        return False

    return True


def _match_gelu_tanh_subgraph_back_part(input_nodes):
    '''
    Function: judge whether match gelu('tanh') subgraph back part
     |  \  /
     |  Add0
     |    |
     |  Mul3
     |    |
     |  Tanh
     |    |
     |  Add1
     |    |
      \  /
      Mul4
        |
      Mul5
    Parameters:
        input_nodes: input_nodes of subgraph
    Return:
        True: match
        False: mismatch
    '''
    add0_node = input_nodes[3]
    mul3_node = add0_node.get_output_anchor(0).get_peer_input_anchor()[0].node
    if mul3_node.type != MUL:
        return False
    consumers, _ = mul3_node.get_consumers(0)
    if len(consumers) != 1:
        return False
    mul3_value = QuantOpInfo.get_node_input_value(mul3_node, 0)
    except_value = np.array(0.7978845834732056, dtype=np.float32)
    if mul3_value != except_value and mul3_value != except_value.astype(np.float16):
        return False

    tanh_node = consumers[0]
    if tanh_node.type != 'Tanh':
        return False
    consumers, _ = tanh_node.get_consumers(0)
    if len(consumers) != 1:
        return False

    add1_node = consumers[0]
    if add1_node.type != 'Add':
        return False
    consumers, _ = add1_node.get_consumers(0)
    if len(consumers) != 1:
        return False
    add1_value = QuantOpInfo.get_node_input_value(add1_node, 0)
    if add1_value != 1:
        return False

    mul4_node = consumers[0]
    if mul4_node.type != MUL or mul4_node is not input_nodes[4]:
        return False
    consumers, _ = mul4_node.get_consumers(0)
    if len(consumers) != 1:
        return False

    mul5_node = consumers[0]
    if mul5_node.type != MUL:
        return False
    mul5_value = QuantOpInfo.get_node_input_value(mul5_node, 0)
    if mul5_value != 0.5:
        return False

    return True


def _match_gelu_tanh_subgraph(input_nodes):
    '''
    Function: judge whether match gelu('tanh') subgraph 
    Parameters:
        input_nodes: input_nodes of subgraph
    Return:
        True: match
        False: mismatch
    '''
    if len(input_nodes) != 5:
        return False

    if not _match_gelu_tanh_subgraph_front_part(input_nodes):
        return False

    if not _match_gelu_tanh_subgraph_back_part(input_nodes):
        return False

    return True


def _match_rrelu_subgraph(input_nodes):
    '''
    Function: judge whether match rrelu subgraph
        torch 2.1 earlier, rrelu export to RandomUniformLike + PRelu
        / \
        | |
        | RandomUniformLike
        | |
        \ /
        PRelu
    Parameters:
        input_nodes: input_nodes of subgraph
    Return:
        True: has rrelu
        False: don't have rrelu
    '''
    if len(input_nodes) != 2:
        return False

    rand_node = input_nodes[0]
    if rand_node.type != 'RandomUniformLike':
        return False
    consumers, _ = rand_node.get_consumers(0)
    if len(consumers) != 1:
        return False

    prelu_node = consumers[0]
    if prelu_node.type != 'PRelu' or prelu_node is not input_nodes[1]:
        return False

    return True


def _get_other_activation(consumers):
    '''
    Function: get activation function of quant layer
    Parameters:
        consumers: consumers of quant layer
    Return:
        act_module: activation function
    '''
    if len(consumers) != 1:
        return None

    act_module = None
    consumer = consumers[0]
    if consumer.type in ('Relu', 'Sigmoid', 'Tanh'):
        act_module = ACTIVATION_MAP.get(consumer.type)
    elif consumer.type == 'LeakyRelu':
        attrs_helper = AttributeProtoHelper(consumer.proto)
        alpha = attrs_helper.get_attr_value('alpha')
        act_module = torch.nn.LeakyReLU(negative_slope=alpha)
    elif consumer.type == 'PRelu':
        slope_value = QuantOpInfo.get_node_input_value(consumer, 1).flatten()
        act_module = torch.nn.PReLU(num_parameters=slope_value.size)
        act_module.weight = torch.nn.Parameter(torch.from_numpy(slope_value), requires_grad=False)
    elif consumer.type == 'Clip':
        if len(consumer.input_anchors) != 3:
            return None
        clip_min_value = QuantOpInfo.get_node_input_value(consumer, 1)
        clip_max_value = QuantOpInfo.get_node_input_value(consumer, 2)
        if clip_min_value != 0 or clip_max_value != 6:
            return None
        act_module = torch.nn.ReLU6()

    return act_module


def get_ada_round_groups(graph, quant_config):
    '''
    Function: get ada round quant groups
    Parameters:
        graph: model in amct ir
        quant_config: config which parsed from json file
    Return:
        quant_groups: [[quant_layer_name, activation_module], ...]
            if don't have activation module, it's None
    '''
    ada_round_groups = list()
    for node in graph.nodes:
        if not _is_weight_ada_round(quant_config, node.name):
            continue

        consumers, _ = node.get_consumers(0)
        if _match_gelu_subgraph(consumers):
            act_module = torch.nn.GELU()
        elif _match_gelu_tanh_subgraph(consumers):
            act_module = torch.nn.GELU('tanh')
        elif _match_rrelu_subgraph(consumers):
            attrs_helper = AttributeProtoHelper(consumers[0].proto)
            high = attrs_helper.get_attr_value('high')
            low = attrs_helper.get_attr_value('low')
            act_module = torch.nn.RReLU(lower=low, upper=high)
        else:
            act_module = _get_other_activation(consumers)

        ada_round_groups.append([node.name, act_module])

    return ada_round_groups