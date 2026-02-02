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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from ...amct_pytorch.capacity import CAPACITY
from ...amct_pytorch.common.utils.onnx_node_util import AttributeProtoHelper
from ...amct_pytorch.common.utils.vars_util import INT4, INT8, INT16
from ...amct_pytorch.common.utils.vars_util import RNN_LAYER_TYPE, RNN_TENSOR_NUM
from ...amct_pytorch.common.utils.vars_util import QUANT_INDEXES_MAP
from ...amct_pytorch.utils.onnx_initializer_util import TensorProtoHelper
from ...amct_pytorch.utils.weight_quant_api import get_deconv_group


class QuantOpInfo():
    '''
    Find infomation of quant_op.
    '''
    @staticmethod
    def get_scale_shape(node, channel_wise):
        """
        Function: Get the weights' scale's shape of node.
        Inputs:
            node: Node, it's type should be in QUANTIZABLE_ONNX_TYPES.
            channel_wise: a bool, parameter of quantization.
        Returns:
            shape: a list, the shape of scale.
            scale_length : a number, the length of scale.
        """

        if node.type not in CAPACITY.get_value('QUANTIZABLE_ONNX_TYPES') and \
            node.type not in CAPACITY.get_value('RETRAIN_ONNX_TYPES'):
            raise RuntimeError(
                'Not supported get scale shape from type:%s(%s)' %
                (node.type, node.name))
        if channel_wise and node.type in CAPACITY.get_value(
                'CHANNEL_WISE_ONNX_TYPES'):
            weight_param = QuantOpInfo.get_weight_tensor(node)
            if node.type == 'ConvTranspose':
                # deconv2d or deconv3d
                group = get_deconv_group(node)
                length = weight_param.dims[1] * group
                shape = [1] * len(weight_param.dims)
                shape[1] = length
            elif node.type in RNN_LAYER_TYPE:
                length = weight_param.dims[1]
                shape = [1] * len(weight_param.dims)
                shape[1] = length
            else:
                # conv2D or conv3D
                length = weight_param.dims[0]
                shape = [1] * len(weight_param.dims)
                shape[0] = weight_param.dims[0]
        else:
            if node.type in RNN_LAYER_TYPE:
                length = RNN_TENSOR_NUM.get(node.type)
                shape = [length]
            else:
                shape = []
                length = 1

        return shape, length

    @staticmethod
    def get_quant_index(node):
        """
        Function: get act's index and weight's index of node.
        Inputs:
            node: Node, it's type should be in QUANTIZABLE_ONNX_TYPES.
        Returns:
            act_index: the act's index in inputs of node.
            weight_index: the weight's index in inputs of node.
        """
        if node.type not in CAPACITY.get_value('QUANTIZABLE_ONNX_TYPES') and \
            node.type not in CAPACITY.get_value('RETRAIN_ONNX_TYPES'):
            raise RuntimeError("%s is not supported." % (node.type))

        if node.type not in QUANT_INDEXES_MAP:
            raise RuntimeError(f'Node {node.name} type {node.type} cannot get quant index map.')
        return QUANT_INDEXES_MAP.get(node.type)

    @staticmethod
    def get_dequant_shape(node):
        """
        Function: Get the dequant scale's shape from node
        Inputs: node: the node te be quantized
        Returns: the shape of dequant scale
        """
        if node.type in ["Conv", "AscendDequant", "ConvTranspose"]:
            dequant_shape = [1, -1, 1, 1]
            if node.type == 'Conv' or node.type == 'ConvTranspose':
                conv_weights_node = QuantOpInfo.get_weight_node(node)
                weight_tensor = QuantOpInfo.get_node_tensor(conv_weights_node)
                if len(weight_tensor.dims) == 5:
                    dequant_shape = [1, -1, 1, 1, 1]
                elif len(weight_tensor.dims) == 3:
                    dequant_shape = [1, -1, 1]
        elif node.type in ["Gemm", "MatMul", "AveragePool", "Add"]:
            dequant_shape = [1, -1]
        else:
            raise RuntimeError("%s is not supported." % (node.type))

        return dequant_shape

    @staticmethod
    def get_weight_node(node):
        '''
        Function: Get quantizable_node's weight node
        Parameters:
            quantizable_node: a node, which is quantizable
        Return:
            weight_param: a node, it's quantizable_node' weight
        '''
        weight_index = QuantOpInfo.get_quant_index(node).get('weight_index')

        weight_input_anchor = node.get_input_anchor(weight_index)
        weight_param = weight_input_anchor.get_peer_output_anchor().node

        if weight_param.type == 'Transpose':
            weight_input_anchor = weight_param.get_input_anchor(0)
            weight_param = weight_input_anchor.get_peer_output_anchor().node
            node.set_attr('with_weights_trans', True)

        return weight_param

    @staticmethod
    def get_recurrence_weight_node(quantizable_node):
        '''
        Function: Get quantizable rnn node's recurrence weight node
        Parameters:
            quantizable_node: a rnn node, which is quantizable
        Return:
            recurrence_weight_param: a node, it's quantizable rnn node's recurrence weight
        '''
        recurrence_weight_index = QuantOpInfo.get_quant_index(quantizable_node).get('recurrence_weight_index')
        recurrence_weight_in_anchor = quantizable_node.get_input_anchor(recurrence_weight_index)
        recurrence_weight_param = recurrence_weight_in_anchor.get_peer_output_anchor().node
        return recurrence_weight_param

    @staticmethod
    def get_bias_node(quantizable_node):
        '''
        Function: Get quantizable_node's bias node
        Parameters:
            quantizable_node: a node, which is quantizable
        Return:
            bias_param: a node, it's quantizable_node' bias
        '''
        if quantizable_node.type == 'MatMul':
            return QuantOpInfo.get_bias_for_matmul(quantizable_node)

        bias_index = QuantOpInfo.get_quant_index(quantizable_node).get('bias_index')
        if bias_index is None or \
                bias_index >= len(quantizable_node.input_anchors):
            return None

        bias_in_anchor = quantizable_node.get_input_anchor(bias_index)
        bias_output_anchor = bias_in_anchor.get_peer_output_anchor()
        if not bias_output_anchor:
            return None
        bias_param = bias_output_anchor.node

        return bias_param

    @staticmethod
    def get_weight_tensor(node):
        '''
        Function: Get quantizable_node's weight node
        Parameters:
            quantizable_node: a node, which is quantizable
        Return:
            weight_param: a node, it's quantizable_node' weight
        '''
        weight_node = QuantOpInfo.get_weight_node(node)
        weight_tensor = QuantOpInfo.get_node_tensor(weight_node)

        return weight_tensor

    @staticmethod
    def get_cout_length(node):
        """
        Function: Get cout length of the given node
        Parameter: node
        Return: cout_length
        """
        if node.type not in CAPACITY.get_value('PASSIVE_PRUNABLE_ONNX_TYPES'):
            raise RuntimeError("Unexpected node's type {} for {}".format(node.type, node.name))
        cout_length = None
        if node.type == 'BatchNormalization':
            scale_node, _ = node.get_producer(1)
            tensor = QuantOpInfo.get_node_tensor(scale_node)
            cout_length = tensor.dims[0]
        else:
            # 'Conv', 'ConvTranspose', 'Gemm', 'MatMul'
            tensor = QuantOpInfo.get_weight_tensor(node)
            if node.type == 'Conv':
                cout_length = tensor.dims[0]
            elif node.type in ['ConvTranspose', 'MatMul']:
                # group conv case
                cout_length = tensor.dims[1]
            elif node.type == 'Gemm':
                attr_helper = AttributeProtoHelper(node.proto)
                if not attr_helper.has_attr('transB') or attr_helper.get_attr_value('transB') == 0:
                    cout_length = tensor.dims[1]
                else:
                    cout_length = tensor.dims[0]
        return cout_length

    @staticmethod
    def get_cin_length(node):
        """
        Function: Get cin length of the given node
        Parameter: node
        Return: cin_length
        """
        if node.type not in CAPACITY.get_value('QUANTIZABLE_ONNX_TYPES'):
            raise RuntimeError("Unexpected node's type {} for {}".format(node.type, node.name))
        cin_length = None
        tensor = QuantOpInfo.get_weight_tensor(node)
        if node.type == 'Conv':
            group = get_deconv_group(node)
            cin_length = tensor.dims[1] * group
        elif node.type in ['ConvTranspose', 'MatMul']:
            cin_length = tensor.dims[0]
        elif node.type == 'Gemm':
            attr_helper = AttributeProtoHelper(node.proto)
            if not attr_helper.has_attr('transB') or attr_helper.get_attr_value('transB') == 0:
                cin_length = tensor.dims[0]
            else:
                cin_length = tensor.dims[1]
        return cin_length

    @staticmethod
    def get_node_tensor(node):
        '''
        Function: Get node's value tensor
        Parameters:
            node: a node, which contain value, usually is initializer or constant
        Return:
        '''
        node_type = node.type
        if node_type not in ['initializer', 'Constant']:
            raise RuntimeError("Do not support get tensor from node {} with type {}".format(node.name, node_type))
        if node_type == 'initializer':
            # for 'initializer' type
            node_tensor = node.proto
        else:
            # for 'Constant' type
            attr_helper = AttributeProtoHelper(node.proto)
            node_tensor = attr_helper.get_attr_value('value')

        return node_tensor

    @staticmethod
    def get_node_value(node):
        '''
        Function: Get node's value
        Parameters:
            node: a node, which contain value, usually is initializer or constant
        Return:
        '''
        node_tensor = QuantOpInfo.get_node_tensor(node)
        tensor_helper = TensorProtoHelper(node_tensor, node.model_path)
        node_value = tensor_helper.get_data()

        return node_value

    @staticmethod
    def get_node_input_value(node, input_index):
        '''
        Function: Get node's input value
        Parameters:
            node: a node
            input_index: input index of node
        Return:
            value of input index
        '''
        if len(node.input_anchors) <= input_index:
            return np.array(np.nan)

        output_anchor = node.get_input_anchor(input_index).get_peer_output_anchor()
        if output_anchor is None:
            return np.array(np.nan)

        node_input = output_anchor.node
        try:
            node_input_tensor = QuantOpInfo.get_node_tensor(node_input)
        except RuntimeError:
            return np.array(np.nan)
        tensor_helper = TensorProtoHelper(node_input_tensor, node_input.model_path)
        node_input_value = tensor_helper.get_data()

        return node_input_value

    @staticmethod
    def get_dst_num_bits(records, op_name, data_type=None):
        """
        Function: check and return dst_type
        Inputs:
            records: records dict
            op_name: name of operation
            data_type: act or wts
        Return: int num_bit
        """
        if records is None or records.get(op_name) is None:
            raise RuntimeError(
                'records is None or layer [{}] not in records.'.format(op_name))

        default_num_bits = 8
        if data_type is not None:
            if data_type == 'act':
                dst_type_dict = {INT4: 4, INT8: 8, INT16: 16}
                dst_type = records.get(op_name).get('act_type')
                if dst_type in dst_type_dict:
                    num_bit = dst_type_dict.get(dst_type)
                elif dst_type == 'UNSET':
                    num_bit = default_num_bits
                else:
                    raise RuntimeError(
                        'act_type in layer [{}] is not INT8 or INT16,'
                        'actual value is {}'.format(op_name, dst_type))
            elif data_type == 'wts':
                dst_type_dict = {INT4: 4, 'INT6': 6, 'INT7': 7, INT8: 8}
                dst_type = records.get(op_name).get('wts_type')
                if dst_type in dst_type_dict:
                    num_bit = dst_type_dict.get(dst_type)
                elif dst_type == 'UNSET':
                    num_bit = default_num_bits
                else:
                    raise RuntimeError(
                        'wts_type in layer [{}] is not INT8, '
                        'actual value is {}'.format(op_name, dst_type))
        elif 'dst_type' in records.get(op_name):
            dst_type_dict = {INT4: 4, INT8: 8}
            dst_type = records.get(op_name).get('dst_type')
            if dst_type in dst_type_dict:
                num_bit = dst_type_dict.get(dst_type)
            else:
                raise RuntimeError(
                    'dst_type in layer [{}] is not INT4 or INT8,'
                    'actual value is {}'.format(op_name, dst_type))
        else:
            num_bit = default_num_bits
        return num_bit

    @staticmethod
    def get_bias_for_matmul(matmul_node):
        '''
        Function: Get matmul_node's bias node
        param:matmul_node: a node, which type is MatMul
        return:bias_param: a node, it's quantizable_node' bias. return None when no bias
        '''
        consumers, _ = matmul_node.get_consumers(0)
        if len(consumers) != 1 or consumers[0].type != 'Add':
            return None
        bias_node, _ = consumers[0].get_producer(1)
        if bias_node is None or bias_node.type not in ['Constant', 'initializer']:
            return None

        bias_value = QuantOpInfo.get_node_value(bias_node)
        if len(bias_value.shape) != 1:
            return None

        weight_node = QuantOpInfo.get_weight_node(matmul_node)
        if weight_node is not None and weight_node.type in ['Constant', 'initializer']:
            weight_value = QuantOpInfo.get_node_value(weight_node)
            if bias_value.shape[0] != weight_value.shape[-1]:
                return None
        else:  # dual input matmul
            if bias_value.shape[0] == 1:
                return None
        return bias_node
