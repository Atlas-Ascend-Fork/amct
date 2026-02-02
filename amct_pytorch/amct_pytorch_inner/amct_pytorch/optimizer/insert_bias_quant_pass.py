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
import numpy as np
from onnx import onnx_pb # pylint: disable=import-error

from ...amct_pytorch.optimizer.base_fusion_pass import BaseFusionPass
from ...amct_pytorch.utils.quant_node import QuantOpInfo
from ...amct_pytorch.utils.vars import QUANT_BIAS_BITS
from ...amct_pytorch.utils.vars import BASE
from ...amct_pytorch.utils.log import LOGGER
from ...amct_pytorch.utils.onnx_initializer_util import TensorProtoHelper
from ...amct_pytorch.common.utils.onnx_node_util import AttributeProtoHelper
from ...amct_pytorch.common.utils.vars_util import RNN_LAYER_TYPE


class InsertBiasQuantPass(BaseFusionPass):
    """
    Function: Quant weight from float32 to int8
    APIs: match_pattern, do_pass, quant_weight
    """
    def __init__(self, records):
        """
        Function: init object
        Parameter: None
        Return: None
        """
        BaseFusionPass.__init__(self)
        self.records = records

    @staticmethod
    def quant_bias(bias, scale_w, scale_d, layer_name):
        '''Function: kernel function, quant bias with scale and offset
        Parameters:
            bias: np.array, to be quantized
            scale: float number, quant factor
            offset: int number, quant factor
        Returns:
            quant_bias, np.array with type int32, quantized bias
        '''
        left_bound = -pow(1.0 * BASE, QUANT_BIAS_BITS - 1)
        right_bound = pow(1.0 * BASE, QUANT_BIAS_BITS - 1) - 1
        deq_scale = np.multiply(scale_w, scale_d).reshape([-1])
        quant_bias = np.round(np.true_divide(bias, deq_scale))
        # check the quant_bias in range of int32
        quant_bias = quant_bias.reshape(-1)
        cmp_ans = np.add(quant_bias < left_bound, quant_bias > right_bound)
        if cmp_ans.any():
            invalid_value = quant_bias[np.argmax(cmp_ans)]
            LOGGER.loge('Quantized bias {} of layer "{}" exceed int32 ' \
                'range:[{}, {}], please add it to skip layer.'.format(
                    invalid_value, layer_name,
                    left_bound, right_bound))
            raise RuntimeError('Do bias quantize failed.')
        quant_bias = quant_bias.astype(np.int32)
        return quant_bias

    def match_pattern(self, node):
        """
        Function: Match the node to be quantized in graph
        Parameters: node: node in graph to be matched
        Return: True: matched
                False: mismatch
        """
        if node.type == 'AveragePool':
            return False

        if node.name not in self.records:
            return False

        bias_node = QuantOpInfo.get_bias_node(node)
        if bias_node is None:
            return False

        return True

    def do_pass(self, graph, object_node, model=None):
        """
        Function: Do actual quantization and node's bias is changed to int32.
        Parameters:
            graph: graph structure
            object_node: node to process
            model: torch.nn.Module, the model to be modified. if it's
                None, the gaph will be modified.
        Return: None
        """
        bias_param = QuantOpInfo.get_bias_node(object_node)
        bias_helper = TensorProtoHelper(bias_param.proto, bias_param.model_path)
        bias = bias_helper.get_data()
        bias_helper.clear_data()
        if object_node.type in RNN_LAYER_TYPE:
            int32_bias = self.bias_quant_rnn(bias, object_node.name)
        else:
            int32_bias = InsertBiasQuantPass.quant_bias(
                bias,
                self.records.get(object_node.name).get('weight_scale'),
                self.records.get(object_node.name).get('data_scale'),
                object_node.name)
        bias_helper.set_data(int32_bias, 'INT32')

        # Determine the quantization type based on the value of dst_type and obtain the corresponding num_bits
        if self.records.get(object_node.name).get('dst_type') == 'UNSET':
            num_bits = QuantOpInfo.get_dst_num_bits(self.records, object_node.name, 'act')
        else:
            num_bits = QuantOpInfo.get_dst_num_bits(self.records, object_node.name)

        if num_bits == 4:
            # Step1: add a new_node
            node_proto = construct_bias_quant_node(
                inputs=['.'.join([object_node.name, 'bias_quant', 'input0'])],
                outputs=['.'.join([object_node.name, 'bias_quant', 'output0'])],
                layer_name=object_node.name)
            bias_quant_node = graph.add_node(node_proto)

            # Step2: Relink nodes in th graph
            # remove output links
            bias_index = QuantOpInfo.get_quant_index(object_node).get('bias_index')
            graph.remove_edge(bias_param, 0,
                            object_node, bias_index)
            # add links
            graph.add_edge(bias_param, 0, bias_quant_node, 0)
            graph.add_edge(bias_quant_node, 0, object_node, bias_index)

        LOGGER.logd("Quant bias from float32 to int32 for layer '{}' " \
            "success!".format(object_node.name), 'BiasQuantPass')

    def bias_quant_rnn(self, bias, layer_name):
        """
        Function: Quant bias data for rnn
        Parameters: bias: bias data of quantized rnn op
        Return: bias_int32: quantized bias
        """
        bias_length = bias.shape[1]
        weight_bias = bias.flatten()[:bias_length // 2]
        recurrence_weight_bias = bias.flatten()[bias_length // 2:]

        # broad scale_w in per-tensor
        scale_w = self.records.get(layer_name).get('weight_scale')
        if weight_bias.size != scale_w.size:
            scale_w = np.repeat(scale_w, weight_bias.size // scale_w.size)
        # do real quant for bias corresponding to w and r separately
        scale_d = self.records.get(layer_name).get('data_scale')
        weight_bias_int32 = InsertBiasQuantPass.quant_bias(weight_bias,
                                                           scale_w,
                                                           scale_d,
                                                           layer_name)

        # broadcast scale_r in per-tensor
        scale_r = self.records.get(layer_name).get('recurrence_weight_scale')
        if recurrence_weight_bias.size != scale_r.size:
            scale_r = np.repeat(scale_r, recurrence_weight_bias.size // scale_r.size)
        scale_h = self.records.get(layer_name).get('h_scale')
        recurrence_weight_bias_int32 = InsertBiasQuantPass.quant_bias(recurrence_weight_bias,
                                                                      scale_r,
                                                                      scale_h,
                                                                      layer_name)

        bias_int32 = np.concatenate([weight_bias_int32, recurrence_weight_bias_int32], 0)
        return bias_int32


def construct_bias_quant_node(inputs,
                              outputs,
                              layer_name):
    """
    Function: construct quant node in onnx
    Inputs:
        input: a list of inputs' name
        output: a list of outputs' name
        attrs: a dict of attrs including
            scale: numpy.array
            offset: numpy.array
            dst_type: a string
        quantize_layer: a string, layer to be quantized
    """
    node_proto = onnx_pb.NodeProto()

    node_proto.name = '.'.join([layer_name, 'bias_quant'])
    node_proto.op_type = 'Cast'

    cast_dtype = onnx_pb.TensorProto.DataType.FLOAT
    AttributeProtoHelper(node_proto).set_attr_value('to', 'INT', cast_dtype)
    node_proto.input.extend(inputs)
    node_proto.output.extend(outputs)

    return node_proto
