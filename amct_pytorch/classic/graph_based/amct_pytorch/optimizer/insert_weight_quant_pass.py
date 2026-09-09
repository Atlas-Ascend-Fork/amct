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

from ...amct_pytorch.optimizer.base_fusion_pass import BaseFusionPass
from ...amct_pytorch.custom_op.arq.arq import weight_quant_np
from ...amct_pytorch.utils.onnx_initializer_util import TensorProtoHelper
from ...amct_pytorch.utils.quant_node import QuantOpInfo
from ...amct_pytorch.utils.log import LOGGER
from ...amct_pytorch.utils.weight_quant_api import get_deconv_group
from ...amct_pytorch.utils.weight_quant_api import adjust_deconv_weight_shape
from ...amct_pytorch.common.utils.vars_util import RNN_LAYER_TYPE


class InsertWeightQuantPass(BaseFusionPass):
    """
    Function: Quantize weight and write it back into the weight tensor.
    APIs: match_pattern, do_pass
    """

    def __init__(self, records):
        """
        Function: init object
        Parameter:
            records: dict including quant factors such as scale_w
        Return: None
        """
        BaseFusionPass.__init__(self)
        self.records = records

    def match_pattern(self, node):
        """
        Function: Match pattern of node to be quantized in graph
        Parameters: node: node in graph to be matched
        Return: True: matched
                False: mismatch
        """
        if node.name not in self.records or node.type in ['AveragePool']:
            return False
        return True

    def do_pass(self, graph, object_node, model=None):
        """
        Function: Quantize weight by num_bits and write it back into the weight tensor.
        Parameters:
            graph: graph structure
            object_node: node to process
            model: torch.nn.Module, the model to be modified. if it's
                None, the gaph will be modified.
        Return: None
        """
        # Determine the quantization type based on the value of dst_type and obtain the corresponding num_bits
        if self.records.get(object_node.name).get('dst_type') == 'UNSET':
            num_bits = QuantOpInfo.get_dst_num_bits(
                self.records, object_node.name, 'wts'
            )
        else:
            num_bits = QuantOpInfo.get_dst_num_bits(self.records, object_node.name)
        weight_param = QuantOpInfo.get_weight_node(object_node)

        weight_helper = TensorProtoHelper(weight_param.proto, weight_param.model_path)
        weight = weight_helper.get_data().astype(np.float32)

        if object_node.type == 'ConvTranspose':
            group = get_deconv_group(object_node)
            weight = adjust_deconv_weight_shape(group, weight)
        scale_w = self.records.get(object_node.name).get('weight_scale')
        offset_w = self.records.get(object_node.name).get('weight_offset')
        quant_weight = weight_quant_np(weight, scale_w, offset_w, num_bits)
        if object_node.type == 'ConvTranspose':
            group = get_deconv_group(object_node)
            quant_weight = adjust_deconv_weight_shape(group, quant_weight)
        quant_weight = quant_weight.reshape([-1])

        weight_helper.clear_data()
        # num_bits==4 时值域为 [-8,7]，写回 INT4 使 dtype 标签与实际值域一致
        weight_helper.set_data(quant_weight, 'INT4' if num_bits == 4 else 'INT8')

        if object_node.type in RNN_LAYER_TYPE:
            self.quant_recurrence_weight(object_node)

        LOGGER.logd(
            "Quant weight to int{} for layer '{}' success!".format(
                num_bits, object_node.name
            ),
            'WeightQuantPass',
        )

    def quant_recurrence_weight(self, object_node):
        """
        Function: quant recurrence weight of rnn op
        Inputs:
            object_node: node to process
        Returns: None
        """
        recurrence_weight_node = QuantOpInfo.get_recurrence_weight_node(object_node)
        recurrence_weight_tensor = QuantOpInfo.get_node_tensor(recurrence_weight_node)
        recurrence_weight_helper = TensorProtoHelper(
            recurrence_weight_tensor, model_path=recurrence_weight_node.model_path
        )
        recurrence_weight = recurrence_weight_helper.get_data()

        scale_r = self.records.get(object_node.name).get('recurrence_weight_scale')
        offset_r = self.records.get(object_node.name).get('recurrence_weight_offset')

        if self.records.get(object_node.name).get('dst_type') == 'UNSET':
            num_bits = QuantOpInfo.get_dst_num_bits(
                self.records, object_node.name, 'wts'
            )
        else:
            num_bits = QuantOpInfo.get_dst_num_bits(self.records, object_node.name)
        int_recurrence_weight = weight_quant_np(
            recurrence_weight, scale_r, offset_r, num_bits
        )
        int_recurrence_weight = int_recurrence_weight.reshape([-1])
        recurrence_weight_helper.clear_data()
        # num_bits==4 时值域为 [-8,7]，写回 INT4 使 dtype 标签与实际值域一致
        recurrence_weight_helper.set_data(
            int_recurrence_weight, 'INT4' if num_bits == 4 else 'INT8'
        )
        LOGGER.logd(
            "Quant recurrence_weight to int{} for layer '{}'".format(
                num_bits, object_node.name
            ),
            'WeightQuantPass',
        )
