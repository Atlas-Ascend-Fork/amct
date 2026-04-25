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

from ...amct_pytorch.utils.log import LOGGER
from ...amct_pytorch.utils.onnx_initializer_util import TensorProtoHelper
from ...amct_pytorch.utils.quant_node import QuantOpInfo
from ...amct_pytorch.utils.weight_quant_api import get_deconv_group
from ...amct_pytorch.utils.weight_quant_api import adjust_deconv_weight_shape
from ...amct_pytorch.common.utils.vars_util import RNN_LAYER_TYPE

WTS_FAKEQUANT_NUM_BITS = (6, 7, 8)


class WeightFakequantPass(BaseFusionPass):
    """
    Function: Fakequant weight from int8 to int9
    APIs: match_pattern, do_pass
    """
    def __init__(self, records):
        """
        Function: init object
        Parameter:
            records: dict including quant factors such as scale_w
            num_bits: int number indicating the bit to be quanted such as 8
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
        if node.type == 'AveragePool':
            return False

        if node.name not in self.records:
            return False

        # Determine the quantization type based on the value of dst_type and obtain the corresponding num_bits
        if self.records.get(node.name).get('dst_type') == 'UNSET':
            num_bits = QuantOpInfo.get_dst_num_bits(self.records, node.name, 'wts')
        else:
            num_bits = QuantOpInfo.get_dst_num_bits(self.records, node.name)
        if num_bits not in WTS_FAKEQUANT_NUM_BITS:
            return False
        return True

    def do_pass(self, graph, object_node, model=None):
        """
        Function: Do actual quantization and node's weight is changed to int9.
        Parameters:
            graph: graph structure
            object_node: node to process
            model: torch.nn.Module, the model to be modified. if it's
                None, the gaph will be modified.
        Return: None
        """
        if object_node.type in RNN_LAYER_TYPE:
            self.dequant_weight(object_node)
            self.dequant_weight(object_node, True)
            return
        weight_param = QuantOpInfo.get_weight_node(object_node)
        weight_helper = TensorProtoHelper(weight_param.proto, weight_param.model_path)
        # get data
        int8_weight = weight_helper.get_data()
        if object_node.type == 'ConvTranspose' and get_deconv_group(object_node) > 1:
            group = get_deconv_group(object_node)
            int8_weight = adjust_deconv_weight_shape(group, int8_weight)
            trans_axes = (1, 0, 2, 3, 4)[:len(int8_weight.shape)]
            int8_weight = np.transpose(int8_weight, trans_axes)
        weight_helper.clear_data()

        weight_offset = self.records.get(object_node.name).get('weight_offset')
        int9_weight = int8_weight.astype(np.float32)
        if not np.all(weight_offset == 0):
            int9_weight = int9_weight - weight_offset.astype(np.float32)

        if object_node.type == 'ConvTranspose' and get_deconv_group(object_node) > 1:
            group = get_deconv_group(object_node)
            trans_axes = (1, 0, 2, 3, 4)[:len(int9_weight.shape)]
            int9_weight = np.transpose(int9_weight, trans_axes)
            int9_weight = adjust_deconv_weight_shape(group, int9_weight)

        int9_weight = int9_weight.reshape([-1])

        weight_helper.set_data(int9_weight, 'FLOAT')

        LOGGER.logd("Fakequant weight from int8 to int9 for layer '{}' " \
            "success!".format(object_node.name), 'WeightFakequantPass')

    def dequant_weight(self, object_node, is_recurrence_weight=False):
        """
        Function: dequant weight of rnn node to convert it to float32
        Parameters:
            object_node: node to process
        Return: None
        """
        if not is_recurrence_weight:
            weight_node = QuantOpInfo.get_weight_node(object_node)
            scale = self.records.get(object_node.name).get('weight_scale')
            offset = self.records.get(object_node.name).get('weight_offset')
        else:
            weight_node = QuantOpInfo.get_recurrence_weight_node(object_node)
            scale = self.records.get(object_node.name).get('recurrence_weight_scale')
            offset = self.records.get(object_node.name).get('recurrence_weight_offset')
        weight_tensor = QuantOpInfo.get_node_tensor(weight_node)
        weight_helper = TensorProtoHelper(weight_tensor, model_path=weight_node.model_path)
        weight = weight_helper.get_data().astype(np.float32)
        weight_helper.clear_data()

        if weight.shape[1] != scale.size or len(scale.shape) != 3:
            scale = np.repeat(scale, weight.shape[1] // scale.size)
            scale = scale.reshape(1, scale.size, 1)
            offset = np.repeat(offset, weight.shape[1] // offset.size)
            offset = offset.reshape(1, offset.size, 1)

        # Dequant w to fp32 to introduce quant error
        weight = (weight - offset) * scale.astype(np.float32)
        weight_helper.set_data(weight.reshape(-1), 'FLOAT')
