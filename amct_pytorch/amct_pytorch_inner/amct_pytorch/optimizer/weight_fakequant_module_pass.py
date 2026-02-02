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

from torch import Tensor # pylint: disable=E0401
from torch.nn.parameter import Parameter # pylint: disable=E0401

from ...amct_pytorch.optimizer.base_module_fusion_pass import BaseModuleFusionPass
from ...amct_pytorch.custom_op.arq.arq import weight_quant_np
from ...amct_pytorch.custom_op.fake_quant import FAKE_MODULES
from ...amct_pytorch.custom_op.fake_quant import FAKE_CONV_TRANSPOSE
from ...amct_pytorch.custom_op.fake_quant import FAKE_CONV
from ...amct_pytorch.utils.log import LOGGER
from ...amct_pytorch.utils.weight_quant_api import adjust_deconv_weight_shape


class WeightFakequantModulePass(BaseModuleFusionPass):
    """
    Function: Fakequant weight from int8 to int9
    APIs: match_pattern, do_pass
    """
    def __init__(self, records, num_bits):
        """
        Function: init object
        Parameter:
            records: dict including quant factors such as scale_w
            num_bits: int number indicating the bit to be quanted such as 8
        Return: None
        """
        BaseModuleFusionPass.__init__(self)
        self.num_bits = num_bits
        self.records = records

    def match_pattern(self, module, name, graph=None):
        """
        Function: Match pattern of node to be quantized in graph
        Parameters: node: node in graph to be matched
        Return: True: matched
                False: mismatch
        """
        if name not in self.records:
            return False

        if type(module).__name__ in FAKE_MODULES:
            return True

        return False

    def do_pass(self, model, object_module, object_name, graph=None):
        """
        Function: Do actual quantization and node's weight is changed to int9.
        Parameters:
            graph: graph structure
            object_node: node to process
            model: torch.nn.Module, the model to be modified. if it's
                None, the graph will be modified.
        Return: None
        """
        weight_np = object_module.sub_module.weight.cpu().detach().numpy()
        weight_offset = self.records.get(object_name).get('weight_offset')
        ori_weight_shape = weight_np.shape
        if type(object_module).__name__ == FAKE_CONV_TRANSPOSE:
            group = object_module.sub_module.groups
            weight_np = adjust_deconv_weight_shape(group, weight_np)
            trans_axes = (1, 0, 2, 3, 4)[:len(weight_np.shape)]
            weight_np = weight_np.transpose(trans_axes)
        int8_weight = weight_quant_np(
            weight_np,
            self.records.get(object_name).get('weight_scale'),
            self.records.get(object_name).get('weight_offset'),
            self.num_bits)
        if type(object_module).__name__ == FAKE_CONV_TRANSPOSE:
            weight_offset = weight_offset.astype(np.float32).reshape(
                [1, -1, 1, 1])
        elif type(object_module).__name__ == FAKE_CONV:
            reshaped_weight_shape = [1] * len(object_module.sub_module.weight.shape)
            reshaped_weight_shape[0] = -1
            weight_offset = weight_offset.astype(np.float32).reshape(
                reshaped_weight_shape)

        int9_weight = int8_weight.astype(np.float32) - weight_offset
        if type(object_module).__name__ == FAKE_CONV_TRANSPOSE:
            trans_axes = (1, 0, 2, 3, 4)[:len(weight_np.shape)]
            int9_weight = int9_weight.transpose(trans_axes)
            group = object_module.sub_module.groups
            weight_np = adjust_deconv_weight_shape(group, int9_weight)
        int9_weight = int9_weight.reshape(ori_weight_shape)
        object_module.sub_module.weight = Parameter(
            Tensor(int9_weight).to(device=object_module.sub_module.weight.device))

        LOGGER.logd("Fakequant weight from float32 to int9 for module '{}' " \
            "success!".format(object_name), 'WeightFakequantModulePass')
