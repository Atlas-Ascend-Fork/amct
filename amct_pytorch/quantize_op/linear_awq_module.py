# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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
import math
import copy
import torch
from torch import nn
import torch.nn.functional as F

from amct_pytorch.quantize_op.base_quant_module import BaseQuantizeModule
from amct_pytorch.utils.data_utils import check_linear_input_dim
from amct_pytorch.algorithm.awq import search_scale, apply_scale, calculate_scale_offset_by_granularity
from amct_pytorch.utils.vars import INT4, INT8
from amct_pytorch.utils.log import LOGGER


class LinearAWQuant(BaseQuantizeModule):
    """
    Function: Customized torch.nn.Module of the LinearAWQuant class.
    APIs: forward.
    """
    def __init__(self,
                 ori_module,
                 layer_name,
                 quant_config):
        """
        Function: init objective.
        Args:
        ori_module: torch module. Linear.
        layer_name: ori_module's name.
        quant_config: calibration algorithm parameters.
        """
        super().__init__(ori_module, layer_name, quant_config)
        self.weight = copy.deepcopy(ori_module.weight)
        self.bias = ori_module.bias
        self.layer_name = layer_name
        self.quant_config = quant_config
        self.wts_symmetric = quant_config.get('weights_cfg').get('symmetric')
        self.ori_module = ori_module
        self.wts_type = self.quant_config.get('weights_cfg').get('quant_type')
        if self.quant_config.get('weights_cfg').get("group_size") is not None:
            self.group_size = self.quant_config.get('weights_cfg').get("group_size")
        self.calc_done = False
    
    @torch.no_grad()
    def forward(self, inputs):
        """
        Function: LinearAWQuant foward funtion.
        Args:
        inputs: data used for calibration in torch.tensor.
        """
        input_data = inputs.clone()
        with torch.no_grad():
            check_linear_input_dim(input_data)
            input_data = input_data.to(self.weight.device)
            output = F.linear(input_data, self.weight, self.bias)
        if self.calc_done:
            return output

        scale_awq = search_scale(input_data, [self.ori_module], self.ori_module, self.quant_config)
        apply_scale(scale_awq, self.ori_module, input_data)
        self.scale = 1 / scale_awq.detach()

        self.scale_w, self.offset_w = \
            calculate_scale_offset_by_granularity(self.ori_module.weight.data, self.quant_config)
        self.calc_done = True
        LOGGER.logd("Calculate awq quant params of layer '{}' success!".format(self.layer_name), 'LinearAWQuant')
        return output