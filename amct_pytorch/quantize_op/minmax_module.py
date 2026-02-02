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
import torch
from torch import nn

from amct_pytorch.quantize_op.base_quant_module import BaseQuantizeModule
from amct_pytorch.quantize_op.utils import calculate_scale_offset
from amct_pytorch.quantize_op.utils import calculate_progressive_weights_scale_factor
from amct_pytorch.quantize_op.utils import get_weight_min_max_by_granularity
from amct_pytorch.utils.data_utils import check_linear_input_dim
from amct_pytorch.utils.vars import FLOAT8_E4M3FN, FLOAT4_E2M1
from amct_pytorch.utils.log import LOGGER


class MinMaxQuant(BaseQuantizeModule):
    """
    Function: calibration operator to obtain act quant factors using min-max algo
    APIs: forward.
    """
    def __init__(self,
                 ori_module,
                 layer_name,
                 quant_config):
        """
        Function: init objective.
        Args:
        ori_module: torch module. Quantized_type.
        layer_name: ori_module's name.
        quant_config: calibration algorithm parameters.
        """
        super().__init__(ori_module, layer_name, quant_config)
        self.ori_module_type = type(ori_module).__name__
        self.weight = ori_module.weight
        self.bias = ori_module.bias
        self.layer_name = layer_name
        self.cur_batch = 0
        self.weight_compress_only = True
        self.ori_module = ori_module
        self.scale_w1 = None
        self.scale_w2 = None

        self.batch_num = quant_config.get('batch_num')
        if quant_config.get('inputs_cfg').get('enable_quant') is None or \
            quant_config.get('inputs_cfg').get('enable_quant') == True:
            self.weight_compress_only = False
            self.act_type = quant_config.get('inputs_cfg').get('quant_type')
            self.act_symmetric = quant_config.get('inputs_cfg').get('symmetric')
            self.act_granularity = quant_config.get('inputs_cfg').get('strategy')
            self.data_max = None
            self.data_min = None

        self.wts_type = quant_config.get('weights_cfg').get('quant_type')
        self.wts_symmetric = quant_config.get('weights_cfg').get('symmetric')
        self.group_size = quant_config.get('weights_cfg').get('group_size', None)
        self.weight_granularity = quant_config.get('weights_cfg').get('strategy')

        if self.act_type == FLOAT8_E4M3FN and self.wts_type == FLOAT4_E2M1:
            self.scale_w1, self.scale_w2 = \
                calculate_progressive_weights_scale_factor(self.ori_module.weight.data)
        else:
            self.scale_w, self.offset_w = \
                self.calculate_weights_scale_factor_and_quantize(self.weight.data, quant_config)

    def calculate_weights_scale_factor_and_quantize(self, weight_data, quant_config):
        """
        Function: calculate weights's quant factor and do fakequant
        Parameters: quant_config: configuration of quantization
        """
        weight_min, weight_max = get_weight_min_max_by_granularity(weight_data, quant_config)

        scale_w, offset_w = calculate_scale_offset(weight_max, weight_min, 
                                            self.wts_symmetric, self.wts_type)
        return scale_w, offset_w

    @torch.no_grad()
    def forward(self, inputs):
        """
        Function: MinMaxQuant forward function.
        Args:
        inputs: data used for calibration in torch.tensor.
        """
        check_linear_input_dim(inputs)
        fp_out = self.ori_module(inputs)

        if self.weight_compress_only:
            return fp_out

        self.cur_batch += 1
        if self.cur_batch > self.batch_num:
            return fp_out
        if self.act_granularity == 'tensor':
            batch_max = torch.max(inputs).reshape(-1)
            batch_min = torch.min(inputs).reshape(-1)
        elif self.act_granularity == 'token':
            cout_axis = -2
            axis_list = list(range(0, inputs.dim()))
            axis_list.pop(cout_axis)
            axis = tuple(axis_list)
            batch_max = torch.amax(inputs, dim=axis, keepdim=True)
            batch_min = torch.amin(inputs, dim=axis, keepdim=True)

        self.data_max = batch_max if self.data_max is None else self.data_max
        self.data_min = batch_min if self.data_min is None else self.data_min
        self.data_max = torch.where(batch_max > self.data_max, batch_max, self.data_max)
        self.data_min = torch.where(batch_min < self.data_min, batch_min, self.data_min)

        if self.cur_batch == self.batch_num:
            self.scale_d, self.offset_d = calculate_scale_offset(
                self.data_max, self.data_min, self.act_symmetric, self.act_type)

            LOGGER.logd("Calculate minmax quant params of layer '{}' success!".format(self.layer_name), 'MinMaxQuant')
        return fp_out
