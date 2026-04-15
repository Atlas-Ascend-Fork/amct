# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------
import torch
from torch import nn
import torch.nn.functional as F
from amct_pytorch.quantize_op.base_quant_module import BaseQuantizeModule
from amct_pytorch.quantize_op.utils import get_weight_min_max_by_granularity, calculate_quantile_ema_scale
from amct_pytorch.utils.data_utils import check_linear_input_dim
from amct_pytorch.utils.vars import HIFLOAT8
from amct_pytorch.utils.log import LOGGER


class QuantileQuant(BaseQuantizeModule):
    """
    Function: calibration operator to obtain quant factors using quantile algo
    Features:
        1. Weight max value scaled to 16
        2. Activation max value = 0.99 * current_max + 0.01 * previous_max
        3. Support HIF8 data type
        4. Support weight per-tensor/per-channel, activation per-tensor/static per-token/dynamic per-token
    APIs: forward.
    """
    def __init__(self, ori_module, layer_name, quant_config):
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
        self.layer_name = layer_name
        self.cur_batch = 0
        self.weight_compress_only = True
        
        self.batch_num = quant_config.get('batch_num')
        
        if quant_config.get('inputs_cfg').get('enable_quant') is None or \
            quant_config.get('inputs_cfg').get('enable_quant') == True:
            self.weight_compress_only = False
            self.act_type = quant_config.get('inputs_cfg').get('quant_type')
            self.act_symmetric = quant_config.get('inputs_cfg').get('symmetric')
            self.act_granularity = quant_config.get('inputs_cfg').get('strategy')
            self.dynamic = quant_config.get('inputs_cfg').get('dynamic')
            self.previous_max = None
            self.batch_input = None
        
        self.wts_type = quant_config.get('weights_cfg').get('quant_type')
        
        if self.wts_type != HIFLOAT8:
            raise ValueError("Quantile algorithm only supports hifloat8 type for weights")
        
        if not self.weight_compress_only and self.act_type != HIFLOAT8:
            raise ValueError("Quantile algorithm only supports hifloat8 type for activation")
        
        self.calculate_weights_scale_factor_and_quantize(self.weight.data, quant_config)
        self.bias = ori_module.bias
    
    @torch.no_grad()
    def forward(self, inputs):
        check_linear_input_dim(inputs)
        inputs = inputs.to(self.weight.device)
        fp_out = F.linear(inputs, self.weight, self.bias)
        if self.weight_compress_only or self.dynamic is True:
            return fp_out
        self.cur_batch += 1
        if self.cur_batch > self.batch_num:
            return fp_out
        batch_max = self._compute_batch_max(inputs)
        self._update_act_scale_tensor(batch_max)
        if self.cur_batch == self.batch_num:
            self._finalize_calibration()
        return fp_out

    def calculate_weights_scale_factor_and_quantize(self, weight_data, quant_config):
        """
        Function: calculate weights's quant factor using quantile algo
        Parameters: quant_config: configuration of quantization
        """
        weight_min, weight_max = get_weight_min_max_by_granularity(weight_data, quant_config)
        self.scale_w = self.calculate_hif8_scale(weight_max)
        self.offset_w = None
        LOGGER.logd("Calculate quantile quant params of layer '{}' success!".format(self.layer_name), 'QuantileQuant')
    
    def calculate_hif8_scale(self, tensor_max):
        """
        Function: calculate HIF8 scale by scaling tensor max to 16
        Args:
            tensor_max: max value of tensor (weights or activations)
        Returns:
            scale: HIF8 scale factor
        """
        if tensor_max is not None:
            return (tensor_max / 16.0).to(torch.float32)
        else:
            return tensor_max

    def _compute_batch_max(self, inputs):
        if self.act_granularity == 'tensor':
            return torch.max(torch.abs(inputs)).reshape(-1)
        elif self.act_granularity == 'token':
            cout_axis = -2
            axis_list = list(range(0, inputs.dim()))
            axis_list.pop(cout_axis)
            axis = tuple(axis_list)
            return torch.amax(torch.abs(inputs), dim=axis, keepdim=True)
        return None

    def _update_act_scale_tensor(self, batch_max):
        if self.previous_max is None:
            self.previous_max = batch_max
        else:
            self.previous_max = calculate_quantile_ema_scale(self.previous_max, batch_max)

    def _finalize_calibration(self):
        self.scale_d = self.calculate_hif8_scale(self.previous_max)
        self.offset_d = None
        LOGGER.logd("Calculate quantile activation quant params of layer '{}' success!".format(self.layer_name),
            'QuantileQuant')
