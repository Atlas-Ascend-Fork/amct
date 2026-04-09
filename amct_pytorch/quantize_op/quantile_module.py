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
        self.bias = ori_module.bias
        self.layer_name = layer_name
        self.cur_batch = 0
        self.weight_compress_only = True
        self.ori_module = ori_module
        
        self.batch_num = quant_config.get('batch_num')
        
        if quant_config.get('inputs_cfg').get('enable_quant') is None or \
            quant_config.get('inputs_cfg').get('enable_quant') == True:
            self.weight_compress_only = False
            self.act_type = quant_config.get('inputs_cfg').get('quant_type')
            self.act_symmetric = quant_config.get('inputs_cfg').get('symmetric')
            self.act_granularity = quant_config.get('inputs_cfg').get('strategy')
            self.dynamic = quant_config.get('inputs_cfg').get('dynamic', None)
            
            if self.dynamic is not None and self.act_granularity != 'token':
                raise ValueError("The 'dynamic' field can only be enabled when strategy is 'token'")
            
            self.data_max = None
            self.data_min = None
            self.previous_max = None
            self.batch_input = None
            self.dynamic_per_token_scales = []
        
        self.wts_type = quant_config.get('weights_cfg').get('quant_type')
        
        if self.wts_type != HIFLOAT8:
            raise ValueError("Quantile algorithm only supports hifloat8 type for weights")
        
        if not self.weight_compress_only and self.act_type != HIFLOAT8:
            raise ValueError("Quantile algorithm only supports hifloat8 type for activation")
        
        self.calculate_weights_scale_factor_and_quantize(self.weight.data, quant_config)
    
    def calculate_weights_scale_factor_and_quantize(self, weight_data, quant_config):
        """
        Function: calculate weights's quant factor using quantile algo
        Parameters: quant_config: configuration of quantization
        """
        weight_min, weight_max = get_weight_min_max_by_granularity(weight_data, quant_config)
        
        if self.wts_type == HIFLOAT8:
            self.scale_w = self.calculate_hif8_scale(weight_max)
            self.offset_w = None
        LOGGER.logd("Calculate quantile quant params of layer '{}' success!".format(self.layer_name), 'QuantileQuant')
    
    def calculate_hif8_scale(self, weight_max):
        """
        Function: calculate HIF8 scale by scaling weight max to 16
        Args:
            weight_max: max value of weights
        Returns:
            scale: HIF8 scale factor
        """
        scale = (weight_max / 16.0).to(torch.float32)
        return scale
    
    def _compute_batch_min_max(self, inputs):
        if self.act_granularity == 'tensor':
            return torch.max(inputs).reshape(-1), torch.min(inputs).reshape(-1)
        elif self.act_granularity == 'token':
            cout_axis = -2
            axis_list = list(range(0, inputs.dim()))
            axis_list.pop(cout_axis)
            axis = tuple(axis_list)
            batch_max = torch.amax(inputs, dim=axis, keepdim=True)
            batch_min = torch.amin(inputs, dim=axis, keepdim=True)
            self._handle_token_batch_input(inputs, batch_max)
            return batch_max, batch_min
        return None, None

    def _handle_token_batch_input(self, inputs, batch_max):
        if self.dynamic is False:
            if self.batch_input is None:
                self.batch_input = inputs.cpu()
            else:
                self.batch_input = torch.cat((self.batch_input, inputs.cpu()), dim=-2)
        elif self.dynamic is True:
            self.dynamic_per_token_scales.append(batch_max.cpu())

    def _update_act_scale_tensor(self, batch_max, batch_min):
        self.data_max = batch_max if self.data_max is None else self.data_max
        self.data_min = batch_min if self.data_min is None else self.data_min
        if self.act_type == HIFLOAT8:
            if self.scale_d is None:
                self.scale_d = self.data_max
            else:
                self.scale_d = calculate_quantile_ema_scale(self.scale_d, self.previous_max)
            self.previous_max = self.data_max

    def _finalize_calibration(self):
        if self.act_type != HIFLOAT8:
            return
        if self.act_granularity == 'tensor':
            self.scale_d = self.calculate_hif8_scale(self.scale_d)
        elif self.act_granularity == 'token':
            self._finalize_token_calibration()
        self.offset_d = None
        LOGGER.logd("Calculate quantile activation quant params of layer '{}' success!".format(self.layer_name),
            'QuantileQuant')

    def _finalize_token_calibration(self):
        if self.dynamic is False:
            if self.batch_input is not None:
                cout_axis = -2
                axis_list = list(range(0, self.batch_input.dim()))
                axis_list.pop(cout_axis)
                axis = tuple(axis_list)
                per_token_max = torch.amax(self.batch_input, dim=axis, keepdim=True)
                self.scale_d = self.calculate_hif8_scale(per_token_max)
        elif self.dynamic is True:
            if len(self.dynamic_per_token_scales) > 0:
                self.dynamic_per_token_scales = torch.stack(self.dynamic_per_token_scales, dim=0)
                self.scale_d = self.calculate_hif8_scale(self.dynamic_per_token_scales)

    @torch.no_grad()
    def forward(self, inputs):
        check_linear_input_dim(inputs)
        fp_out = self.ori_module(inputs)
        if self.weight_compress_only:
            return fp_out
        self.cur_batch += 1
        if self.cur_batch > self.batch_num:
            return fp_out
        batch_max, batch_min = self._compute_batch_min_max(inputs)
        if self.act_granularity == 'tensor':
            self._update_act_scale_tensor(batch_max, batch_min)
        if self.cur_batch == self.batch_num:
            self._finalize_calibration()
        return fp_out
