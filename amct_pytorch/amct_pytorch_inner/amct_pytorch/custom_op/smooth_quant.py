#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use
this file except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

SmoothQuant module and Function

"""
import torch
from torch import nn
import torch.nn.functional as F

from amct_pytorch.utils.data_utils import check_data_type, check_linear_input_dim
from amct_pytorch.custom_op.utils import calculate_scale_offset, calculate_progressive_weights_scale_factor
from amct_pytorch.utils.vars import FLOAT8_E4M3FN, INT8, FLOAT4_E2M1
from amct_pytorch.utils.log import LOGGER


class SmoothQuant(nn.Module):
    """
    Function: Customized torch.nn.Module of the SmoothQuant class.
    APIs: forward.
    """
    def __init__(self,
                 ori_module,
                 record_module,
                 layer_name,
                 quant_config,
                 group_size=32):
        """
        Function: init objective.
        Args:
        ori_module: torch module. Linear.
        record_module: customized record module. To read and write.
        layer_name: ori_module's name.
        quant_config: calibration algorithm parameters.
        """
        super().__init__()
        self.weight = ori_module.weight
        self.bias = ori_module.bias
        self.post_quant_record_module = record_module
        self.layer_name = layer_name
        self.quant_config = quant_config
        self.group_size = group_size
        self.batch_num = quant_config.get('batch_num')
        self.act_granularity = quant_config.get('act_granularity', 'PER_TENSOR')
        self.act_asymmetric = quant_config.get('act_asymmetric', False)
        self.act_type = quant_config.get('act_type')
        self.wts_type = quant_config.get('wts_type')
        self.cur_batch = 0
        self.device = ori_module.weight.device

        self.data_max = torch.ones((1, self.weight.shape[-1]), device=self.weight.device, \
            dtype=self.weight.dtype) * torch.inf * -1
        self.data_min = torch.ones((1, self.weight.shape[-1]), device=self.weight.device, \
            dtype=self.weight.dtype) * torch.inf
        self.batch_input = None
        

    @torch.no_grad()
    def forward(self, inputs):
        """
        Function: SmoothQuant forward function.
        Args:
        inputs: data used for calibration in torch.tensor.
        """
        check_data_type(inputs.dtype, (torch.bfloat16, torch.float16))
        check_linear_input_dim(inputs)
        inputs = inputs.to(self.weight.device)
        output = F.linear(inputs, self.weight, self.bias)

        self.cur_batch += 1
        if self.cur_batch > self.batch_num:
            return output
        batch_max = inputs.reshape(-1, inputs.shape[-1]).amax(dim=0, keepdim=True)
        batch_min = inputs.reshape(-1, inputs.shape[-1]).amin(dim=0, keepdim=True)
        if self.act_granularity == 'PER_TOKEN':
            if self.batch_input is None:
                self.batch_input = inputs.cpu()
            else:
                self.batch_input = torch.cat((self.batch_input, inputs.cpu()), dim=-1)

        # update data_max and data_min to calculate smooth factor
        self.data_max = torch.where(batch_max > self.data_max, batch_max, self.data_max)
        self.data_min = torch.where(batch_min < self.data_min, batch_min, self.data_min)

        if self.cur_batch == self.batch_num:
            act_max = torch.max(self.data_max.abs(), self.data_min.abs())
            smooth_factor = self.calculate_smooth(act_max)
            weight = self.weight * smooth_factor
            # only FP8 * FP4 do progressive scale
            if self.act_type == FLOAT8_E4M3FN and self.wts_type == FLOAT4_E2M1:
                self.scale_w1, self.scale_w2 = calculate_progressive_weights_scale_factor(weight.data)
            else:
                self.scale_w = self.calculate_weights_scale_factor(weight, self.quant_config.get('weight_granularity'))

            if self.act_granularity == 'PER_TENSOR':
                scale_d, offset_d = self._calculate_per_tensor_params(smooth_factor)
            elif self.act_granularity == 'PER_TOKEN':
                scale_d, offset_d = self._calculate_per_token_params(smooth_factor)

            self._record_quant_factors(scale_d, offset_d, smooth_factor)

        return output

    def calculate_smooth(self, act_max):
        """
        Function: smooth the input tensor by smooth_strength
        Parameters: inputs: max activation per channel
        Return: output_tensor: smooth factor
        """

        # weight shape is [cout, cin], wts_max shape is [1,cin]
        wts_max = self.weight.abs().amax(dim=0, keepdim=True)

        smooth_strength = self.quant_config.get('act_smooth').get('smooth_strength')
        smooth_factor = (act_max ** smooth_strength) / (wts_max ** (1 - smooth_strength))
        # set smooth_factor to 1 if nan or inf or 0
        zero_mask = smooth_factor < torch.finfo(act_max.dtype).eps
        invalid_mask = ~torch.isfinite(smooth_factor)
        if zero_mask.any().item() or invalid_mask.any().item():
            smooth_factor = torch.where(zero_mask | invalid_mask,
                torch.tensor(1.0, dtype=smooth_factor.dtype, device=smooth_factor.device), smooth_factor)
            LOGGER.logw("The smooth factor is calculated as abnormal and set to 1")

        return smooth_factor

    def calculate_weights_scale_factor(self, weight, weight_granularity):
        """
        Function: calculate weights's quant factor
        Parameters: 
        weight: weight data
        weight_granularity: weight granularity
        """
        weight_data = weight.data

        if weight_granularity == 'PER_CHANNEL':
            weight_max = weight_data.max(dim=1, keepdim=True).values
            weight_min = weight_data.min(dim=1, keepdim=True).values
        elif weight_granularity == 'PER_TENSOR':
            weight_max = weight_data.max().reshape(1, 1)
            weight_min = weight_data.min().reshape(1, 1)
        scale_w, _ = calculate_scale_offset(weight_max, weight_min, 
                                            False, self.wts_type)

        return scale_w

    def _calculate_per_token_params(self, smooth_factor):
        """
        Function: calculate per token scale_d and offset_d
        Parameters: 
        smooth_factor: smooth factor
        """
        smooth_input = self.batch_input / smooth_factor.cpu().repeat(1, self.batch_num)
        cout_axis = -2
        axis_list = list(range(0, smooth_input.dim()))
        axis_list.pop(cout_axis)
        axis = tuple(axis_list)
        per_token_max = torch.amax(smooth_input, dim=axis, keepdim=True)
        per_token_min = torch.amin(smooth_input, dim=axis, keepdim=True)
        return calculate_scale_offset(per_token_max, per_token_min, self.act_asymmetric, self.act_type)

    def _calculate_per_tensor_params(self, smooth_factor):
        """
        Function: calculate per tensor scale_d and offset_d
        Parameters: 
        smooth_factor: smooth factor
        """
        max_val = (self.data_max / smooth_factor).max().reshape(-1)
        min_val = (self.data_min / smooth_factor).min().reshape(-1)
        return calculate_scale_offset(max_val, min_val, self.act_asymmetric, self.act_type)

    def _record_quant_factors(self, scale_d, offset_d, smooth_factor):
        """
        Function: write quant param and quant factor in record
        Parameters: 
        smooth_factor: smooth factor
        scale_d: scale_d
        offset_d: offset_d
        """
        if self.act_type == FLOAT8_E4M3FN:
            quant_factors = {
                'scale_d': scale_d.detach().cpu(),
                'scale_w1': self.scale_w1.detach().cpu(),
                'scale_w2': self.scale_w2.detach().cpu(),
                'offset_d': offset_d.detach().cpu()
            }
        elif self.act_type == INT8:
            quant_factors = {
                'scale_d': scale_d.detach().cpu(),
                'scale_w': self.scale_w.detach().cpu(),
                'offset_d': offset_d.detach().cpu()
            }

        quant_params = {
            self.layer_name[0]: {
                'smooth_quantize': {
                    'smooth_factor': smooth_factor.detach().cpu()
                },
                'quant_factors': quant_factors
            }
        }

        quant_info = {
            'act_type': self.quant_config.get('act_type'),
            'wts_type': self.quant_config.get('wts_type'),
            'group_size': self.group_size,
            'round_mode': 'RINT'
        }

        self.post_quant_record_module(self.layer_name, quant_info, quant_params)