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

calibration operator to obtain act quant factors using min-max algo

"""
import torch
import math
from torch import nn

from amct_pytorch.custom_op.utils import convert_to_per_group_shape, calculate_scale_offset, \
    calculate_progressive_weights_scale_factor, get_weight_min_max_by_granularity
from amct_pytorch.utils.vars import FLOAT4_E2M1, FLOAT8_E4M3FN
from amct_pytorch.utils.data_utils import check_linear_input_dim


class MinMaxCalibrator(nn.Module):
    """
    Function: calibration operator to obtain act quant factors using min-max algo
    APIs: forward.
    """
    def __init__(self,
                 ori_module,
                 record_module,
                 layer_name,
                 quant_config):
        """
        Function: init objective.
        Args:
        ori_module: torch module. Quantized_type.
        record_module: custom record module. To read and write.
        layer_name: ori_module's name.
        quant_config: calibration algorithm parameters.
        """
        super().__init__()
        self.ori_module = ori_module
        self.post_quant_record_module = record_module
        self.layer_name = layer_name
        self.cur_batch = 0

        self.batch_num = quant_config.get('batch_num')
        self.act_type = quant_config.get('act_type')
        self.wts_type = quant_config.get('wts_type')
        self.act_asymmetric = quant_config.get('act_asymmetric')
        self.weight_compress_only = quant_config.get('weight_compress_only')
        self.wts_asymmetric = quant_config.get('wts_asymmetric')
        self.group_size = quant_config.get('group_size', None)
        self.weight_granularity = quant_config.get('weight_granularity')
        self.act_granularity = quant_config.get('act_granularity', 'PER_TENSOR')

        self.data_max = None
        self.data_min = None
        self.scale_w = None
        self.scale_w1 = None
        self.scale_w2 = None
        self.offset_w = None
        if self.act_type == FLOAT8_E4M3FN and self.wts_type == FLOAT4_E2M1:
            self.scale_w1, self.scale_w2 = \
                calculate_progressive_weights_scale_factor(self.ori_module.weight.data)
        else:
            self.scale_w, self.offset_w = self.calculate_weights_scale_factor_and_quantize(quant_config)

    def calculate_weights_scale_factor_and_quantize(self, quant_config):
        """
        Function: calculate weights's quant factor and do fakequant
        Parameters: quant_config: configuration of quantization
        """
        weight_data = self.ori_module.weight.data

        weight_min, weight_max = get_weight_min_max_by_granularity(weight_data, quant_config)

        scale_w, offset_w = calculate_scale_offset(weight_max, weight_min, 
                                            self.wts_asymmetric, self.wts_type)

        # write record while weight_compress_only is True
        if quant_config.get('weight_compress_only'):
            quant_params = {
                    self.layer_name[0]: {
                        'quant_factors': {
                            'scale_w': scale_w.detach().cpu(),
                            'offset_w': offset_w.detach().cpu(),
                        }
                    }
                }
            if self.weight_granularity == 'PER_GROUP':
                quant_info = {
                    'wts_type': self.wts_type,
                    'round_mode': 'RINT',
                    'group_size': quant_config.get("group_size"),
                    'weight_compress_only': True
                }
            else:
                quant_info = {
                    'wts_type': self.wts_type,
                    'round_mode': 'RINT',
                    'weight_compress_only': True
                }

            self.post_quant_record_module(self.layer_name, quant_info, quant_params)
        return scale_w, offset_w

    def forward(self, inputs):
        """
        Function: MinMaxCalibrator forward function.
        Args:
        inputs: data used for calibration in torch.tensor.
        """
        with torch.no_grad():
            check_linear_input_dim(inputs)
            fp_out = self.ori_module(inputs)

            if self.weight_compress_only:
                return fp_out

            self.cur_batch += 1
            if self.cur_batch > self.batch_num:
                return fp_out
            if self.act_granularity == 'PER_TENSOR':
                batch_max = torch.max(inputs).reshape(-1)
                batch_min = torch.min(inputs).reshape(-1)
            elif self.act_granularity == 'PER_TOKEN':
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
                scale_d, offset_d = calculate_scale_offset(
                    self.data_max, self.data_min, self.act_asymmetric, self.act_type)
                quant_params = {
                    self.layer_name[0]: {
                        'quant_factors': {
                            'scale_d': scale_d.detach().cpu(),
                            'offset_d': offset_d.detach().cpu(),
                        }
                    }
                }
                quant_info = {
                    'act_type': self.act_type,
                    'wts_type': self.wts_type,
                    'round_mode': 'RINT'
                }
                if self.scale_w is not None:
                    quant_params[self.layer_name[0]]['quant_factors']['scale_w'] = self.scale_w.detach().cpu()
                if self.offset_w is not None:
                    quant_params[self.layer_name[0]]['quant_factors']['offset_w'] = self.offset_w.detach().cpu()
                if self.group_size is not None:
                    quant_info['group_size'] = self.group_size
                if self.scale_w1 is not None and self.scale_w2 is not None:
                    quant_params[self.layer_name[0]]['quant_factors']['scale_w1'] = self.scale_w1.detach().cpu()
                    quant_params[self.layer_name[0]]['quant_factors']['scale_w2'] = self.scale_w2.detach().cpu()
                    quant_info['group_size'] = 32 # fp8*fp4 group size only support 32
                self.post_quant_record_module(self.layer_name, quant_info, quant_params)
            return fp_out
