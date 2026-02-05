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
import os
import math
from torch import nn
import torch
import numpy as np

from ....amct_pytorch.utils.weight_quant_utils import quant_dequant_weight
from ....amct_pytorch.custom_op.utils import get_quant_factor, get_optimized_weight, check_scale_offset_shape
from ....amct_pytorch.custom_op.utils import apply_quantize_by_algo, get_algo_params, save_algo_params
from ....amct_pytorch.utils.weight_quant_api import apply_lut_quantize_weight
from ....amct_pytorch.utils.vars import LUT_DEFAULT_GROUP_SIZE, NUM_BITS_MAP
from ....amct_pytorch.utils.data_utils import cal_shared_exponent


class FakeWeightQuantizedConv2d(nn.Module):
    """
    Function: Customized torch.nn.Module of the weight fake quantized conv2d operator.
    APIs: forward
    """
    def __init__(self, ori_module, quant_params, module_name):
        super().__init__()
        quant_factors = get_quant_factor(quant_params, module_name)
        device = ori_module.weight.device
        scale_w = quant_factors.get('scale_w').to(device)
        if len(scale_w) > 1:
            scale_w = scale_w.reshape(-1, 1, 1, 1)
        quantized_weight = quant_dequant_weight(ori_module.weight.data, quant_params, scale_w)
        self.register_buffer('quantized_weight', quantized_weight.to(device=device))
        self.bias = ori_module.bias
        self.infer_attrs = dict()
        self.infer_attrs['stride'] = ori_module.stride
        self.infer_attrs['padding'] = ori_module.padding
        self.infer_attrs['dilation'] = ori_module.dilation
        self.infer_attrs['groups'] = ori_module.groups

    def forward(self, inputs):
        """
        do inference process
        Params:
        inputs: input data in torch.tensor.
        """
        output = nn.functional.conv2d(inputs, self.quantized_weight, self.bias,
            stride=self.infer_attrs.get('stride'), padding=self.infer_attrs.get('padding'),
            dilation=self.infer_attrs.get('dilation'), groups=self.infer_attrs.get('groups'))
        return output


class FakeWeightQuantizedLinear(nn.Module):
    """
    Function: Customized torch.nn.Module of the weight fake quantized linear operator.
    APIs: forward
    """
    def __init__(self, ori_module, quant_params, module_name):
        super().__init__()
        weight = ori_module.weight.data
        device = ori_module.weight.device
        group_size = quant_params.get('group_size', None)
        scale_w = None
        offset_w = None
        awq_scale = None

        if 'quant_result_path' in quant_params:
            algo_params = get_algo_params(quant_params, module_name, 'awq_quantize')
            if algo_params:
                weight = apply_quantize_by_algo(weight, group_size, 'awq_quantize', algo_params)
            if algo_params and algo_params.get('scale') is not None:
                awq_scale = algo_params.get('scale').to(device=device)

            if quant_params.get('wts_type') in ("MXFP4_E2M1",):
                scale_w = cal_shared_exponent(weight, quant_params.get('wts_type'))

            optimized_weight = get_optimized_weight(quant_params, module_name)
            if isinstance(optimized_weight, torch.Tensor):
                weight = optimized_weight.to(device)

            quant_factors = get_quant_factor(quant_params, module_name)
            if quant_factors:
                check_scale_offset_shape(weight, quant_factors.get('scale_w'),
                    quant_factors.get('offset_w'), group_size)
                scale_w = quant_factors.get('scale_w').to(device)
                if quant_factors.get('offset_w') is not None:
                    offset_w = quant_factors.get('offset_w').to(device)

        scale_w = scale_w if isinstance(scale_w, torch.Tensor) else \
            cal_shared_exponent(weight, quant_params.get('wts_type'))
        quantized_weight = quant_dequant_weight(weight, quant_params, scale_w, offset_w)

        self.register_buffer('awq_scale', awq_scale)
        self.register_buffer('quantized_weight', quantized_weight.to(device=device))
        if ori_module.bias is not None:
            self.register_buffer('bias', ori_module.bias.to(device=device))
        else:
            self.bias = None

    def forward(self, inputs):
        """
        do inference process
        Params:
        inputs: input data in torch.tensor.
        """
        inputs = inputs.to(self.quantized_weight.device)
        if self.awq_scale is not None:
            inputs = torch.mul(inputs, self.awq_scale)
        output = nn.functional.linear(inputs, self.quantized_weight, self.bias)
        return output


class LutFakeWeightQuantizedLinear(nn.Module):
    """
    Function: Customized Module of fake quantized linear operator using lut to quant weight.
    APIs: forward
    """
    def __init__(self, ori_module, quant_params, layer_name):
        super().__init__()
        group_size = quant_params.get('group_size', LUT_DEFAULT_GROUP_SIZE)
        if ori_module.weight.dtype != torch.float:
            raise RuntimeError("Not supported weights data type {}".format(
                ori_module.weight.dtype))

        algo_params = get_algo_params(quant_params, layer_name, 'lut_quantize')
        if algo_params is None:
            raise RuntimeError('lut_quantize not exists in quant result file')
        if not isinstance(algo_params, dict):
            raise RuntimeError('Lut params in {} should be dict'.format(quant_params.get('quant_result_path')))
        lut_tables = algo_params.get('lut_table')
        if lut_tables is None:
            raise RuntimeError('Lut table not exists in quant result file')
        if lut_tables.dtype != ori_module.weight.dtype:
            raise RuntimeError("Lut table dtype {} and weight dtype {} is not same".format(
                lut_tables.dtype, ori_module.weight.dtype))
        # lut table shape should be: [cout * ceil(cin / group_size), 2 ** num_bits]
        num_bits = NUM_BITS_MAP.get(quant_params.get('wts_type'))
        if lut_tables.shape[1] != 2 ** num_bits:
            raise RuntimeError("lut table's element number {} is invalid for {}".format(
                lut_tables.shape[1], layer_name))

        weight_group_num = ori_module.weight.shape[0] * np.ceil(ori_module.weight.shape[1] / group_size)
        lut_group_num = lut_tables.shape[0]
        if weight_group_num != lut_group_num:
            raise RuntimeError("group number in Lut {} for {} is invalid".format(
                lut_group_num, layer_name))

        quantized_weight = apply_lut_quantize_weight(ori_module.weight, lut_tables, group_size)
        self.register_buffer('quantized_weight',
            quantized_weight.to(device=ori_module.weight.device))
        if ori_module.bias is not None:
            self.register_buffer('bias', ori_module.bias)
        else:
            self.bias = None

    def forward(self, inputs):
        """
        do inference process
        Params:
        inputs: input data in torch.tensor.
        return: output, a torch.tensor infered by fake quant module
        """
        inputs = inputs.to(self.quantized_weight.device)
        output = nn.functional.linear(inputs, self.quantized_weight, self.bias)
        return output
