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
from torch import nn
import torch

from ....amct_pytorch.utils.data_utils import convert_precision
from ....amct_pytorch.custom_op.utils import get_quant_factor, apply_quantize_by_algo, get_algo_params, \
        apply_progressive_quant, save_algo_params, cal_deq_scale
from ....amct_pytorch.utils.vars import INT32_MAX, INT32_MIN, INT8, INT4


class FakeQuantLinear(nn.Module):
    """
    Function: Customized torch.nn.Module of the fake quantized linear operator.
    APIs: forward
    """
    def __init__(self, ori_module, quant_params, layer_name):
        super().__init__()
        self.ori_dtype = ori_module.weight.dtype
        weight = ori_module.weight
        self.device = ori_module.weight.device
        # get base info
        self.act_type = quant_params.get('act_type')
        wts_type = quant_params.get('wts_type')
        self.round_mode = quant_params.get('round_mode')
        self.group_size = quant_params.get('group_size', 0)
        self.act_granularity = quant_params.get('act_granularity', 'PER_TENSOR')
        # parse pt file
        algo_param_dict = {}
        quant_factors = get_quant_factor(quant_params, layer_name)
        algo_list = ['smooth_quantize']
        for algo in algo_list:
            algo_params = get_algo_params(quant_params, layer_name, algo)
            if algo_params is None:
                continue
            save_algo_params(algo, algo_params, algo_param_dict)
            weight = apply_quantize_by_algo(weight, self.group_size, algo, algo_params)
        # quant twice
        if quant_factors.get('scale_w1') is not None and quant_factors.get('scale_w2') is not None:
            quantized_weight, deq_scale = \
                apply_progressive_quant(weight, quant_factors, self.round_mode, self.group_size)
        # base quant
        else:
            scale_w, deq_scale = cal_deq_scale(quant_factors.get('scale_w'), quant_factors.get('scale_d'), 
                                                        type(ori_module).__name__)
            quantized_weight = convert_precision(weight / scale_w.to(device=weight.device), wts_type, self.round_mode)
        # register params
        self.register_buffer('scale_d', quant_factors.get('scale_d').to(device=self.device, dtype=self.ori_dtype))
        self.register_buffer('offset_d',
            quant_factors.get('offset_d', torch.Tensor([0])).to(device=self.device, dtype=self.ori_dtype))
        self.register_buffer('deq_scale', torch.Tensor(deq_scale).to(device=self.device))
        self.register_buffer('quantized_weight', quantized_weight.to(device=self.device))
        if algo_param_dict.get('smooth_factor') is not None:
            self.scale_d = self.scale_d * algo_param_dict.get('smooth_factor').to(device=self.device)
        if ori_module.bias is not None:
            if self.act_granularity == 'PER_TENSOR' and self.act_type in (INT8, INT4):
                bias_precision = 'INT32'
                bias = ori_module.bias / (deq_scale.to(self.device))
                self.register_buffer('bias', convert_precision(bias, bias_precision, self.round_mode))
            elif self.act_granularity == 'PER_TOKEN':
                self.register_buffer('bias', ori_module.bias.to(self.device))
            else:
                self.register_buffer(
                    'bias', ori_module.bias / (self.scale_d * scale_w.flatten().to(self.device)))
        else:
            self.bias = None

    def infer_func(self, inputs):
        """
        do inference process
        Params:
        inputs: input data in torch.tensor.
        """
        input_dim = len(inputs.shape)
        if input_dim < 2 or input_dim > 6:
            raise RuntimeError("Quantization only support dim from 2 to 6 for linear.")
        if self.bias is not None:
            bias = self.bias.to(torch.float32)
        else:
            bias = None
        output = nn.functional.linear(inputs.to(torch.float32),
            self.quantized_weight.to(torch.float32), bias)
        return output

    def forward(self, inputs):
        """
        Function: fake quantize process
        Inputs:
            inputs: input data in torch.tensor.
        """

        out = inputs.to(device=self.device)
        out = out / self.scale_d
        out += self.offset_d
        out = convert_precision(out, self.act_type, self.round_mode)
        out -= self.offset_d
        out = self.infer_func(out)
        if self.act_type in (INT8, INT4):
            out = torch.clamp(out, INT32_MIN, INT32_MAX)
        out = out * self.deq_scale
        out = out.to(self.ori_dtype)
        return out


class FakeQuantConv2d(nn.Module):
    """
    Function: Customized torch.nn.Module of the fake quantized conv2d operator.
    APIs: forward
    """
    def __init__(self, ori_module, quant_params, layer_name):
        super().__init__()
        self.infer_attrs = dict()
        self.infer_attrs['stride'] = ori_module.stride
        self.infer_attrs['padding'] = ori_module.padding
        self.infer_attrs['dilation'] = ori_module.dilation
        self.infer_attrs['groups'] = ori_module.groups
        self.ori_dtype = ori_module.weight.dtype
        weight = ori_module.weight
        self.device = ori_module.weight.device
        # get base info
        self.act_type = quant_params.get('act_type')
        wts_type = quant_params.get('wts_type')
        self.round_mode = quant_params.get('round_mode')
        # parse pt file
        algo_param_dict = {}
        quant_factors = get_quant_factor(quant_params, layer_name)
        # base quant
        scale_w, deq_scale = cal_deq_scale(quant_factors.get('scale_w'), quant_factors.get('scale_d'), 
                                                    type(ori_module).__name__)
        quantized_weight = convert_precision(weight / scale_w.to(device=weight.device), wts_type, self.round_mode)
        # register params
        self.register_buffer('scale_d', quant_factors.get('scale_d').to(device=self.device, dtype=self.ori_dtype))
        self.register_buffer('deq_scale', torch.Tensor(deq_scale).to(device=self.device))
        self.register_buffer('quantized_weight', quantized_weight.to(device=self.device))
        if ori_module.bias is not None:
            if self.act_type in (INT8, INT4):
                bias_precision = 'INT32'
                bias = ori_module.bias / (self.scale_d * scale_w.flatten().to(self.device))
                self.register_buffer('bias', convert_precision(
                    bias, bias_precision, self.round_mode))
            else:
                self.register_buffer(
                    'bias', ori_module.bias / (self.scale_d * scale_w.flatten().to(self.device)))
        else:
            self.bias = None

    def infer_func(self, inputs):
        """
        do inference process
        Params:
        inputs: input data in torch.tensor.
        """
        if self.bias is not None:
            bias = self.bias.to(torch.float32)
        else:
            bias = None
        output = nn.functional.conv2d(inputs.to(torch.float32), self.quantized_weight.to(torch.float32),
            bias, stride=self.infer_attrs.get('stride'), padding=self.infer_attrs.get('padding'),
            dilation=self.infer_attrs.get('dilation'), groups=self.infer_attrs.get('groups'))
        return output

    def forward(self, inputs):
        """
        Function: fake quantize process
        Inputs:
            inputs: input data in torch.tensor.
        """

        out = inputs.to(device=self.device)
        out = out / self.scale_d
        out = convert_precision(out, self.act_type, self.round_mode)
        out = self.infer_func(out)
        if self.act_type in (INT8, INT4):
            out = torch.clamp(out, INT32_MIN, INT32_MAX)
        out = out * self.deq_scale
        out = out.to(self.ori_dtype)
        return out
