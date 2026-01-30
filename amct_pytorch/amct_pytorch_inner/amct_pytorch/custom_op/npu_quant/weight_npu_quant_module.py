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
 
fake quant module and Function in weight compress only
 
"""
from torch import nn
import torch

from amct_pytorch.utils.vars import HIFLOAT8, FLOAT8_E4M3FN, INT8, INT4
from amct_pytorch.custom_op.utils import get_quant_factor, get_algo_params, \
    apply_quantize_by_algo, get_optimized_weight, check_scale_offset_shape
from amct_pytorch.utils.weight_quant_utils import quant_weight


class NpuWeightQuantizedLinear(nn.Module):
    """
    Function: Customized torch.nn.Module of the npu weight quantized linear operator.
    APIs: forward
    """
    def __init__(self, ori_module, quant_params, layer_name):
        super().__init__()
        if quant_params.get('wts_type') not in [HIFLOAT8, FLOAT8_E4M3FN, INT8, INT4]:
            raise RuntimeError('Only support wts_dtype HIFLOAT8, FLOAT8_E4M3FN, INT8, INT4 '
                               'for weight only compress npu')
        wts_type = quant_params.get('wts_type')
        self.weight_dtype = None
        if wts_type == 'HIFLOAT8':
            import torch_npu
            self.weight_dtype = torch_npu.hifloat8

        scale_w = None
        offset_w = None
        awq_scale = None
        device = ori_module.weight.device
        self.group_size = quant_params.get('group_size', None)
        round_mode = quant_params.get('round_mode')
        weight = ori_module.weight.data
        ori_weight_shape = weight.shape

        # Replace the original weights with the optimized weights obtained from GPTQ.
        optimized_weight = get_optimized_weight(quant_params, layer_name)
        if isinstance(optimized_weight, torch.Tensor):
            weight = optimized_weight.to(device)

        algo_params = get_algo_params(quant_params, layer_name, 'awq_quantize')
        if algo_params:
            weight = apply_quantize_by_algo(weight, self.group_size, 'awq_quantize', algo_params)
        if algo_params and algo_params.get('scale') is not None:
            awq_scale = algo_params.get('scale').to(device=device)
            self.register_buffer('awq_scale', awq_scale)
        else:
            self.awq_scale = None

        quant_factors = get_quant_factor(quant_params, layer_name)
        if quant_factors:
            check_scale_offset_shape(weight, quant_factors.get('scale_w'), quant_factors.get('offset_w'), \
                self.group_size)
            scale_w = quant_factors.get('scale_w').to(device)
            if quant_factors.get('offset_w') is not None:
                offset_w = quant_factors.get('offset_w').to(device)
        quantized_weight = quant_weight(weight, wts_type, scale_w, offset_w, self.group_size)
        quantized_weight, scale_w, offset_w = self.process_params(quantized_weight, scale_w, offset_w, ori_weight_shape)

        self.register_buffer('quantized_weight', quantized_weight.transpose(-1, -2).contiguous().to(device=device))
        if wts_type == INT4:
            import torch_npu
            self.quantized_weight = \
                torch_npu.npu_convert_weight_to_int4pack(self.quantized_weight.contiguous().npu()).to(device=device)

        self.register_buffer('scale_w',
                             scale_w.to(ori_module.weight.dtype).transpose(-1, -2).contiguous().to(device=device))
        
        # for HIFLOAT8 and FLOAT8_E4M3FN, npu op do not support passing offset
        if offset_w is not None and wts_type not in (HIFLOAT8, FLOAT8_E4M3FN):
            self.register_buffer('offset_w',
                                 offset_w.to(self.scale_w.dtype).transpose(-1, -2).contiguous().to(device=device))
        else:
            self.offset_w = None
        if ori_module.bias is not None:
            self.register_buffer('bias', ori_module.bias.to(device=device))
        else:
            self.bias = None

    def process_params(self, quantized_weight, scale_w, offset_w, ori_weight_shape):
        if self.group_size is not None:
            scale_w = scale_w.reshape(scale_w.shape[0], scale_w.shape[1])
            if offset_w is not None:
                offset_w = offset_w.reshape(offset_w.shape[0], offset_w.shape[1])
            quantized_weight = quantized_weight.reshape(ori_weight_shape)
        else:
            # scale repeat to cout, because npu op only support perchannel quant
            scale_w = scale_w.reshape(-1, 1) if len(scale_w.shape) == 1 else scale_w
            scale_w = scale_w.repeat(ori_weight_shape[0], 1) if scale_w.shape[0] == 1 else scale_w
            if offset_w is not None:
                offset_w = offset_w.reshape(-1, 1) if len(offset_w.shape) == 1 else offset_w
                offset_w = offset_w.repeat(ori_weight_shape[0], 1) if offset_w.shape[0] == 1 else offset_w
            self.group_size = 0
 
        return quantized_weight, scale_w, offset_w

    def forward(self, inputs):
        """
        do inference process
        Params:
        inputs: input data in torch.tensor.
        """
        import torch_npu
        ori_shape = inputs.shape
        inputs = inputs.to(self.quantized_weight.device)
        # input shape reshape to 2d for npu op
        inputs = inputs.reshape(-1, inputs.shape[-1])        
        if self.awq_scale is not None:
            inputs = torch.mul(inputs, self.awq_scale)

        output = torch_npu.npu_weight_quant_batchmatmul(inputs, self.quantized_weight, self.scale_w,
                antiquant_offset=self.offset_w, bias=self.bias, weight_dtype=self.weight_dtype,
                antiquant_group_size=self.group_size)

        output = output.reshape(*ori_shape[:-1], -1)

        return output