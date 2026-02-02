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
import torch

from ....amct_pytorch.utils.data_utils import convert_dtype
from ....amct_pytorch.custom_op.utils import get_quant_factor, get_algo_params, apply_smooth_weight


class NpuQuantizationLinear(torch.nn.Module):
    def __init__(self, ori_module, quant_params, layer_name):
        super().__init__()
        self.output_dtype = ori_module.weight.dtype

        self.act_type = quant_params.get('act_type')
        self.act_granularity = 'PER_TENSOR'
        self.x1_dtype = None
        self.axis = -1
        self._init_npu_quantize_type()

        wts_type = quant_params.get('wts_type')
        weight = ori_module.weight.data
        self.x2_dtype = None
        if wts_type == 'HIFLOAT8':
            import torch_npu
            self.x2_dtype = torch_npu.hifloat8

        device = ori_module.weight.device
        if device == "npu" and wts_type != 'HIFLOAT8':
            raise RuntimeError("do not support npu weight quant")
        quant_factors = get_quant_factor(quant_params, layer_name)
        smooth_params = get_algo_params(quant_params, layer_name, 'smooth_quantize')
        act_scale = quant_factors.get('scale_d')
        act_offset = quant_factors.get('offset_d', torch.tensor([0]))

        if self.act_type == 'INT8' and quant_factors.get('scale_d').dim() > 1:
            self.act_granularity = 'PER_TOKEN'
            self.axis = -2
        if smooth_params is not None:
            weight = apply_smooth_weight(smooth_params, weight)
            # npu quantize need offset's shape same with scale
            act_offset = act_offset.repeat(act_scale.shape)
            smooth_factor = smooth_params.get('smooth_factor')
            self.register_buffer('smooth_factor', (1 / smooth_factor).to(device=device))
        self.register_buffer('act_scale', (1 / act_scale).reshape(-1).to(device=device))
        self.register_buffer('pertoken_scale', 
            (quant_factors.get('scale_d')).reshape(-1).to(torch.float32).to(device=device))
        self.register_buffer('act_offset', act_offset.reshape(-1).to(device=device))
        self.act_offset = torch.zeros_like(self.act_scale)
        if self.act_type == 'INT8':
            self.act_scale = self.act_scale.to(self.output_dtype)
            self.act_offset = self.act_offset.to(self.output_dtype)
        scale_w_tensor = quant_factors.get('scale_w').reshape(-1).to(device=device)
        self.register_buffer('scale_w', scale_w_tensor.to(device=device))
        deq_scale_tensor = quant_factors.get('scale_d').to(device=device) * scale_w_tensor

        
        weight_tensor = NpuQuantizationLinear.quant_wts(weight, scale_w_tensor, wts_type)
        weight_tensor = torch.transpose(weight_tensor, 1, 0)

        self.register_buffer('deq_scale', deq_scale_tensor)
        self.register_buffer('quantized_weight', weight_tensor)
        self._init_bias(ori_module, act_scale, scale_w_tensor)

    @staticmethod
    def quant_wts(weight, scale_w_tensor, wts_type):
        # weight quant
        if scale_w_tensor.shape[0] > 1:
            scale_w_tensor = scale_w_tensor.reshape(-1, 1)
        weight = weight / scale_w_tensor
        quant_weight = convert_dtype(weight, wts_type)
        return quant_weight

    def forward(self, x):
        import torch_npu
        if hasattr(self, 'smooth_factor'):
            x = x * self.smooth_factor
        quant_x = torch_npu.npu_quantize(x, self.act_scale, self.act_offset, dtype=self.npu_quantize_act_type, 
            axis=self.axis, div_mode=False)
        if self.act_granularity == 'PER_TENSOR':
            output = torch_npu.npu_quant_matmul(quant_x, self.quantized_weight,
                self.deq_scale, bias=self.bias, output_dtype=self.output_dtype,
                x1_dtype=self.x1_dtype, x2_dtype=self.x2_dtype)
        elif self.act_granularity == 'PER_TOKEN':
            output = torch_npu.npu_quant_matmul(quant_x, self.quantized_weight,
                scale=self.scale_w, pertoken_scale=self.pertoken_scale,
                bias=self.bias, output_dtype=self.output_dtype,
                x1_dtype=self.x1_dtype, x2_dtype=self.x2_dtype)
        return output

    def _init_npu_quantize_type(self):
        if self.act_type == 'HIFLOAT8':
            import torch_npu
            self.x1_dtype = torch_npu.hifloat8
            self.npu_quantize_act_type = torch_npu.hifloat8
        elif self.act_type == 'FLOAT8_E4M3FN':
            self.npu_quantize_act_type = torch.float8_e4m3fn
        elif self.act_type == 'INT8':
            self.npu_quantize_act_type = torch.int8
        else:
            raise ValueError('Not supported dtype {}'.format(self.act_type))

    def _init_bias(self, ori_module, act_scale, scale_w_tensor):
        if ori_module.bias is not None:
            if self.act_granularity == 'PER_TENSOR' and self.act_type in ('INT8', 'INT4'):
                bias_tensor = ori_module.bias.round().to(torch.int32)
                self.register_buffer('bias', bias_tensor)
            elif self.act_granularity == 'PER_TOKEN':
                self.register_buffer('bias', ori_module.bias)
            else:
                bias_tensor = ori_module.bias.data / (act_scale * scale_w_tensor)
                self.register_buffer('bias', bias_tensor)
        else:
            self.bias = None
