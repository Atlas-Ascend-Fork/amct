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
from ....amct_pytorch.custom_op.utils import get_quant_factor


class NpuQuantizationConv2d(torch.nn.Module):
    def __init__(self, ori_module, quant_params, layer_name):
        super().__init__()
        device = ori_module.weight.device
        wts_type = quant_params.get('wts_type')
        if device == "npu" and wts_type != 'HIFLOAT8':
            raise RuntimeError("do not support npu weight quant")

        quant_factors = get_quant_factor(quant_params, layer_name)
        self.register_buffer('act_scale', torch.Tensor(
            1 / quant_factors.get('scale_d')).to(device=device))
        self.register_buffer('act_offset', torch.Tensor(
            quant_factors.get('offset_d', [0])).to(device=device))

        self.stride = ori_module.stride
        self.padding = ori_module.padding
        if isinstance(ori_module.padding, str):
            self.padding = self.calc_padding(ori_module.padding, ori_module.dilation, ori_module.kernel_size)
        self.asymmetric_pad = self.is_asymmetric_pad(self.padding)
        self.dilation = ori_module.dilation
        self.groups = ori_module.groups
        self.stride = ori_module.stride

        self.act_type = quant_params.get('act_type')
        self.input_dtype = None
        if self.act_type == 'HIFLOAT8':
            import torch_npu
            self.input_dtype = torch_npu.hifloat8
            self.npu_quantize_act_type = torch_npu.hifloat8
        elif self.act_type == 'FLOAT8_E4M3FN':
            self.npu_quantize_act_type = torch.float8_e4m3fn

        self.weight_dtype = None
        if wts_type == 'HIFLOAT8':
            import torch_npu
            self.weight_dtype = torch_npu.hifloat8
        self.output_dtype = ori_module.weight.dtype

        scale_d_tensor = torch.Tensor(quant_factors.get('scale_d').to(self.output_dtype)).to(device=device)
        scale_w_tensor = torch.Tensor(quant_factors.get('scale_w').to(self.output_dtype)).to(device=device)
        deq_scale_tensor = scale_d_tensor * scale_w_tensor
        if deq_scale_tensor.shape[0] == 1:
            deq_scale_tensor = deq_scale_tensor.expand(ori_module.weight.shape[0])
        weight_tensor = NpuQuantizationConv2d.quant_wts(ori_module.weight.data, scale_w_tensor, wts_type)
        self.register_buffer('deq_scale', deq_scale_tensor)
        self.register_buffer('quantized_weight', weight_tensor)
        if ori_module.bias is not None:
            bias_tensor = ori_module.bias.data / (scale_d_tensor * scale_w_tensor)
            if wts_type not in ['HIFLOAT8', 'FLOAT8_E4M3FN']:
                bias_tensor = bias_tensor.to(torch.int32)
            self.register_buffer('bias', bias_tensor)
        else:
            self.bias = None

    @staticmethod
    def calc_padding(padding, dilation, kernel_size):
        """
        Function: calculate padding value when 'same' and 'valid'
        Args:
            padding: str. 'same' or 'valid'
            dilation: int/tuple. tuple is (H, W)
            kernel_size: int/tuple. tuple is (H, W)
        Returns:
            padding_value:
                2-d tuple: can be used directly.
                4-d tuple: asymmetric pad, need pad manually.
        """
        if padding == 'valid':
            return (0, 0)

        # when padding is 'same', value is dilation*(kernel_size-1)
        if isinstance(dilation, int):
            dilation_h = dilation
            dilation_w = dilation
        else:
            dilation_h = dilation[0]
            dilation_w = dilation[1]

        if isinstance(kernel_size, int):
            padding_h = dilation_h * (kernel_size - 1)
            padding_w = dilation_w * (kernel_size - 1)
        else:
            padding_h = dilation_h * (kernel_size[0] - 1)
            padding_w = dilation_w * (kernel_size[1] - 1)

        padding_top = padding_h // 2
        padding_bottom = padding_h - padding_top
        padding_left = padding_w // 2
        padding_right = padding_w - padding_left
        if padding_top == padding_bottom and padding_left == padding_right:
            padding_value = (padding_top, padding_left)
        else:
            padding_value = (padding_left, padding_right, padding_top, padding_bottom)
        return padding_value

    @staticmethod
    def is_asymmetric_pad(padding):
        if isinstance(padding, tuple) and len(padding) == 4:
            return True
        return False

    @staticmethod
    def quant_wts(weight, scale_w_tensor, wts_type):
        # weight quant
        if scale_w_tensor.shape[0] > 1:
            scale_w_tensor = scale_w_tensor.reshape(-1, 1, 1, 1)

        weight = weight / scale_w_tensor
        quant_weight = convert_dtype(weight, wts_type)
        return quant_weight

    def forward(self, x):
        pads = self.padding
        if self.asymmetric_pad:
            x = torch.nn.functional.pad(x, self.padding)
            pads = 0
        import torch_npu
        quant_x = torch_npu.npu_quantize(x, self.act_scale, None, self.npu_quantize_act_type, div_mode=False)
        if self.deq_scale.dtype == torch.float32:
            deq_scale = torch_npu.npu_trans_quant_param(self.deq_scale)
        output = torch_npu.npu_quant_conv2d(quant_x, self.quantized_weight, deq_scale, self.stride,
            pads, self.dilation, self.groups, offset_x=0, output_dtype=self.output_dtype,
            bias=self.bias, input_dtype=self.input_dtype, weight_dtype=self.weight_dtype)
        return output
