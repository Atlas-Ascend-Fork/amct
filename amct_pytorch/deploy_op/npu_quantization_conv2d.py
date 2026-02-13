# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the Apache License Version 2.0.You may not use this file except in compliance with the License.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# Apache License for more details at
# http://www.apache.org/licenses/LICENSE-2.0
# ----------------------------------------------------------------------------
import torch

from amct_pytorch.utils.vars import HIFLOAT8, FLOAT8_E4M3FN
from amct_pytorch.utils.quant_util import quant_tensor


class NpuQuantizationConv2d(torch.nn.Module):
    """
    Function: Customized torch.nn.Module of the npu quantized convert operator.
    APIs: forward
    """
    def __init__(self, quant_module):
        super().__init__()
        import torch_npu
        device = quant_module.weight.device
        act_type = quant_module.act_type
        wts_type = quant_module.wts_type
    
        self.output_dtype = quant_module.weight.dtype
        scale_d = quant_module.scale_d
        offset_d = quant_module.offset_d if quant_module.offset_d is not None else torch.tensor([0])
        offset_d = offset_d.repeat(scale_d.shape)
        self.register_buffer('act_scale', torch.Tensor((1 / scale_d)).to(device=device).to(self.output_dtype))
        self.register_buffer('act_offset', torch.Tensor(offset_d).to(device=device).to(self.output_dtype))

        self.stride = quant_module.ori_module.stride
        self.padding = quant_module.ori_module.padding
        self.dilation = quant_module.ori_module.dilation
        self.groups = quant_module.ori_module.groups
        if isinstance(self.padding, str):
            self.padding = self.calc_padding(self.padding, self.dilation, quant_module.ori_module.kernel_size)
        self.asymmetric_pad = self.is_asymmetric_pad(self.padding)

        self.input_dtype = None
        self.weight_dtype = None
        self.init_dtype(act_type, wts_type)

        scale_d_tensor = torch.Tensor(quant_module.scale_d.to(self.output_dtype)).to(device=device)
        scale_w_tensor = torch.Tensor(quant_module.scale_w.to(self.output_dtype)).to(device=device)
        deq_scale_tensor = scale_d_tensor * scale_w_tensor
        if deq_scale_tensor.shape[0] == 1:
            deq_scale_tensor = deq_scale_tensor.expand(quant_module.weight.shape[0])
        weight_tensor = NpuQuantizationConv2d.quant_wts(quant_module.weight.data, scale_w_tensor, wts_type)
        self.register_buffer('deq_scale', deq_scale_tensor)
        self.register_buffer('quantized_weight', weight_tensor)
        if quant_module.bias is not None:
            bias_tensor = quant_module.bias.data / (scale_d_tensor * scale_w_tensor)
            if wts_type not in [HIFLOAT8, FLOAT8_E4M3FN]:
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
        res, _ = quant_tensor(weight, wts_type, scale_w_tensor)
        return res

    def init_dtype(self, act_type, wts_type):
        import torch_npu
        if act_type == HIFLOAT8:
            self.input_dtype = torch_npu.hifloat8
            self.npu_quantize_act_type = torch_npu.hifloat8
        elif act_type == FLOAT8_E4M3FN:
            self.npu_quantize_act_type = torch.float8_e4m3fn

        if wts_type == HIFLOAT8:
            self.weight_dtype = torch_npu.hifloat8

    def forward(self, x):
        import torch_npu
        pads = self.padding
        if self.asymmetric_pad:
            x = torch.nn.functional.pad(x, self.padding)
            pads = 0
        quant_x = torch_npu.npu_quantize(x, self.act_scale, None, self.npu_quantize_act_type, div_mode=False)
        deq_scale = torch_npu.npu_trans_quant_param(self.deq_scale.to(torch.float32))
        output = torch_npu.npu_quant_conv2d(quant_x, self.quantized_weight, deq_scale, self.stride,
            pads, self.dilation, self.groups, offset_x=0, output_dtype=self.output_dtype,
            bias=self.bias, input_dtype=self.input_dtype, weight_dtype=self.weight_dtype)
        return output
