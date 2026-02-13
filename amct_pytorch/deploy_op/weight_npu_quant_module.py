# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
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

from amct_pytorch.utils.quant_util import quant_weight, check_scale_offset_shape
from amct_pytorch.utils.quant_util import apply_awq_quantize_weight
from amct_pytorch.utils.vars import INT8, INT4, HIFLOAT8, FLOAT8_E4M3FN, FLOAT4_E2M1, MXFP4_E2M1
from amct_pytorch.utils.quant_util import quant_tensor
from amct_pytorch.utils.check_params import check_parameters_in_schema


class NpuWeightQuantizedLinear(nn.Module):
    """
    Function: Customized torch.nn.Module of the npu weight quantized linear operator.
    APIs: forward
    """
    def __init__(self, quant_module):
        super().__init__()
        import torch_npu
        self.wts_type = quant_module.wts_type
        self.weight_dtype = None
        if self.wts_type not in [HIFLOAT8, FLOAT8_E4M3FN, INT8, INT4, FLOAT4_E2M1, MXFP4_E2M1]:
            raise RuntimeError('Only support wts_dtype hifloat8, float8_e4m3fn, int8, int4, mxfp4_e2m1 '
                            'for weight only compress npu')

        device = quant_module.weight.device
        self.group_size = quant_module.group_size
        weight = quant_module.weight.data
        self.ori_weight_shape = weight.shape
        if self.wts_type == HIFLOAT8:
            self.weight_dtype = torch_npu.hifloat8

        if hasattr(quant_module, 'scale') and quant_module.scale is not None:
            scale = quant_module.scale.to(device=device)
            weight = apply_awq_quantize_weight(weight, scale, self.group_size)
            self.register_buffer('scale_factor', scale)
        else:
            self.scale_factor = None

        # npu op support scale & offset's shape is (K,N)
        weight_tensor, scale_w, offset_w = self.get_quantize_weight(weight, quant_module, device)
        self.register_buffer('quantized_weight', weight_tensor, device)
        self.register_buffer('scale_w', scale_w)
        if offset_w is not None:
            self.register_buffer('offset_w', (offset_w * -1).to(
                self.scale_w.dtype).transpose(-1, -2).contiguous().to(device=device))
        else:
            self.offset_w = None

        if quant_module.bias is not None:
            self.register_buffer('bias', quant_module.bias.to(device=device))
            if quant_module.weight.dtype == torch.bfloat16 and self.wts_type in [INT8, INT4]:
                self.bias = self.bias.to(torch.float32).to(device)
        else:
            self.bias = None

        self.is_new_torch_npu = False
        if check_parameters_in_schema(torch_npu.npu_weight_quant_batchmatmul, 'weight_dtype'):
            self.is_new_torch_npu = True

    def get_quantize_weight(self, weight, quant_module, device):
        """
        Function: get quantize weight & quanize factor
        Args:
            weight: torch.tensor, weight of quantized module
            quant_module: quantize model
            device: original weight shape
        Returns:
            torch.tensor
        """
        import torch_npu
        offset_w = quant_module.offset_w
        if self.wts_type == MXFP4_E2M1:
            weight_tensor, shared_exponent_w = quant_tensor(weight, self.wts_type)
            weight_tensor = torch_npu.npu_convert_weight_to_int4pack(weight_tensor.npu()).to(device=device)
            weight_tensor = weight_tensor.npu().transpose(1, 0)
            shared_exponent_w = (shared_exponent_w + 127).to(torch.uint8)
            scale_w = shared_exponent_w.transpose(-1, -2)
        else:
            scale_w = quant_module.scale_w.to(device)
            offset_w = offset_w.to(device) if offset_w is not None else None
            check_scale_offset_shape(weight, scale_w, offset_w, self.group_size)
            weight_tensor = quant_weight(weight, self.wts_type, scale_w, offset_w, self.group_size)
            weight_tensor, scale_w, offset_w = self.process_params(
                weight_tensor, scale_w, offset_w, self.ori_weight_shape)
            weight_tensor = weight_tensor.transpose(-1, -2).contiguous().to(device=device)
            if self.wts_type == FLOAT4_E2M1:
                weight_tensor = torch_npu.npu_format_cast(weight_tensor, 29, weight.dtype)
            if self.wts_type in (INT4, FLOAT4_E2M1):
                weight_tensor = torch_npu.npu_convert_weight_to_int4pack(
                    weight_tensor.contiguous().npu()).to(device=device)
            scale_w = scale_w.to(weight.dtype).transpose(-1, -2).contiguous().to(device=device)
        return weight_tensor, scale_w, offset_w

    def process_params(self, quantized_weight, scale_w, offset_w, ori_weight_shape):
        """
        Function: reshape quantize factor to satisfy the requirement of NPU op.
        Args:
            quantized_weight: torch.tensor, quantized weight
            scale_w: torch.tensor, scale factor for weight
            offset_w: torch.tensor, offset factor for weight
            ori_weight_shape: original weight shape
        Returns:
            torch.tensor
        """
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
        if self.scale_factor is not None:
            inputs = torch.mul(inputs, self.scale_factor)

        if self.is_new_torch_npu:
            output = torch_npu.npu_weight_quant_batchmatmul(inputs, self.quantized_weight, self.scale_w,
                    antiquant_offset=self.offset_w, bias=self.bias, antiquant_group_size=self.group_size,
                    weight_dtype=self.weight_dtype)
        else:
            output = torch_npu.npu_weight_quant_batchmatmul(inputs, self.quantized_weight, self.scale_w,
                    antiquant_offset=self.offset_w, bias=self.bias, antiquant_group_size=self.group_size,)

        output = output.reshape(*ori_shape[:-1], -1)

        return output
    
