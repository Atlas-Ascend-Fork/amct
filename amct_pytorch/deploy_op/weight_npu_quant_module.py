# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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


class NpuWeightQuantizedLinear(nn.Module):
    """
    Function: Customized torch.nn.Module of the npu weight quantized linear operator.
    APIs: forward
    """
    def __init__(self, quant_module):
        super().__init__()
        wts_type = quant_module.wts_type

        device = quant_module.weight.device
        self.group_size = quant_module.group_size
        weight = quant_module.weight.data
        ori_weight_shape = weight.shape

        if hasattr(quant_module, 'scale') and quant_module.scale is not None:
            scale = quant_module.scale.to(device=device)
            weight = apply_awq_quantize_weight(weight, scale, quant_module.clip_max, self.group_size)
            self.register_buffer('scale_factor', scale)
        else:
            self.scale_factor = None

        scale_w = quant_module.scale_w.to(device)
        offset_w = quant_module.offset_w
        if offset_w is not None:
            offset_w = offset_w.to(device)
        check_scale_offset_shape(weight, scale_w, offset_w, self.group_size)

        quantized_weight = quant_weight(weight, wts_type, scale_w, offset_w, self.group_size)
        quantized_weight, scale_w, offset_w = self.process_params(quantized_weight, scale_w, offset_w, ori_weight_shape)
        self.register_buffer('quantized_weight', quantized_weight.transpose(-1, -2).contiguous().to(device=device))
        if wts_type == 'int4':
            import torch_npu
            self.quantized_weight = \
                torch_npu.npu_convert_weight_to_int4pack(self.quantized_weight.contiguous().npu()).to(device=device)

        # npu op support scale & offset's shape is (K,N)
        self.register_buffer('scale_w', scale_w.to(
            quant_module.weight.dtype).transpose(-1, -2).contiguous().to(device=device))
        if offset_w is not None:
            self.register_buffer('offset_w', (offset_w * -1).to(
                self.scale_w.dtype).transpose(-1, -2).contiguous().to(device=device))
        else:
            self.offset_w = None

        if quant_module.bias is not None:
            self.register_buffer('bias', quant_module.bias.to(device=device))
            if quant_module.weight.dtype == torch.bfloat16:
                self.bias = self.bias.to(torch.float32).to(device)
        else:
            self.bias = None

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

        output = torch_npu.npu_weight_quant_batchmatmul(inputs, self.quantized_weight, self.scale_w,
                antiquant_offset=self.offset_w, bias=self.bias, antiquant_group_size=self.group_size)

        output = output.reshape(*ori_shape[:-1], -1)

        return output
    
