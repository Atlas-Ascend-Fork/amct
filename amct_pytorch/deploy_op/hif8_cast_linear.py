# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------
import torch
from torch import nn
from amct_pytorch.utils.quant_util import quant_weight
from amct_pytorch.utils.vars import HIFLOAT8
from amct_pytorch.utils.log import LOGGER


class HIF8CastLinear(nn.Module):
    """
    Function: Customized torch.nn.Module of the HIF8 cast linear operator.
    This operator implements BF16/FP16 to HIF8 conversion with offline weight quantization
    and online activation clip+cast.
    APIs: forward
    """
    def __init__(self, quant_module):
        """
        Function: init objective.
        Args:
            quant_module: quant_module with HIF8 quantized weights
        """
        super().__init__()
        import torch_npu
        
        self.wts_type = quant_module.wts_type
        if self.wts_type != HIFLOAT8:
            raise RuntimeError('HIF8CastLinear only supports hifloat8 weight type')
        
        device = quant_module.weight.device
        weight = quant_module.weight.data
        self.ori_weight_shape = weight.shape
        
        self.weight_dtype = torch_npu.hifloat8
        
        scale_w = quant_module.scale_w.to(device)
        offset_w = quant_module.offset_w.to(device) if quant_module.offset_w is not None else None
        
        weight_tensor = self._get_quantize_weight(weight, scale_w, offset_w, device)
        self.register_buffer('quantized_weight', weight_tensor)
        self.register_buffer('scale_w', scale_w)
        
        if offset_w is not None:
            self.register_buffer('offset_w', offset_w)
        else:
            self.offset_w = None
        
        if quant_module.bias is not None:
            self.register_buffer('bias', quant_module.bias.to(device=device))
        else:
            self.bias = None
        
        LOGGER.logd(f'HIF8CastLinear initialized with weight shape {self.ori_weight_shape}')
    
    def _get_quantize_weight(self, weight, scale_w, offset_w, device):
        """
        Function: get quantize weight for HIF8
        Args:
            weight: torch.tensor, original weight
            scale_w: torch.tensor, scale factor for weight
            offset_w: torch.tensor, offset factor for weight
            device: device
        Returns:
            torch.tensor: quantized weight
        """
        import torch_npu
        
        weight_tensor = quant_weight(weight, self.wts_type, scale_w, offset_w)
        weight_tensor = weight_tensor.transpose(-1, -2).contiguous().to(device=device)
        
        return weight_tensor
    
    def forward(self, inputs):
        """
        Function: HIF8CastLinear forward function.
        Implements online activation clip+cast and HIF8 weight quantization.
        Args:
            inputs: input data in torch.tensor (BF16/FP16)
        Returns:
            torch.tensor: output in BF16/FP16
        """
        import torch_npu
        
        ori_shape = inputs.shape
        inputs = inputs.to(self.quantized_weight.device)
        
        inputs = inputs.reshape(-1, inputs.shape[-1])
        
        clip_min = torch.tensor(-240.0, device=inputs.device, dtype=inputs.dtype)
        clip_max = torch.tensor(240.0, device=inputs.device, dtype=inputs.dtype)
        
        clipped_inputs = torch.clamp(inputs, clip_min, clip_max)
        
        cast_inputs = torch_npu.npu_dtype_cast(clipped_inputs, dtype=torch_npu.hifloat8)
        
        output = torch_npu.npu_weight_quant_batchmatmul(
            cast_inputs, 
            self.quantized_weight, 
            self.scale_w,
            antiquant_offset=self.offset_w, 
            bias=self.bias, 
            antiquant_group_size=0
        )
        
        output = output.reshape(*ori_shape[:-1], -1)
        
        return output