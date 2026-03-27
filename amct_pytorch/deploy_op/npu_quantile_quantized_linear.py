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
from torch import nn

from amct_pytorch.utils.quant_util import quant_weight, check quant_scale_offset_shape
from amct_pytorch.utils.vars import HIFLOAT8
from amct_pytorch.utils.check_params import check_parameters_in_schema


class NpuQuantileQuantizedLinear(nn.Module):
    """
    Function: Customized torch.nn.Module of NPU quantile quantized linear operator.
    Features:
        1. Weight quantization factor saved for deployment
        2. Activation quantization factor saved for deployment
        3. Support HIF8 data type
        4. Support weight per-tensor/per-channel
        5. Support activation per-tensor/static per-token/dynamic per-token
    APIs: forward
    """
    def __init__(self, quant_module):
        super().__init__()
        import torch_npu
        
        self.wts_type = quant_module.wts_type
        self.act_type = quant_module.act_type if hasattr(quant_module, 'act_type') else None
        self.weight_dtype = None
        
        if self.wts_type == HIFLOAT8:
            self.weight_dtype = torch_npu.hifloat8
        
        device = quant_module.weight.device
        weight = quant_module.weight.data
        
        self.group_size = quant_module.group_size if hasattr(quant_module, 'group_size') else None
        self.weight_granularity = quant_module.weight_granularity
        
        weight_tensor, scale_w, offset_w = self.get_quantize_weight(weight, quant_module, device)
        self.register_buffer('quantized_weight', weight_tensor.contiguous(), device)
        self.register_buffer('scale_w', scale_w)
        
        if offset_w is not None:
            self.register_buffer('offset_w', (offset_w * -1).to(
                self.scale_w.dtype).transpose(-1, -2).contiguous().to(device=device))
        else:
            self.offset_w = None
        
        if quant_module.bias is not None:
            self.register_buffer('bias', quant_module.bias.to(device=device))
            if quant_module.weight.dtype == torch.bfloat16 and self.wts_type in ['int8', 'int4']:
                self.bias = self.bias.to(torch.float32).to(device)
        else:
            self.bias = None
        
        if not quant_module.weight_compress_only:
            if self.act_type == HIFLOAT8:
                self.scale_d = quant_module.scale_d.to(device=device)
                self.offset_d = None
            else:
                self.scale_d = quant_module.scale_d.to(device=device)
                if quant_module.offset_d is not None:
                    self.offset_d = quant_module.offset_d.to(device=device)
                else:
                    self.offset_d = None
        
        self.is_new_torch_npu = False
        if check_parameters_in_schema(torch_npu.npu_weight_quant_batchmatmul, 'weight_dtype'):
            self.is_new_torch_npu = True
    
    def get_quantize_weight(self, weight, quant_module, device):
        """
        Function: get quantize weight & quantize factor
        Args:
            weight: torch.tensor, weight of quantized module
            quant_module: quantize model
            device: original weight device
        Returns:
            torch.tensor
        """
        import torch_npu
        
        scale_w = quant_module.scale_w.to(device)
        offset_w = quant_module.offset_w.to(device) if quant_module.offset_w is not None else None
        check_scale_offset_shape(weight, scale_w, offset_w, self.group_size)
        weight_tensor = quant_weight(weight, self.wts_type, scale_w, offset_w, self.group_size)
        
        weight_tensor = weight_tensor.transpose(-1, -2).contiguous().to(device=device)
        scale_w = scale_w.to(weight.dtype).transpose(-1, -2).contiguous().to(device=device)
        
        return weight_tensor, scale_w, offset_w
    
    def forward(self, inputs):
        """
        Function: do inference process
        Params:
            inputs: input data in torch.tensor.
        Returns:
            torch.tensor
        """
        import torch_npu
        ori_shape = inputs.shape
        inputs = inputs.to(self.quantized_weight.device)
        inputs = inputs.reshape(-1, inputs.shape[-1])
        
        if self.is_new_torch_npu:
            output = torch_npu.npu_weight_quant_batchmatmul(inputs, self.quantized_weight, self.scale_w,
                    antiquant_offset=self.offset_w, bias=self.bias, antiquant_group_size=self.group_size,
                    weight_dtype=self.weight_dtype)
        else:
            output = torch_npu.npu_weight_quant_batchmatmul(inputs, self.quantized_weight, self.scale_w,
                    antiquant_offset=self.offset_w, bias=self.bias, antiquant_group_size=self.group_size)
        
        output = output.reshape(*ori_shape[:-1], -1)
        
        return output
