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
import torch
from amct_pytorch.utils.check_params import check_parameters_in_schema


class NpuQuantizationLinear(torch.nn.Module):
    """
    Function: Customized torch.nn.Module of the npu quantized linear operator.
    APIs: forward
    """
    def __init__(self, quant_module):
        """
        Function: Initialize NPU quantized linear operator
        Args:
            quant_module: quantization module containing quantization parameters
        """
        super().__init__()
        import torch_npu
        device = quant_module.weight.device
        
        self.x1_dtype = torch_npu.hifloat8
        self.npu_quantize_act_type = torch_npu.hifloat8
        self.x2_dtype = torch_npu.hifloat8
        
        self._init_weight_quant(quant_module, device)
        if quant_module.bias is None:
            self.bias = None
        self.register_buffer('bias', quant_module.bias)
        self.is_new_torch_npu = True if check_parameters_in_schema(torch_npu.npu_quant_matmul, 'x1_dtype') else False
        
    def _init_weight_quant(self, quant_module):
        fp8_weight = quant_module.weight
        fp8_scale = quant_module.weight_scale_inv
        dequant_weight = fp8_weight.to(torch.float32) / fp8_scale
        weight_max = dequant_weight.max(dim=1, keepdim=True).values
        scale = (weight_max / 16.0).to(torch.float32)
        self.register_buffer('deq_scale', scale)
        import torch_npu
        hif8_quantized_tensor = torch_npu.npu_quantize(dequant_weight, scale, None, dtype=torch_npu.hifloat8)
        self.register_buffer('quantized_weight', hif8_quantized_tensor)

    def forward(self, x):
        """
        do inference process
        Params:
        inputs: input data in torch.tensor.
        """
        import torch_npu
        
        output_dtype = x.dtype
        
        quant_x, pertoken_scale = torch_npu.npu_dynamic_quant(x, dst_type=self.npu_quantize_act_type,
            dst_type_max=15)
        self.pertoken_scale = pertoken_scale.view(-1)
            
        if self.is_new_torch_npu:
            output = torch_npu.npu_quant_matmul(quant_x, self.quantized_weight,
                scale=self.deq_scale, pertoken_scale=self.pertoken_scale,
                output_dtype=output_dtype,
                x1_dtype=self.x1_dtype, x2_dtype=self.x2_dtype,
                group_sizes=None, y_scale=None)
            if self.bias is not None:
                output = output + self.bias
        else:
            output = torch_npu.npu_quant_matmul(quant_x, self.quantized_weight,
                scale=self.deq_scale, pertoken_scale=self.pertoken_scale,
                bias=self.bias, output_dtype=output_dtype)

        return output