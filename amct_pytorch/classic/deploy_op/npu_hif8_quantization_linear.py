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
from amct_pytorch.common.utils.check_params import check_parameters_in_schema


class NpuHIF8Linear(torch.nn.Module):
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
        self._init_weight_quant(quant_module)
        if quant_module.bias is None:
            self.bias = None
        else:
            self.register_buffer('bias', quant_module.bias)
        self.is_support_hifloat8 = True if check_parameters_in_schema(torch_npu.npu_quant_matmul, 'x1_dtype') else False

    def forward(self, x):
        """
        do inference process
        Params:
        inputs: input data in torch.tensor.
        """
        import torch_npu
        quant_x, pertoken_scale = torch_npu.npu_dynamic_quant(x, dst_type=torch_npu.hifloat8, dst_type_max=15)
            
        if self.is_support_hifloat8:
            output = torch_npu.npu_quant_matmul(quant_x, self.quantized_weight,
                scale=self.deq_scale, pertoken_scale=pertoken_scale.view(-1),
                output_dtype=x.dtype, x1_dtype=torch_npu.hifloat8, x2_dtype=torch_npu.hifloat8)
            if self.bias is not None:
                output = output + self.bias
        else:
            raise RuntimeError(f'current environment do not support hifloat8 quantize')

        return output
        
    def _init_weight_quant(self, quant_module):
        fp8_weight = quant_module.weight # n, k
        fp8_scale = quant_module.weight_scale_inv
        if quant_module.block_size is not None: # block quantize
            n, k = quant_module.weight.shape
            block_h, block_w = quant_module.block_size
            fp8_scale = torch.repeat_interleave(fp8_scale, block_h, dim=0)
            fp8_scale = torch.repeat_interleave(fp8_scale, block_w, dim=1)
            fp8_scale = fp8_scale[:n, :k]
        dequant_weight = fp8_weight.to(torch.float32) / fp8_scale
        weight_max = dequant_weight.max(dim=1, keepdim=True).values
        scale = (weight_max / 16.0).to(torch.float32).flatten() # n
        self.register_buffer('deq_scale', scale)
        import torch_npu
        # k, n * n
        hif8_quantized_tensor = torch_npu.npu_quantize(dequant_weight.transpose(0, 1),
                                                       scale, None, dtype=torch_npu.hifloat8)
        self.register_buffer('quantized_weight', hif8_quantized_tensor)
