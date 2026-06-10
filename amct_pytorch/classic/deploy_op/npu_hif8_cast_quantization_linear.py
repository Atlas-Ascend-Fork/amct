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

from amct_pytorch.common.utils.vars import HIFLOAT8
from amct_pytorch.common.utils.log import LOGGER
from amct_pytorch.common.utils.quant_util import convert_dtype, quant_tensor


class NpuHIF8CastLinear(torch.nn.Module):
    """
    Function: quantization operator to cast weight/activation to low precision (HIF8)
    Features: 
        1. Weight: direct cast to target dtype without calibration
        2. Activation: insert cast operator in forward for online conversion
        3. Support HIF8 data types
    APIs: forward.
    """
    def __init__(self, quant_module):
        """
        Function: init objective.
        Args:
            quant_module: HIF8CastQuant. The cast quantize op holding the weight, bias and
                          pre-computed weight scale; this deploy op reads them directly so
                          the deterministic cast scale is never recomputed.
        """
        super().__init__()
        self.device = quant_module.weight.device
        self.layer_name = quant_module.layer_name
        self.weight_compress_only = quant_module.weight_compress_only

        self.register_buffer('scale_w', quant_module.scale_w)
        weight, _ = quant_tensor(quant_module.weight.transpose(1, 0), quant_module.wts_type, scale=self.scale_w)
        self.register_buffer('quantized_weight', weight.to(self.device))

        bias = quant_module.bias
        if bias is not None and not self.weight_compress_only:
            # npu_quant_matmul adds bias before scaling, so pre-divide by scale_w.
            bias = bias.to(torch.float32) / self.scale_w
        self.register_buffer('bias', bias)

    @torch.no_grad()
    def forward(self, x):
        """
        Function: NpuHIF8CastLinear forward function (core feature 3: activation online cast)
        Args:
            x: torch.tensor. Input data for forward computation.
        Return:
            torch.tensor. Output after activation cast and original module forward.
        """
        import torch_npu
        if self.weight_compress_only:
            # input shape reshape to 2d for npu op
            ori_shape = x.shape
            x = x.reshape(-1, x.shape[-1]).to(self.device)
            output = torch_npu.npu_weight_quant_batchmatmul(x, self.quantized_weight,
                self.scale_w.to(x.dtype), bias=self.bias, weight_dtype=torch_npu.hifloat8)
            output = output.reshape(*ori_shape[:-1], -1)
        else:
            quantized_x = convert_dtype(x, HIFLOAT8)
            output = torch_npu.npu_quant_matmul(quantized_x, self.quantized_weight, self.scale_w,
                x1_dtype=torch_npu.hifloat8, x2_dtype=torch_npu.hifloat8,
                pertoken_scale=None, bias=self.bias, output_dtype=x.dtype)
            LOGGER.logd(f"Cast activation of layer '{self.layer_name}' to {HIFLOAT8} success",
                'NpuHIF8CastLinear')
        return output
