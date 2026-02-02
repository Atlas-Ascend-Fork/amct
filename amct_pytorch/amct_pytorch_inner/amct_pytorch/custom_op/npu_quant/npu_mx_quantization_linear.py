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

from ....amct_pytorch.utils.data_utils import convert_mx_dtype
from ....amct_pytorch.utils.vars import GROUP_SIZE, MXFP4_E2M1, MXFP8_E4M3FN


class NpuMXQuantizationLinear(torch.nn.Module):
    """
    Function: class for npu mx_data quant operator inherited from nn.module.
    APIs: forward
    """
    def __init__(self, ori_module, quant_params):
        super().__init__()
        self.ori_dtype = ori_module.weight.dtype
        self.wts_type = quant_params.get('wts_type')
        weight_tensor, shared_exponent = convert_mx_dtype(ori_module.weight, self.wts_type)
        import torch_npu
        weight_tensor = weight_tensor.npu().transpose(1, 0)
        self.register_buffer('weight', weight_tensor)
        self.register_buffer('shared_exponent_w', shared_exponent)
        
        if self.wts_type == MXFP4_E2M1:
            self.x2_dtype = torch_npu.float4_e2m1fn_x2
            self.group_sizes = [0, 0, 32]
        elif self.wts_type == MXFP8_E4M3FN:
            # npu op require bias be float32
            self.x2_dtype = None
            self.group_sizes = [1, 1, 32]
            self.shared_exponent_w = self.shared_exponent_w.reshape(self.shared_exponent_w.shape[0], -1, 2)

        if ori_module.bias is not None:
            bias = ori_module.bias
            # npu op require 2-dim bias
            if self.wts_type == MXFP4_E2M1:
                bias = bias.unsqueeze(0)
            elif self.wts_type == MXFP8_E4M3FN:
                # npu op require bias be float32
                bias = bias.to(torch.float32)
            self.register_buffer('bias', bias)
        else:
            self.bias = None

    def forward(self, x):
        ori_shape = x.shape
        if len(ori_shape) < 2 or len(ori_shape) > 6:
            raise RuntimeError("Only support activation dims in [2,6] for linear.")

        # npu op only support dim=2 for mxfp4 and mxfp8 now
        if len(ori_shape) != 2:
            x = x.reshape(-1, ori_shape[-1])

        import torch_npu
        quant_x, shared_exponent_d = torch_npu.npu_dynamic_mx_quant(
            x, axis=-1, round_mode='rint', dst_type=torch.float8_e4m3fn, block_size=GROUP_SIZE)
        if self.wts_type == MXFP4_E2M1:
            shared_exponent_d = shared_exponent_d.reshape(shared_exponent_d.shape[0], -1)
        output = torch_npu.npu_quant_matmul(quant_x, self.weight, self.shared_exponent_w,
            bias=self.bias, pertoken_scale=shared_exponent_d, output_dtype=self.ori_dtype,
            x2_dtype=self.x2_dtype, pertoken_scale_dtype=torch_npu.float8_e8m0fnu,
            scale_dtype=torch_npu.float8_e8m0fnu, group_sizes=self.group_sizes)

        if len(ori_shape) != 2:
            output = output.reshape(*ori_shape[:-1], -1)
        return output
