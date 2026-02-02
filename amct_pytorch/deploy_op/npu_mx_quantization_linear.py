# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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

from amct_pytorch.utils.vars import MXFP4_E2M1, MXFP8_E4M3FN


class NpuMXQuantizationLinear(torch.nn.Module):
    """
    Function: class for npu mx_data quant operator inherited from nn.module.
    APIs: forward
    """
    def __init__(self, ori_module, layer_name, quant_config):
        super().__init__()
        import torch_npu
        self.ori_module_type = type(ori_module).__name__
        self.ori_dtype = ori_module.weight.dtype
        self.weight = ori_module.weight
        self.wts_type = quant_config.get('weights_cfg').get('quant_type')
        self.group_size = quant_config.get('weights_cfg').get('group_size', None)
        device = ori_module.weight.device
        self.weight_compress_only = True

        if quant_config.get('inputs_cfg').get('enable_quant') is None or \
            quant_config.get('inputs_cfg').get('enable_quant') == True:
            self.weight_compress_only = False

        if self.wts_type == MXFP4_E2M1:
            weight_tensor, shared_exponent_w = torch_npu.npu_dynamic_mx_quant(
                self.weight, axis=-1, round_mode='rint', dst_type=torch_npu.float4_e2m1fn_x2, block_size=32)
            weight_tensor = torch_npu.npu_dtype_cast(
                weight_tensor.npu(), dtype=torch.float32, input_dtype=torch_npu.float4_e2m1fn_x2)
            weight_tensor = \
                torch_npu.npu_convert_weight_to_int4pack(weight_tensor.npu()).to(device=device)
            shared_exponent_w = shared_exponent_w.reshape(shared_exponent_w.shape[0], -1).transpose(-1, -2)
        elif self.wts_type == MXFP8_E4M3FN:
            weight_tensor, shared_exponent_w = torch_npu.npu_dynamic_mx_quant(
                self.weight, axis=-1, round_mode='rint', dst_type=torch.float8_e4m3fn, block_size=32)
            self.group_sizes = [1, 1, 32]
        
        weight_tensor = weight_tensor.npu().transpose(1, 0)
        self.register_buffer('quantized_weight', weight_tensor)
        self.register_buffer('scale_w', shared_exponent_w)

        if ori_module.bias is not None:
            bias = ori_module.bias
            if self.wts_type == MXFP8_E4M3FN:
                # npu op require bias be float32
                bias = bias.to(torch.float32)
            self.register_buffer('bias', bias)
        else:
            self.bias = None

    def forward(self, x):
        import torch_npu
        ori_shape = x.shape
        if len(ori_shape) < 2 or len(ori_shape) > 6:
            raise RuntimeError("Only support activation dims in [2,6] for linear.")

        # npu op only support dim=2 for mxfp4 and mxfp8 now
        if len(ori_shape) != 2:
            x = x.reshape(-1, ori_shape[-1])

        if self.weight_compress_only:
            output = torch_npu.npu_weight_quant_batchmatmul(x, self.quantized_weight, self.scale_w,
                antiquant_offset=None, bias=self.bias, weight_dtype=None,
                antiquant_group_size=32)
        else:
            quant_x, shared_exponent_d = torch_npu.npu_dynamic_mx_quant(
                x, axis=-1, round_mode='rint', dst_type=torch.float8_e4m3fn, block_size=32)
            output = torch_npu.npu_quant_matmul(quant_x, self.quantized_weight, self.scale_w,
                bias=self.bias, pertoken_scale=shared_exponent_d, output_dtype=self.ori_dtype,
                pertoken_scale_dtype=torch_npu.float8_e8m0fnu,
                scale_dtype=torch_npu.float8_e8m0fnu, group_sizes=self.group_sizes)

        if len(ori_shape) != 2:
            output = output.reshape(*ori_shape[:-1], -1)
        return output
