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

from amct_pytorch.amct_pytorch_inner.amct_pytorch.utils.data_utils import convert_dtype
from amct_pytorch.amct_pytorch_inner.amct_pytorch.utils.data_utils import float_to_mxfp8e4m3fn, mxfp8_convert_to_float, mxfp4_convert_to_float

DATA_MAP = {
    torch.float32: 0,
    torch.float16: 1,
    torch.bfloat16: 2
}

REVERSE_DATA_MAP = {v: k for k, v in DATA_MAP.items()}

def npu_dynamic_mx_quant(ori_tensor, axis=-1, round_mode='rint', dst_type=torch.float8_e4m3fn, block_size=32):
    converted_tensor, shared_exponent = float_to_mxfp8e4m3fn(ori_tensor)
    return converted_tensor, shared_exponent


def npu_quant_matmul(x1, x2, x2_scale, offset=None, bias=None, pertoken_scale=None, output_dtype=torch.bfloat16,
    x2_dtype=None, pertoken_scale_dtype=None, scale_dtype=None, group_sizes=None):
    if len(x1.shape) < 2 or len(x1.shape) > 6:
        raise RuntimeError()
    
    x1 = mxfp8_convert_to_float(x1, pertoken_scale, DATA_MAP.get(output_dtype))
    if x2.dtype == torch.float8_e4m3fn:
        x2_scale = x2_scale.reshape(x2_scale.shape[0], -1)
        x2 = mxfp8_convert_to_float(x2.transpose(1, 0), x2_scale, DATA_MAP.get(output_dtype))
    else:
        x2 = mxfp4_convert_to_float(x2.transpose(1, 0), x2_scale, DATA_MAP.get(output_dtype))
        n, k = x2.shape
        new_x2 = torch.zeros((n, 2 * k), dtype=x2.dtype, device=x2.device)
        new_x2[:, :k] = x2
        x2 = new_x2
    if bias is not None:
        bias = bias.to(torch.float32)
    out = torch.nn.functional.linear(x1.to(torch.float32), x2.to(torch.float32), bias).to(output_dtype)
    return out


@property
def float4_e2m1fn_x2():
    return


@property
def float8_e8m0fnu():
    return

@property
def float8_e4m3fn():
    return


def mock_npu_weight_quant_batchmatmul(x, weight, antiquant_scale,
    antiquant_offset=None, quant_scale=None, quant_offset=None,
    bias=None, antiquant_group_size=0, inner_precise=0,
    weight_dtype=None):
    ori_out_dtype = bias.dtype
    x = x.to(torch.float32)
    weight = weight.to(torch.float32)
    weight = weight.transpose(0, 1)
    antiquant_scale = antiquant_scale.to(torch.float32)
    # do anti quant for weight tensor
    weight *= antiquant_scale
    bias = bias.to(torch.float32)
 
    out = torch.nn.functional.linear(x, weight, bias)
    out = out.to(ori_out_dtype)
    return out
 
def mocked_npu_quant_conv2d(x, weight, scale, stride, pads,
    dilation, groups, offset_x, output_dtype, bias=None, input_dtype=None, weight_dtype=None):
    x = x.to(torch.float32)
    weight = weight.to(torch.float32)
    if bias is not None:
        bias = bias.to(torch.float32)
 
    out = torch.nn.functional.conv2d(x, weight, bias, stride, pads, dilation, groups)
    out *= scale.reshape(1, -1, 1, 1)
    out = out.to(output_dtype)
    return out
 
def mock_npu_quant_matmul(x, weight, scale, output_dtype, bias=None, x1_dtype=None, x2_dtype=None):
    x = x.to(torch.float32)
    weight = weight.to(torch.float32)
    if bias is not None:
        bias = bias.to(torch.float32)
 
    out = torch.nn.functional.linear(x, weight, bias)
    out *= scale.reshape(1, -1)
    out = out.to(output_dtype)
    return out
 
def mock_npu_quantize(input, scales, zero_points=None,
    dtype=torch.qint8, axis=1, div_mode=True):
    if div_mode:
        out = input / scales
    else:
        out = input * scales
    if zero_points is not None:
        zero_points_shape = [1] * len(input.shape)
        zero_points_shape[axis] = -1
        zero_points = zero_points.reshape(zero_points_shape)
        out += zero_points
    if dtype == torch.qint8:
        out = torch.quantize_per_tensor(input, 1, 0, dtype) 
    else:
        if dtype == torch.float8_e4m3fn:
            dtype = 'FLOAT8_E4M3FN'
        else:
            return out
        out = convert_dtype(out, dtype)
    return out
 
def mock_npu_trans_quant_param(scale, offset=None):
    if not offset:
        return scale
    return scale, offset

def mock_npu(self):
    return self