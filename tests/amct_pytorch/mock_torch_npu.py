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
import math
import torch

from amct_pytorch.utils.quant_util import convert_dtype


@property
def float4_e2m1fn_x2():
    return


@property
def float8_e8m0fnu():
    return


@property
def float8_e4m3fn():
    return


def npu_quant_matmul(x1, x2, x2_scale, offset=None, bias=None, pertoken_scale=None, output_dtype=torch.bfloat16,
    x2_dtype=None, pertoken_scale_dtype=None, scale_dtype=None, group_sizes=None):
    if len(x1.shape) < 2 or len(x1.shape) > 6:
        raise RuntimeError()
    if x2.dtype == torch.float8_e4m3fn:
        x2_scale = x2_scale.reshape(x2_scale.shape[0], -1)
    if bias is not None:
        bias = bias.to(torch.float32)
    out = torch.nn.functional.linear(x1.to(torch.float32), x2.to(torch.float32), bias).to(output_dtype)
    return out


def mock_npu_weight_quant_batchmatmul(x, weight, antiquant_scale,
    antiquant_offset=None, quant_scale=None, quant_offset=None,
    bias=None, antiquant_group_size=0, inner_precise=0,
    weight_dtype=None):
    ori_dtype = x.dtype
    x_fp32 = x.float()
    weight_fp32 = weight.float()
    scale_fp32 = antiquant_scale.float()
    bias_fp32 = bias.float() if bias is not None else None

    if weight_fp32.dim() == 3:
        N_div_8, K, _ = weight_fp32.shape
        N = N_div_8 * 8
        weight_fp32 = weight_fp32.reshape(K, N)

    if x_fp32.shape[-1] != weight_fp32.shape[1]:
        weight_fp32 = weight_fp32.transpose(0, 1)
    
    if scale_fp32.dim() == 1 and scale_fp32.numel() == weight_fp32.shape[0]:
        weight_fp32 = weight_fp32 * scale_fp32.unsqueeze(1)

    out = torch.nn.functional.linear(x_fp32, weight_fp32, bias_fp32)
    out = out.to(ori_dtype)
    return out


def mock_npu_quant_matmul(x, weight, scale, pertoken_scale, bias=None, output_dtype=torch.float32,
    x1_dtype=None, x2_dtype=None, group_sizes=None, y_scale=None, pertoken_scale_dtype=None,
    scale_dtype=None):
    x = x.to(torch.float32)
    weight = weight.to(torch.float32)
    if bias is not None:
        bias = bias.to(torch.float32)
    if x2_dtype is not None and 'float4_e2m1fn_x2' in str(x2_dtype):
        n, k = weight.shape
        new_weight = torch.randn((n, 2 * k), dtype=weight.dtype, device=weight.device)
        new_weight[:, :k] = weight
        new_weight = new_weight.reshape(-1, int(new_weight.shape[-1] / scale.shape[-1])) * scale.reshape(-1, 1)
        weight = new_weight.reshape(n, 2 * k)
    else:
        weight = weight.transpose(0, 1)
    out = torch.nn.functional.linear(x, weight, bias)
    if y_scale is not None:
        out *= y_scale
    elif scale_dtype is not None and 'float8_e8m0fnu' in str(scale_dtype):
        out = (out.reshape(-1, 32) * scale.reshape(-1, 1)).reshape(out.shape)
    else:
        out *= scale.reshape(1, -1)
    out = out.to(output_dtype)
    return out


def mock_npu_quantize(input, scales, zero_points=None,
    dtype=torch.qint8, axis=1, div_mode=True):
    if div_mode:
        out = input / scales
    else:
        out = input.transpose(-1, -2) * scales
        out = out.transpose(-1, -2)
    if zero_points is not None:
        zero_points_shape = [1] * len(input.shape)
        zero_points_shape[axis] = -1
        zero_points = zero_points.reshape(zero_points_shape)
        out += zero_points
    if dtype == torch.qint8:
        out = torch.quantize_per_tensor(input, 1, 0, dtype)

    return out


def mock_npu_convert_weight_to_int4pack(weight, inner_k_tiles=0):
    return weight


def mock_npu_dynamic_mx_quant(weight, axis=None, round_mode=None, 
        dst_type=None, block_size=None):
    shape = (weight.shape[0], math.ceil(weight.shape[1] / 64), 2)
    scale = torch.randn(shape)
    if dst_type != torch.float8_e4m3fn:
        weight = weight[:, :int(weight.shape[1] // 2)]
    return weight, scale


def mock_npu_dtype_cast(weight, dtype, input_dtype=None):
    if input_dtype is not None and 'float4_e2m1fn_x2' in str(input_dtype):
        n, k = weight.shape
        new_weight = torch.randn((n, 2 * k), dtype=weight.dtype, device=weight.device)
        new_weight[:, :k] = weight
        weight = new_weight
    if 'float4_e2m1fn_x2' in str(dtype):
        weight = weight[:, :int(weight.shape[1] // 2)]
    return weight


def mock_npu_format_cast(weight, dst):
    return weight


def mock_npu(self):
    return self


def mock_npu_trans_quant_param(scale, offset=None):
    if not offset:
        return scale
    return scale, offset


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