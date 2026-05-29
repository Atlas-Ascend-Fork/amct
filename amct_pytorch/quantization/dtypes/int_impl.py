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

import numpy as np
import torch


def round_ste(x: torch.Tensor) -> torch.Tensor:
    return (torch.round(x) - x).detach() + x


def dynamic_per_token_quant(tensor: torch.Tensor, bits: int) -> torch.Tensor:
    dtype = tensor.dtype
    qmax = 2 ** (bits - 1) - 1
    qmin = -2 ** (bits - 1)
    abs_max = torch.abs(tensor).max(dim=-1, keepdim=True)[0].clamp(min=1e-5)
    scale = (abs_max / qmax).clamp_min(1e-9)
    tensor = round_ste(tensor / scale)
    tensor = torch.clamp(tensor, qmin, qmax)
    tensor = tensor * scale
    return tensor.to(dtype)


def scale_fp32_to_u64(weight_scale):
    """
    Convert FP32 scale to UINT64 scale for W4A8 MoEGMM.
    """
    k, n = weight_scale.shape
    scale_np = weight_scale.detach().float().cpu().numpy()
    scale_uint32 = scale_np.astype(np.float32)
    scale_uint32.dtype = np.uint32
    scale_uint64 = np.zeros((k, n * 2), dtype=np.uint32)
    scale_uint64[..., ::2] = scale_uint32
    scale_uint64.dtype = np.uint64
    scale_uint64 = torch.from_numpy(scale_uint64).to(torch.uint64)
    return scale_uint64


def pack_4bit(x: torch.Tensor):
    """
    Pack int4 weight for W4A8 MoEGMM. Each two int4 numbers are packed into one byte.
    """
    assert x.dtype == torch.int8
    x = x.T.contiguous()  # pack along output channel dim.
    shape = x.shape
    x = x.view(-1, 2)
    # for example, 5(0b00000101) << 4 -> 0b01010000, -7 (0b11111001) & 0b00001111 -> 0b00001001,
    # then 0b01010000 | 0b00001001 -> 0b01011001
    x1 = x[:, 0]
    x2 = x[:, 1]
    y_x2 = torch.bitwise_left_shift(x2, 4)
    y_x1 = x1 & 0b00001111
    y = torch.bitwise_or(y_x1, y_x2)
    y = y.view(shape[0], shape[1] // 2)
    return y.T.contiguous()


def weight_quant(tensor: torch.Tensor, bits=8, real_quant=False, v: torch.Tensor = 0.0):
    assert tensor.dim() == 2
    qmax = 2 ** (bits - 1) - 1
    abs_max = torch.abs(tensor).max(dim=1, keepdim=True)[0]
    scale = (abs_max / qmax).clamp_min(1e-9)
    assert scale.shape == (tensor.shape[0], 1)
    if isinstance(v, torch.Tensor):
        v = v.to(device=tensor.device, dtype=tensor.dtype)
    quantized = round_ste(tensor / scale + v)
    quantized = torch.clamp(quantized, -qmax, qmax)
    if real_quant:
        if bits == 4:
            # pack 4bit for W4A8 MoEGMM
            quantized = quantized.to(torch.int8)
            bias = int4_assistance_bias(quantized, scale)
            quantized = pack_4bit(quantized)
            scale = scale_fp32_to_u64(scale)
            return quantized, scale, bias
        else:
            return quantized.to(torch.int8), scale.to(torch.float32), None
    else:
        return quantized * scale


def int4_assistance_bias(weight, weight_scale):
    """
    Calculate the int4 weight assistance bias matrix for W4A8 MoEGEMM.
    """
    repeat_times = weight.shape[1] // weight_scale.shape[1]
    expanded_scale = weight_scale.repeat_interleave(repeat_times, dim=1)
    # 8 is the max value of INT4, for normalizing the quantization range of assistance bias.
    weight_assistant_matrix = (expanded_scale * weight * 8).sum(dim=1).float()
    return weight_assistant_matrix
