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
from typing import Optional


DENSE_W4A8_GROUP_SIZE = 256


def round_ste(x: torch.Tensor) -> torch.Tensor:
    return (torch.round(x) - x).detach() + x


def dynamic_per_token_quant(tensor: torch.Tensor, bits: int) -> torch.Tensor:
    dtype = tensor.dtype
    qmax = 2 ** (bits - 1) - 1
    qmin = -(2 ** (bits - 1))
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


def dense_linear_w4a8_quant(
    tensor: torch.Tensor,
    scale: Optional[torch.Tensor] = None,
    group_size: int = DENSE_W4A8_GROUP_SIZE,
):
    """Export dense W4A8 weights in the vLLM-Ascend legacy layout.

    The runtime accepts one logical INT4 value in each INT8 element and packs
    it for the NPU during model loading.  AMCT quantizes weights per output
    channel, so the same channel scale is represented as the first-level scale
    and the second-level group scale is one for every input group.
    """
    if tensor.dim() != 2:
        raise ValueError(f"dense W4A8 weight must be 2-D, got {tensor.dim()}-D.")
    if type(group_size) is not int or group_size <= 0:
        raise ValueError("dense W4A8 group_size must be a positive int.")
    rows, columns = tensor.shape
    if columns % group_size != 0:
        raise ValueError(
            f"dense W4A8 input width {columns} must be divisible by "
            f"group_size {group_size}."
        )

    if scale is None:
        scale = (
            tensor.detach()
            .float()
            .abs()
            .amax(dim=1, keepdim=True)
            .div(7)
            .clamp_min(1e-9)
        )
    if not isinstance(scale, torch.Tensor) or tuple(scale.shape) != (rows, 1):
        shape = tuple(scale.shape) if isinstance(scale, torch.Tensor) else None
        raise ValueError(f"dense W4A8 scale must have shape {(rows, 1)}, got {shape}.")
    if not torch.isfinite(scale).all() or not torch.all(scale > 0):
        raise ValueError("dense W4A8 scale values must be finite and positive.")

    device_scale = scale.to(device=tensor.device, dtype=tensor.dtype)
    qweight = torch.round(tensor / device_scale).clamp(-7, 7).to(torch.int8)
    params_dtype = tensor.dtype if tensor.is_floating_point() else torch.float32
    group_count = columns // group_size
    return {
        "qweight": qweight.detach().cpu(),
        "weight_scale": device_scale.to(params_dtype).detach().cpu(),
        "weight_offset": torch.zeros(
            (rows, 1), dtype=params_dtype, device=tensor.device
        ).cpu(),
        "weight_scale_second": torch.ones(
            (rows, group_count), dtype=params_dtype, device=tensor.device
        ).cpu(),
        "weight_offset_second": torch.zeros(
            (rows, group_count), dtype=params_dtype, device=tensor.device
        ).cpu(),
    }


def int4_assistance_bias(weight, weight_scale):
    """
    Calculate the int4 weight assistance bias matrix for W4A8 MoEGEMM.
    """
    repeat_times = weight.shape[1] // weight_scale.shape[1]
    expanded_scale = weight_scale.repeat_interleave(repeat_times, dim=1)
    # 8 is the max value of INT4, for normalizing the quantization range of assistance bias.
    weight_assistant_matrix = (expanded_scale * weight * 8).sum(dim=1).float()
    return weight_assistant_matrix
