# coding=utf-8
# Adapted from
# https://github.com/microsoft/microxcaling/blob/main/mx/mx_ops.py
# https://github.com/pytorch/ao/blob/main/torchao/prototype/mx_formats/kernels.py
#
# Copyright (c) Microsoft Corporation.
# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from enum import Enum, IntEnum

import torch
from torch import Tensor
from torchao.prototype.custom_fp_utils import (
    _f32_to_floatx_unpacked,
    _floatx_unpacked_to_f32,
)

FP32_EXPONENT_BIAS = 127
FP32_MIN_NORMAL = 2 ** (-FP32_EXPONENT_BIAS + 1)
EBITS_F4_E2M1, MBITS_F4_E2M1 = 2, 1


def unpack_mxfloat4_to_fp32(packed_tensor):
    e2m1_values = torch.tensor([
        0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
        -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0
    ], dtype=torch.float32, device=packed_tensor.device)

    low_4bits = packed_tensor & 0x0F
    high_4bits = (packed_tensor // 16) & 0x0F

    unpacked = torch.stack([low_4bits, high_4bits], dim=-1)

    fp32_tensor = e2m1_values[unpacked.long()]
    new_shape = list(packed_tensor.shape)
    new_shape[-1] = new_shape[-1] * 2

    return fp32_tensor.view(*new_shape)


def weight_dequant(weight: torch.Tensor, scale: torch.Tensor, block_size: int = 128, 
                   is_mx: bool = False, is_packed: bool = False) -> torch.Tensor:
    """
    Dequantizes the given weight tensor using the provided scale tensor, efficiently handling cases where
    `weight` is not a multiple of `block_size` by broadcasting `scale`.

    Args:
        weight (torch.Tensor): The quantized weight tensor of shape(M, N).
        scale (torch.Tensor): The scale tensor of shape (M // block_size, N // block_size).
        block_size (int, optional): The block size to use for dequantization. Defaults to 128.

    Returns:
        torch.Tensor: The dequantized weight tensor of the same shape as `weight`, converted to the default dtype.

    Raises:
        AssertionError: If `scale` dimensions do not align with `weight` shape after scaling.
    """
    if is_packed:
        weight = unpack_mxfloat4_to_fp32(weight.view(torch.uint8))
    # Get the original dimensions of weight
    M, N = weight.shape
    weight = weight.to(torch.float32)
    scale = scale.to(torch.float32)
    if block_size != 1:
        if is_mx:
            scale_expanded = scale.repeat_interleave(block_size, dim=1)
        else:
            # Compute the effective block dimensions for scale
            scale_m, scale_n = scale.shape
            assert scale_m == (
                M + block_size - 1) // block_size, "Mismatch in scale rows and weight rows."
            assert scale_n == (
                N + block_size - 1) // block_size, "Mismatch in scale columns and weight columns."

            # Expand scale to match the weight tensor's shape
            scale_expanded = scale.repeat_interleave(
                block_size, dim=0).repeat_interleave(block_size, dim=1)

        # Trim scale_expanded to match weight's shape if necessary
        scale_expanded = scale_expanded[:M, :N]

        # Perform element-wise multiplication
        dequantized_weight = weight * scale_expanded
    else:
        dequantized_weight = weight * scale

    # Convert the output to the default dtype
    dequantized_weight = dequantized_weight.to(torch.get_default_dtype())

    return dequantized_weight


def down_size(size):
    assert size[-1] % 2 == 0, f"{size} last dim not divisible by two"
    return (*size[:-1], size[-1] // 2)


def up_size(size):
    return (*size[:-1], size[-1] * 2)


def unpack_uint4(uint8_data) -> torch.Tensor:
    """Get the original weight from the normalized float weight format"""
    assert uint8_data.is_contiguous()

    shape = uint8_data.shape

    first_elements = (uint8_data & 0b1111).to(torch.uint8)
    second_elements = (uint8_data >> 4).to(torch.uint8)
    unpacked = torch.stack([first_elements, second_elements], dim=-1).view(
        up_size(shape)
    )

    return unpacked


def pack_uint4(uint8_data: torch.Tensor) -> torch.Tensor:
    shape = uint8_data.shape
    assert shape[-1] % 2 == 0
    uint8_data = uint8_data.contiguous().view(-1)
    return (uint8_data[::2] | uint8_data[1::2] << 4).view(down_size(shape))


def f32_to_f4_unpacked(x):
    """
    Input: torch.Tensor of dtype torch.float
    Output: torch.Tensor of dtype torch.uint8, with bits 0-3 empty and
      bits 4-7 in fp4_e2m1
    """
    return _f32_to_floatx_unpacked(x, EBITS_F4_E2M1, MBITS_F4_E2M1)


def f4_unpacked_to_f32(x: torch.Tensor):
    """
    Input: torch.Tensor of dtype uint8, with bits 0-3 empty and bits 4-7
      containing an fp4_e2m1 encoding
    Output: torch.Tensor of dtype fp32 with the dequantized value
    """
    return _floatx_unpacked_to_f32(x, EBITS_F4_E2M1, MBITS_F4_E2M1)


def round_to_decimal(x):
    exponent = torch.floor(torch.log2(x))
    mantissa = x / (2**exponent)
    exponent = torch.where(mantissa > 1.75, exponent + 1, exponent)
    return exponent


def shared_exponents(x: Tensor, emax):
    shared_exp, _ = torch.max(torch.abs(x), dim=-1, keepdim=True)
    shared_exp = shared_exp + FP32_MIN_NORMAL * (shared_exp == 0).type(shared_exp.dtype)
    shared_exp = round_to_decimal(shared_exp)
    shared_exp = shared_exp - emax
    shared_exp = shared_exp.clamp(min=-127, max=1e10)
    return shared_exp


def round_ste(x: torch.Tensor) -> torch.Tensor:
    rounded = torch.sign(x) * torch.floor(torch.abs(x) + 0.5)
    return (rounded - x).detach() + x


def quantize_elewise(x, min_exp, max_norm, shift_val, v=0.0):
    private_exp = torch.floor(torch.log2(torch.abs(x) + (x == 0).type(x.dtype)))
    private_exp = 2 ** (private_exp.clip(min=min_exp))
    x = x / private_exp * shift_val
    x = round_ste(x + v)
    x = x / shift_val * private_exp
    x = torch.clamp(x, min=-max_norm, max=max_norm)
    return x
