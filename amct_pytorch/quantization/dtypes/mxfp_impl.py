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

import torch
from torch import Tensor
from enum import Enum, IntEnum
from numbers import Number
from torchao.prototype.custom_fp_utils import (
    _f32_to_floatx_unpacked,
    _floatx_unpacked_to_f32,
)


FP32_EXPONENT_BIAS = 127
FP32_MIN_NORMAL = 2 ** (-FP32_EXPONENT_BIAS + 1)
EBITS_F4_E2M1, MBITS_F4_E2M1 = 2, 1


def down_size(size):
    assert size[-1] % 2 == 0, f"{size} last dim not divisible by two"
    return (*size[:-1], size[-1] // 2)


def up_size(size):
    return (*size[:-1], size[-1] * 2)


def unpack_uint4(uint8_data) -> torch.Tensor:
    """Get the original weight from the normalized float weight format"""
    assert uint8_data.is_contiguous()

    shape = uint8_data.shape

    # since we are using uint8 we will decode 2 entries per byte
    # Shift elements down 4 and select out the bottom 4 bits
    #
    # Note: known slow with triton
    # * currently generates two kernels with a cat in between
    # * after https://github.com/pytorch/pytorch/pull/123278 lands I
    #   verified that we get a single triton kernel, but that is even slower
    #   than the two kernels before this PR
    # * TODO add a microbenchmark of just the cast and profile this
    first_elements = (uint8_data & 0b1111).to(torch.uint8)
    second_elements = (uint8_data >> 4).to(torch.uint8)
    unpacked = torch.stack([first_elements, second_elements], dim=-1).view(
        up_size(shape)
    )

    # trying Bert Maher's suggestion
    # 2024-04-04: this works in unit tests but is broken on LLaMa 7B FFN with
    #   ptxas /tmp/tmp84wp7lea.ptx, line 227; error   : Unexpected instruction types specified for 'sub'  # noqa: E501
    # which seems to be the same issue as https://github.com/pytorch/pytorch/issues/118589  # noqa: E501
    # TODO(later): try removing subtractions from our cast to see if we can work around  # noqa: E501
    # shift_tensor = torch.tensor([4, 0], dtype=torch.uint8, device=uint8_data.device)  # noqa: E501
    # unpacked = (uint8_data.reshape(-1)[::, None] >> shift_tensor) & 0b1111
    # unpacked = unpacked.view(up_size(shape))

    return unpacked


def pack_uint4(uint8_data: torch.Tensor) -> torch.Tensor:
    # converting to uint8 for operations
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
    exponent = torch.where(mantissa > 1.75, exponent+1, exponent)
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
    if isinstance(v, torch.Tensor):
        v = v.to(device=x.device, dtype=x.dtype)
    x = round_ste(x + v)
    x = x / shift_val * private_exp
    x = torch.clamp(x, min=-max_norm, max=max_norm)
    return x
