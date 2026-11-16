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
"""Shared fp16 fixed point primitives for the fp16fixed engine family.

Each engine carries its own frozen reference contract (clamp bounds,
rounding carrier, bias ordering), but the low-level primitives -- the
INT32 saturation and the per-element tile alignment -- are structurally
identical apart from the alignment clamp range. They live here once,
parameterized by the engine's contract, so the "hardware alignment
range" has a single code-level definition point (PR review: the
duplicate drift surface bit us once already with the ``_rint_half_even``
/ ``rint_process`` shift == 0 divergence).
"""

import torch

INT32_MIN, INT32_MAX = -(2**31), 2**31 - 1


def saturate_to_int32(values_int64):
    """Saturate out-of-range values to INT32_MIN (hardware behaviour).

    Overflow in either direction collapses to INT32_MIN, not to the
    respective bound of the int32 range.
    """
    out_of_bounds = (values_int64 > INT32_MAX) | (values_int64 < INT32_MIN)
    return torch.where(
        out_of_bounds, torch.full_like(values_int64, INT32_MIN), values_int64
    )


def to_fixed_point_source(
    exponent_raw,
    mantissa_raw,
    sign_bit,
    is_normal,
    tile_max_exponent,
    align_low,
    align_high,
):
    """Align one fp16 tile element onto the tile max exponent.

    Every element of the tile is expressed on the grid of the tile max
    raw exponent: ``value = fixed_point_source * 2 ** (tile_max_exponent
    - 29)``. Normal numbers carry the implicit leading 1 (mantissa +
    1024, 11 bits) with an exponent base of 4, subnormals use the raw
    10-bit mantissa with a base of 5 and no implicit bit -- both maximal
    left shifts (2047 << 4, 1023 << 5) stay inside int16, the fixed
    point container of the hardware. The per-element alignment shift is
    clamped to ``[align_low, align_high]`` (the engine's hardware
    alignment range): elements far below the tile max are truncated
    instead of overflowing.

    Args:
        exponent_raw: (..., K) int32, the raw 5-bit exponent field.
        mantissa_raw: (..., K) int32, the raw 10-bit mantissa field.
        sign_bit: (..., K) int32, 0 or 1.
        is_normal: (..., K) bool, exponent_raw > 0.
        tile_max_exponent: (...,) int32, the per-tile max exponent (the
            implementation unsqueezes a trailing dim for broadcasting).
        align_low: int, the engine's alignment clamp lower bound
            (conv2d: -11, convt / matmul: -16).
        align_high: int, the engine's alignment clamp upper bound
            (conv2d: 5, convt / matmul: 15).

    Returns:
        (..., K) int64, the signed fixed point source values.
    """
    exponent_delta = tile_max_exponent.unsqueeze(-1) - exponent_raw
    exponent_base = torch.where(is_normal, 4, 5)
    aligned_exponent = torch.clamp(
        exponent_base - exponent_delta, align_low, align_high
    )
    mantissa = torch.where(
        is_normal,
        mantissa_raw.to(torch.int64) + 1024,
        mantissa_raw.to(torch.int64),
    )
    zero_shift = torch.zeros_like(aligned_exponent)
    left_shift_amount = torch.where(
        aligned_exponent >= 0, aligned_exponent, zero_shift
    ).to(torch.int64)
    right_shift_amount = torch.where(
        aligned_exponent < 0, -aligned_exponent, zero_shift
    ).to(torch.int64)
    fixed_point_source = torch.where(
        aligned_exponent >= 0,
        mantissa << left_shift_amount,
        mantissa >> right_shift_amount,
    )
    sign = torch.where(sign_bit == 1, -1, 1).to(torch.int64)
    return fixed_point_source * sign
