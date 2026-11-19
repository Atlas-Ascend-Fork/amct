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
"""fp16fixed Conv2d class implementation.

``Fp16FixedConv2d`` subclasses ``torch.nn.Conv2d`` (attributes and parameters
are fully reused) and overrides ``__init__`` / ``forward`` so that the
operator runs the fp16fixed F16xF16 fixed point Cube Unit simulation. The
numerical path is identical to the frozen numpy reference of this
engine (integer-domain half-even rounding, per-element exponent clamp
``[-11, 5]``, bias added after the last tile).

Shift == 0 degenerate tie (documented contract): the reference's
half-even shifter treats an ODD product at zero shift as an exact tie
and carries it up to the even neighbour (``remainder == half == 0`` with
the lsb check in its ``_rint_half_even``); the engine reproduces this
carry. Only the conv2d reference defines a shift 0 behaviour at all --
the historical numpy reference is undefined there (negative shift) and
the other engine families (matmul / convtranspose / linear / S8) round
via ``round_()`` per their own frozen references, so the families
intentionally differ here exactly as they differ in clamp bounds,
rounding carrier and bias ordering. Hardware spec confirmation of this
carry is tracked as a follow-up; the user visible impact is zero (the
accumulator LSB ``2**(sv-58)`` sits far below the fp16 ulp of the
shift 0 output scale).

Naming convention of the fixed point pipeline:

- ``exponent_raw`` / ``mantissa_raw`` / ``sign_bit``: fields decoded from
  the fp16 bit pattern;
- ``tile_max_exponent``: max raw exponent within one 16-element tile, the
  tile wide alignment anchor;
- ``aligned_exponent``: per-element exponent relative to the tile anchor,
  clamped to ``[-11, 5]``;
- ``fixed_point_source``: mantissa shifted onto the tile anchor scale,
  carrying the sign.

How one F16xF16 matmul is simulated (``_matmul_f16f16_int32``):

1. both fp16 operands are reinterpreted as bit patterns and decoded
   into sign / raw exponent / raw mantissa;
2. the K axis is split into tiles of 16; within one tile every element
   is aligned onto the tile max raw exponent and converted to a fixed
   point mantissa (see ``_to_fixed_point_source`` for the grid);
3. the 16-element integer MAC yields ``tile_mac``, which is rescaled by
   ``tile_shift = fixed_shift_value - input_tile_max_exponent -
   weight_tile_max_exponent`` onto the global accumulator grid whose
   LSB is worth ``2 ** (fixed_shift_value - 58)`` of real value;
4. the accumulator is saturated to INT32 after every tile and the bias
   (pre-scaled onto the same grid) is added after the last tile.

``fixed_shift_value`` therefore positions the global accumulator grid:
a larger value gives a finer grid (less rounding noise) but less
headroom against INT32 overflow, which is why it is searched per layer.
"""

import numpy as np
import torch

from ....utils.vars import FIXED_SHIFT_VALUE_FP16_FP16_DEFAULT
from .common import saturate_to_int32 as _saturate_to_int32
from .common import to_fixed_point_source
from .shift_value_search import ShiftValueSearchMixin

INT32_MIN, INT32_MAX = -(2**31), 2**31 - 1


def _to_fp16_single_round(values):
    """float64 -> fp16 with a SINGLE rounding step.

    torch's cast routes through float32 (c10::Half only has a float
    constructor), so values adjacent to an fp16 midpoint double round
    and land 1 ulp off the frozen references (numpy ``astype`` rounds
    once). Measured impact on the sv search (200 seeds): 56% of runs
    had perturbed SSE vectors and 20.5% flipped the argmin. numpy's
    fp16 cast is scalar (about 100x slower than the torch cast) but
    the conversion stays under 1% of one candidate forward. The
    non-CPU path keeps the torch cast: a
    ``.numpy()`` roundtrip would force a device sync per forward and
    the accelerator acceptance is argmin / relative tolerance based,
    not bit parity.
    """
    if values.device.type != 'cpu':
        # non-CPU device: numpy() would force a device sync; use the
        # torch cast instead -- the acceptance is argmin / relative
        # tolerance based, not bit parity
        return values.to(torch.float16)
    return torch.from_numpy(values.numpy().astype(np.float16))


def _matmul_f16f16_int32(input_matrix, weight_matrix, bias, fixed_shift_value=42):
    """F16xF16 matmul of one im2col block, returns the INT32 accumulator.

    ``input_matrix`` is ``[rows, reduction]``, ``weight_matrix`` is
    ``[output_channels, reduction]``, both fp16; the K axis is processed
    in tiles of 16.
    """
    num_rows, reduction_size = input_matrix.shape
    num_output_channels, _ = weight_matrix.shape
    if reduction_size % 16:
        pad_size = 16 - reduction_size % 16
        input_matrix = torch.nn.functional.pad(input_matrix, (0, pad_size))
        weight_matrix = torch.nn.functional.pad(weight_matrix, (0, pad_size))
        reduction_size = input_matrix.shape[-1]

    # decode the fp16 bit patterns: sign(1) | raw exponent(5) | mantissa(10)
    input_bits = input_matrix.contiguous().view(torch.int16).to(torch.int32)
    weight_bits = weight_matrix.contiguous().view(torch.int16).to(torch.int32)
    input_sign = (input_bits >> 15) & 1
    input_exponent = (input_bits >> 10) & 0x1F
    input_mantissa = input_bits & 0x3FF
    weight_sign = (weight_bits >> 15) & 1
    weight_exponent = (weight_bits >> 10) & 0x1F
    weight_mantissa = weight_bits & 0x3FF
    input_is_normal = input_exponent > 0
    weight_is_normal = weight_exponent > 0

    # pre-scale the bias onto the accumulator grid (LSB = 2**(sv-58) of
    # real value); this op adds it after the last tile
    scaled_bias = torch.round(
        bias.to(torch.float32) * float(2 ** (58 - fixed_shift_value))
    ).to(torch.int64)
    bias_int32 = _saturate_to_int32(scaled_bias)

    # the accumulator lives in float64: every tile result is an integer
    # inside the INT32 domain and int32-range doubles are exact, so the
    # saturation contract is bit-identical with the previous int64
    # accumulator while halving the intermediate traffic
    accumulator = torch.zeros(
        (num_rows, num_output_channels),
        dtype=torch.float64,
        device=input_matrix.device,
    )

    # vectorized: compute all tile fixed points and max exponents at once
    # (replaces the per-tile _to_fixed_point_source calls in the loop)
    num_tiles = reduction_size // 16

    # input side: reshape K -> (tiles, 16), compute per-tile amax, align all
    input_exp_tiles = input_exponent.reshape(num_rows, num_tiles, 16)
    input_tile_max_all = input_exp_tiles.amax(-1).to(torch.int64)  # (rows, T)

    input_exp_base_all = (
        torch.where(input_exponent > 0, 4, 5)
        .to(torch.int64)
        .reshape(num_rows, num_tiles, 16)
    )
    input_man_all = torch.where(
        input_is_normal,
        input_mantissa.to(torch.int64) + 1024,
        input_mantissa.to(torch.int64),
    ).reshape(num_rows, num_tiles, 16)
    input_sign_all = (
        torch.where(input_sign == 1, -1, 1)
        .to(torch.int64)
        .reshape(num_rows, num_tiles, 16)
    )

    input_delta = input_tile_max_all.unsqueeze(-1) - input_exp_tiles.to(torch.int64)
    # the per-element exponent clamp of the conv engine is [-11, 5]
    # (the frozen reference contract, see _to_fixed_point_source and
    # the numpy reference) -- NOT the [-16, 15] of the matmul / convt
    # engines; the alignment of elements whose tile exponent delta
    # exceeds 15 collapses onto -11
    input_aligned_exp = (input_exp_base_all - input_delta).clamp(-11, 5)
    input_fixed = torch.where(
        input_aligned_exp >= 0,
        input_man_all << input_aligned_exp.clamp_min(0),
        input_man_all >> (-input_aligned_exp).clamp_min(0),
    )
    input_all_fp = (
        input_fixed.to(torch.float64) * input_sign_all.to(torch.float64)
    ).reshape(num_rows, -1)  # (rows, K)

    # weight side: same vectorized alignment
    weight_exp_tiles = weight_exponent.reshape(num_output_channels, num_tiles, 16)
    weight_tile_max_all = weight_exp_tiles.amax(-1).to(torch.int64)  # (N, T)

    weight_exp_base_all = (
        torch.where(weight_exponent > 0, 4, 5)
        .to(torch.int64)
        .reshape(num_output_channels, num_tiles, 16)
    )
    weight_man_all = torch.where(
        weight_is_normal,
        weight_mantissa.to(torch.int64) + 1024,
        weight_mantissa.to(torch.int64),
    ).reshape(num_output_channels, num_tiles, 16)
    weight_sign_all = (
        torch.where(weight_sign == 1, -1, 1)
        .to(torch.int64)
        .reshape(num_output_channels, num_tiles, 16)
    )

    weight_delta = weight_tile_max_all.unsqueeze(-1) - weight_exp_tiles.to(torch.int64)
    weight_aligned_exp = (weight_exp_base_all - weight_delta).clamp(-11, 5)
    weight_fixed = torch.where(
        weight_aligned_exp >= 0,
        weight_man_all << weight_aligned_exp.clamp_min(0),
        weight_man_all >> (-weight_aligned_exp).clamp_min(0),
    )
    weight_all_fp = (
        weight_fixed.to(torch.float64) * weight_sign_all.to(torch.float64)
    ).reshape(num_output_channels, -1)  # (N, K)

    # exact path: per-tile MAC loop with the shift factor decomposed
    # into an outer product (the R5 scheme of the matmul / convt
    # engines, ported to this conv contract). The per-tile rescale
    # ``round_half_even(mac * 2**-(clamp(sv - Ea - Eb, -26, 36)))``
    # factors as ``2**(Ea - sv) (rows) x 2**Eb (N)`` -- both exact
    # powers of two, so ``mul_ + round_`` is bit-identical with the
    # previous clamp + where + integer shift chain (a power-of-two
    # multiply equals the shift by the same power and torch.round is
    # half to even, the hardware rounding mode). Tiles whose (m, n)
    # shifts all stay inside [-26, 36] (pre-checked from the per-tile
    # min / max exponents) take the pure outer product; only clamped
    # tiles pay the two correction passes. Every tile keeps its
    # round + INT32 saturation (the sequential contract with the numpy
    # reference). Products beyond 2**53 always saturate (the documented
    # fp64 envelope), so the fp64 factor multiply is bit-exact with the
    # int64 shift it replaces.
    input_factor = torch.exp2(
        (input_tile_max_all - fixed_shift_value).to(torch.float64)
    )  # (rows, T)
    weight_factor = torch.exp2(weight_tile_max_all.to(torch.float64))  # (N, T)
    input_exp_min = input_tile_max_all.amin(dim=0)  # (T,)
    input_exp_max = input_tile_max_all.amax(dim=0)
    weight_exp_min = weight_tile_max_all.amin(dim=0)  # (T,)
    weight_exp_max = weight_tile_max_all.amax(dim=0)
    # clean tile: every (m, n) shift of the tile stays in [-26, 36]
    clean_tile = (fixed_shift_value - input_exp_min - weight_exp_min <= 36) & (
        fixed_shift_value - input_exp_max - weight_exp_max >= -26
    )
    factor_hi = 2.0**-36  # shift clamped up: contribution 2**-36
    factor_lo = 2.0**26  # shift clamped down: contribution 2**26
    # zero-shift tiles (cheap necessary pre-filter from the per-tile
    # exponent envelopes): the frozen reference rounds ODD products UP
    # to the even neighbour at shift == 0 (an exact tie of its half-even
    # shifter, see the frozen numpy reference's _rint_half_even) while
    # round() leaves an integer unchanged -- those tiles pay the
    # correction pass below
    maybe_zero_tile = (input_exp_min + weight_exp_min <= fixed_shift_value) & (
        input_exp_max + weight_exp_max >= fixed_shift_value
    )

    for tile_index in range(num_tiles):
        col_start = 16 * tile_index
        col_end = 16 * tile_index + 16

        # pre-computed fixed points for this tile
        input_fp = input_all_fp[:, col_start:col_end]  # (rows, 16)
        weight_fp = weight_all_fp[:, col_start:col_end]  # (N, 16)

        # 16-element integer MAC using BLAS matmul (exact in float64
        # for these magnitudes), in units of real_value * 2**(58 - Ea - Eb)
        tile_mac = input_fp @ weight_fp.T  # (rows, N)

        # rescale onto the global accumulator grid (LSB = 2**(sv-58)):
        # mac * 2**(Ea + Eb - sv) as an exact power-of-two outer
        # product, half-to-even rounded, then accumulated and
        # saturated (the hardware shifter clamp handled below)
        if bool(clean_tile[tile_index]):
            # clean tile: two exact power-of-two broadcast scalings
            # (row factor then column factor) equal the combined outer
            # product without materializing a [rows, N] factor tensor
            tile_mac.mul_(input_factor[:, tile_index].unsqueeze(1))
            tile_mac.mul_(weight_factor[:, tile_index].unsqueeze(0))
        else:
            # some (m, n) shift of this tile hits the shifter clamp:
            # materialize the corrected outer product where the raw
            # shift went outside [-26, 36]
            factor = input_factor[:, tile_index].unsqueeze(1) * weight_factor[
                :, tile_index
            ].unsqueeze(0)
            raw_shift = (
                fixed_shift_value
                - input_tile_max_all[:, tile_index].unsqueeze(1)
                - weight_tile_max_all[:, tile_index].unsqueeze(0)
            )
            factor = torch.where(raw_shift > 36, factor_hi, factor)
            factor = torch.where(raw_shift < -26, factor_lo, factor)
            tile_mac.mul_(factor)

        # Each tile result is mac scaled by an exact power of two
        # (2**-clamped_shift) and rounded half to even; the fp64 scaling
        # is exact, done in-place to avoid per-tile temporaries.
        tile_mac.round_()
        if bool(maybe_zero_tile[tile_index]):
            # exact probe on the per-tile unique exponent values
            # (<= 31 each): does any (m, n) pair hit shift == 0?
            ea_unique = torch.unique(input_tile_max_all[:, tile_index])
            eb_unique = torch.unique(weight_tile_max_all[:, tile_index])
            if bool(torch.isin(ea_unique, fixed_shift_value - eb_unique).any()):
                # at a zero shift the scaled mac is unchanged (the
                # factor is exactly 1 and round() is the identity on
                # integers), so the frozen reference's carry -- an odd
                # product rounds UP to the even neighbour -- is a plain
                # +1 on the odd positions of the (m, n) zero-shift mask
                zero_mask = (
                    input_tile_max_all[:, tile_index].unsqueeze(1)
                    + weight_tile_max_all[:, tile_index].unsqueeze(0)
                    == fixed_shift_value
                )
                odd = tile_mac.remainder(2) != 0
                tile_mac[zero_mask & odd] += 1
        # saturate to INT32 after every tile (hardware behaviour)
        accumulator.add_(tile_mac)
        accumulator.clamp_(INT32_MIN, INT32_MAX)
    # the bias joins only after the last tile (unlike convt / matmul)
    accumulator.add_(bias_int32.to(torch.float64).unsqueeze(0))
    accumulator.clamp_(INT32_MIN, INT32_MAX)
    return accumulator.to(torch.int64)


def _matmul_f16f16_fp16(input_matrix, weight_matrix, bias, fixed_shift_value=42):
    """F16xF16 matmul returning the dequantized fp16 output.

    Dequantization divides by ``2 ** (58 - fixed_shift_value)``, the
    reciprocal of the accumulator grid LSB.
    """
    int32_accumulator = _matmul_f16f16_int32(
        input_matrix, weight_matrix, bias, fixed_shift_value
    )
    dequantized = int32_accumulator.to(torch.float64) * (
        2.0 ** (-58 + fixed_shift_value)
    )
    return _to_fp16_single_round(dequantized)


def _ceil_divide(dividend, divisor):
    return (dividend + divisor - 1) // divisor


def _conv_im2col(
    input_tensor,
    kernel_h,
    kernel_w,
    stride_h,
    stride_w,
    dilation_h,
    dilation_w,
    pad_top,
    pad_bottom,
    pad_left,
    pad_right,
    out_h,
    out_w,
    num_channel_blocks,
    channel_block_size=16,
):
    """Expand one group's input into the im2col matrix ``[out_h*out_w, K]``.

    The K axis is laid out as (channel_block, kh, kw, 16 channels), so
    every 16-element Cube tile covers the 16 channels of one block at a
    single kernel tap. The channel axis is zero padded up to a multiple
    of 16 so that no tile is partially populated.
    """
    in_channels, in_h, in_w = input_tensor.shape
    aligned_channels = num_channel_blocks * channel_block_size
    padded_input = torch.zeros(
        aligned_channels,
        in_h + pad_top + pad_bottom,
        in_w + pad_left + pad_right,
        dtype=input_tensor.dtype,
        device=input_tensor.device,
    )
    padded_input[:in_channels, pad_top : pad_top + in_h, pad_left : pad_left + in_w] = (
        input_tensor
    )
    # [block, 16, H', W'] -> [block, H', W', 16]: channels innermost
    padded_input = padded_input.reshape(
        num_channel_blocks,
        channel_block_size,
        padded_input.shape[1],
        padded_input.shape[2],
    ).permute(0, 2, 3, 1)

    row_indices = (
        torch.arange(kernel_h, device=input_tensor.device) * dilation_h
    ).unsqueeze(1) + torch.arange(out_h, device=input_tensor.device).unsqueeze(
        0
    ) * stride_h
    column_indices = (
        torch.arange(kernel_w, device=input_tensor.device) * dilation_w
    ).unsqueeze(1) + torch.arange(out_w, device=input_tensor.device).unsqueeze(
        0
    ) * stride_w

    # gather the dilated kernel taps: [block, kh, out_h, kw, out_w, 16]
    gathered_rows = padded_input[:, row_indices]
    patches = gathered_rows[:, :, :, column_indices]
    # [out_h, out_w, block, kh, kw, 16]: rows are output positions,
    # columns run (channel_block, kh, kw, 16 channels)
    patches = patches.permute(2, 4, 0, 1, 3, 5)
    return patches.reshape(out_h * out_w, -1)


def _conv_weight_to_matrix(
    weight,
    in_channels,
    kernel_h,
    kernel_w,
    num_channel_blocks,
    channel_block_size=16,
):
    """Reshape one group's weight into the ``[out_c, K]`` matrix.

    The K axis follows the same (channel_block, kh, kw, 16 channels)
    order as ``_conv_im2col`` so that input tiles and weight tiles pair
    up element-wise.
    """
    out_channels = weight.shape[0]
    padded_weight = torch.zeros(
        out_channels,
        num_channel_blocks * channel_block_size,
        kernel_h,
        kernel_w,
        dtype=weight.dtype,
        device=weight.device,
    )
    padded_weight[:, :in_channels] = weight
    # [out_c, block, 16, kh, kw] -> [out_c, block, kh, kw, 16]
    padded_weight = padded_weight.reshape(
        out_channels, num_channel_blocks, channel_block_size, kernel_h, kernel_w
    ).permute(0, 1, 3, 4, 2)
    return padded_weight.reshape(out_channels, -1)


def conv_f16f16(
    x,
    w,
    bias,
    pads,
    strides,
    dilations,
    groups,
    fixed_shift_value=42,
):
    """fp16fixed F16xF16 conv2d, functional entry (legacy-compatible signature).

    ``pads`` is (top, bottom, left, right); ``strides`` / ``dilations``
    are (h, w) pairs; the output layout matches
    ``torch.nn.functional.conv2d``.
    """
    batch_size, in_channels, in_h, in_w = x.shape
    out_channels, _, kernel_h, kernel_w = w.shape
    pad_top, pad_bottom, pad_left, pad_right = pads
    stride_h, stride_w = strides
    dilation_h, dilation_w = dilations

    dilated_kernel_h = (kernel_h - 1) * dilation_h + 1
    dilated_kernel_w = (kernel_w - 1) * dilation_w + 1
    out_h = (in_h + pad_top + pad_bottom - dilated_kernel_h) // stride_h + 1
    out_w = (in_w + pad_left + pad_right - dilated_kernel_w) // stride_w + 1

    in_channels_per_group = in_channels // groups
    out_channels_per_group = out_channels // groups
    channel_block_size = 16
    num_channel_blocks = _ceil_divide(in_channels_per_group, channel_block_size)

    if bias is None:
        bias = torch.zeros(out_channels, dtype=torch.float16, device=x.device)
    bias_f32 = bias.to(torch.float32)

    out = torch.zeros(
        batch_size,
        out_channels,
        out_h,
        out_w,
        dtype=torch.float16,
        device=x.device,
    )
    # groups are reduced independently; every (batch, group) pair becomes
    # one im2col matmul written back into its output channel slice
    for batch_idx in range(batch_size):
        for group_idx in range(groups):
            input_group = x[
                batch_idx,
                group_idx * in_channels_per_group : (group_idx + 1)
                * in_channels_per_group,
            ]
            input_columns = _conv_im2col(
                input_group,
                kernel_h,
                kernel_w,
                stride_h,
                stride_w,
                dilation_h,
                dilation_w,
                pad_top,
                pad_bottom,
                pad_left,
                pad_right,
                out_h,
                out_w,
                num_channel_blocks,
                channel_block_size,
            )
            weight_group = w[
                group_idx * out_channels_per_group : (group_idx + 1)
                * out_channels_per_group
            ]
            weight_matrix = _conv_weight_to_matrix(
                weight_group,
                in_channels_per_group,
                kernel_h,
                kernel_w,
                num_channel_blocks,
                channel_block_size,
            )
            bias_group = bias_f32[
                group_idx * out_channels_per_group : (group_idx + 1)
                * out_channels_per_group
            ]
            matmul_result = _matmul_f16f16_fp16(
                input_columns, weight_matrix, bias_group, fixed_shift_value
            )
            # the matmul result is [out_h*out_w, out_c]; transpose back
            # to channel major
            out[
                batch_idx,
                group_idx * out_channels_per_group : (group_idx + 1)
                * out_channels_per_group,
            ] = matmul_result.T.reshape(out_channels_per_group, out_h, out_w)

    return out


class Fp16FixedConv2d(ShiftValueSearchMixin, torch.nn.Conv2d):
    """Conv2d running the fp16fixed F16xF16 fixed point simulation.

    All Conv2d attributes (weight / bias / stride / padding / dilation /
    groups) are inherited from ``torch.nn.Conv2d``.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        fixed_shift_value=FIXED_SHIFT_VALUE_FP16_FP16_DEFAULT,
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.fixed_shift_value = int(fixed_shift_value)
        self._sse_sum = None

    @classmethod
    def from_module(cls, module, fixed_shift_value=FIXED_SHIFT_VALUE_FP16_FP16_DEFAULT):
        """Build from an existing Conv2d module (state dict copied)."""
        new_module = cls(
            module.in_channels,
            module.out_channels,
            module.kernel_size,
            stride=module.stride,
            padding=module.padding,
            dilation=module.dilation,
            groups=module.groups,
            bias=module.bias is not None,
            fixed_shift_value=fixed_shift_value,
        )
        new_module.load_state_dict(module.state_dict())
        # keep the fp16fixed module on the same device as the source module
        new_module = new_module.to(module.weight.device)
        return new_module

    def _pads(self):
        padding = self.padding
        if isinstance(padding, int):
            padding = (padding, padding)
        return (padding[0], padding[0], padding[1], padding[1])

    def _forward_impl(self, x, fixed_shift_value):
        x = x.to(torch.float16)
        return conv_f16f16(
            x,
            self.weight.to(torch.float16),
            self.bias,
            self._pads(),
            tuple(self.stride),
            tuple(self.dilation),
            self.groups,
            fixed_shift_value,
        )

    def forward(self, x):
        return self._forward_impl(x, self.fixed_shift_value)

    def _reference_output(self, x):
        return torch.nn.functional.conv2d(
            x.to(torch.float32),
            self.weight.to(torch.float32),
            bias=self.bias.to(torch.float32) if self.bias is not None else None,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )


def to_fixed_point_source_with_conv_clamp(
    exponent_raw, mantissa_raw, sign_bit, is_normal, tile_max_exponent
):
    """conv2d engine's alignment: clamp [-11, 5]."""
    return to_fixed_point_source(
        exponent_raw,
        mantissa_raw,
        sign_bit,
        is_normal,
        tile_max_exponent,
        -11,
        5,
    )
