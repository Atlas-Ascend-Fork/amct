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
"""fp16fixed ConvTranspose2d class implementation.

``Fp16FixedConvTranspose2d`` subclasses ``torch.nn.ConvTranspose2d`` and runs
the fp16fixed F16xF16 fixed point Cube Unit simulation. The numerical path is
identical to the frozen numpy reference of this engine.

A conv transpose is evaluated as a regular conv: the input is upsampled
by inserting (stride - 1) zeros between elements (``_upsample_pad``),
the kernel is spatially flipped, and the matmul reduces over
(kh, kw, in_channels) with the convt output channels as rows.

Numerical deviations from ``conv2d.py`` (kept on purpose, they
model a different hardware path): per-element exponent clamp
``[-16, 15]``, tile results scaled in float64 via ``exp2`` instead of
the integer half-even shift, and the bias added to the accumulator
before the first tile (not after the last one).

Naming convention of the fixed point pipeline: see ``conv2d.py``.
"""

import torch
import torch.nn.functional as F

from ....utils.vars import FIXED_SHIFT_VALUE_FP16_FP16_DEFAULT
from .common import saturate_to_int32 as _saturate_to_int32
from .common import to_fixed_point_source
from .shift_value_search import ShiftValueSearchMixin

INT32_MIN, INT32_MAX = -(2**31), 2**31 - 1


def _matmul_f16f16_with_accumulator(
    input_matrix, weight_matrix, fixed_shift_value, initial_accumulator
):
    """F16xF16 matmul accumulating onto ``initial_accumulator`` (INT32).

    The caller seeds the accumulator with the pre-scaled bias; every
    tile result is folded in with a float64 round and the accumulator
    is saturated to INT32 after each tile.
    """
    num_rows, reduction_size = input_matrix.shape
    if reduction_size % 16:
        pad_size = 16 - reduction_size % 16
        input_matrix = F.pad(input_matrix, (0, pad_size))
        weight_matrix = F.pad(weight_matrix, (0, pad_size))
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

    # the accumulator starts from the caller's seed (the pre-added bias)
    accumulator = initial_accumulator.to(torch.int64)
    # one Cube instruction processes 16 consecutive K elements of both
    # operands (K was zero padded to a multiple of 16 above)
    for tile_index in range(reduction_size // 16):
        col_start = 16 * tile_index
        col_end = 16 * tile_index + 16

        # per-tile alignment anchors: the max raw exponent in the tile
        input_tile_max_exponent = input_exponent[:, col_start:col_end].amax(-1)
        weight_tile_max_exponent = weight_exponent[:, col_start:col_end].amax(-1)
        input_fixed_point = to_fixed_point_source_with_convt_clamp(
            input_exponent[:, col_start:col_end],
            input_mantissa[:, col_start:col_end],
            input_sign[:, col_start:col_end],
            input_is_normal[:, col_start:col_end],
            input_tile_max_exponent,
        )
        weight_fixed_point = to_fixed_point_source_with_convt_clamp(
            weight_exponent[:, col_start:col_end],
            weight_mantissa[:, col_start:col_end],
            weight_sign[:, col_start:col_end],
            weight_is_normal[:, col_start:col_end],
            weight_tile_max_exponent,
        )

        # 16-element integer MAC, in units of real_value * 2**(58 - Ea - Eb)
        tile_mac = (
            input_fixed_point.unsqueeze(1) * weight_fixed_point.unsqueeze(0)
        ).sum(-1)
        # rescale onto the global accumulator grid (LSB = 2**(sv-58)):
        # tile_mac * 2**(Ea + Eb - sv); the clamp is the hardware
        # shifter range
        tile_shift = torch.clamp(
            fixed_shift_value
            - input_tile_max_exponent.unsqueeze(1)
            - weight_tile_max_exponent.unsqueeze(0),
            -26,
            36,
        )

        # float64 exp2 scaling replaces conv2d's integer half-even shift
        tile_result = tile_mac.to(torch.float64) * torch.exp2(
            (-tile_shift).to(torch.float64)
        )
        # round + saturate to INT32 after every tile
        accumulator = torch.round(accumulator.to(torch.float64) + tile_result)
        accumulator = torch.clamp(accumulator, INT32_MIN, INT32_MAX).to(torch.int64)
    return accumulator


def _upsample_pad(
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
    output_padding_h,
    output_padding_w,
):
    """Insert (stride - 1) zeros between input rows/columns and crop pads.

    First step of the conv-transpose-as-conv equivalence: conv_transpose
    (stride s, pad p, kernel k) == conv over the input with (s - 1)
    zeros embedded between elements, using the spatially flipped kernel.
    The embedded data starts at ``top_offset`` / ``left_offset`` (derived
    from the kernel extent and the pads) so that windows sliding over
    the canvas reproduce the conv transpose padding semantics.
    Returns the strided-embedded input the conv transpose is computed from.
    """
    in_channels, in_h, in_w = input_tensor.shape
    upsampled_h = (in_h - 1) * stride_h + 1
    upsampled_w = (in_w - 1) * stride_w + 1
    dilated_kernel_h = (kernel_h - 1) * dilation_h + 1
    dilated_kernel_w = (kernel_w - 1) * dilation_w + 1
    # over padding (pad larger than the dilated kernel extent - 1) breaks
    # the conv-transpose-as-conv canvas layout; the amct_onnx kernel
    # guards the same condition in CheckConstraints. Raise the
    # NotImplementedError the calibration wrapper catches, so the layer
    # falls back to the default fixed_shift_value instead of crashing
    # the calibration flow.
    if (
        pad_top > dilated_kernel_h - 1
        or pad_bottom > dilated_kernel_h - 1
        or pad_left > dilated_kernel_w - 1
        or pad_right > dilated_kernel_w - 1
    ):
        raise NotImplementedError(
            'conv transpose padding {} exceeds the dilated kernel extent '
            '- 1 ({}, {}): the conv-transpose-as-conv equivalence does '
            'not cover over padding, fixed_shift_value falls back to the '
            'regime default value.'.format(
                (pad_top, pad_bottom, pad_left, pad_right),
                dilated_kernel_h - 1,
                dilated_kernel_w - 1,
            )
        )
    # the input data is embedded at these offsets inside the padded canvas
    top_offset = dilated_kernel_h - 1 - pad_top
    bottom_offset = dilated_kernel_h - 1 - pad_bottom
    left_offset = dilated_kernel_w - 1 - pad_left
    right_offset = dilated_kernel_w - 1 - pad_right
    padded_h = upsampled_h + top_offset + bottom_offset + output_padding_h
    padded_w = upsampled_w + left_offset + right_offset + output_padding_w
    upsampled_input = torch.zeros(
        in_channels,
        padded_h,
        padded_w,
        dtype=input_tensor.dtype,
        device=input_tensor.device,
    )
    upsampled_input[
        :,
        top_offset : top_offset + upsampled_h : stride_h,
        left_offset : left_offset + upsampled_w : stride_w,
    ] = input_tensor
    return upsampled_input


def _im2col(upsampled_input, kernel_h, kernel_w, dilation_h, dilation_w, out_h, out_w):
    """Gather the dilated kernel taps into ``[out_h * out_w, K]`` columns.

    Unlike ``conv2d``, the K axis order is (kh, kw, cin): channels
    are not split into 16-channel blocks here; they are aligned to a
    multiple of 16 by the caller's ``F.pad`` so every 16-element tile
    still spans 16 consecutive channels at one tap.
    """
    row_indices = (
        torch.arange(kernel_h, device=upsampled_input.device) * dilation_h
    ).unsqueeze(1) + torch.arange(out_h, device=upsampled_input.device).unsqueeze(0)
    column_indices = (
        torch.arange(kernel_w, device=upsampled_input.device) * dilation_w
    ).unsqueeze(1) + torch.arange(out_w, device=upsampled_input.device).unsqueeze(0)

    # gather the dilated kernel taps: [cin, kh, out_h, kw, out_w]
    gathered_rows = upsampled_input[:, row_indices]
    patches = gathered_rows[:, :, :, column_indices]
    # Patches are laid out as [out_h, out_w, kh, kw, cin]: rows are output
    # positions, columns iterate the kernel window in the order
    # kh, then kw, then cin.
    patches = patches.permute(2, 4, 1, 3, 0)
    return patches.reshape(out_h * out_w, -1)


def _convt_weight_to_matrix(weight_group):
    """Flip one group's weight and reshape into the ``[cout_g, K]`` matrix.

    The weight group ``[cin, cout_g, kh, kw]`` is spatially flipped (the
    conv-transpose-as-conv equivalence) and laid out with the convt
    output channels as rows; the K axis follows the (kh, kw, cin) order
    of ``_im2col`` so the tiles pair up.
    """
    flipped_weight = torch.flip(weight_group, [2, 3])
    return flipped_weight.permute(1, 2, 3, 0).reshape(weight_group.shape[1], -1)


def convt_f16f16_int32(
    x,
    w,
    bias,
    pads,
    strides,
    dilations,
    groups,
    output_shape,
    fixed_shift_value=42,
):
    """fp16fixed F16xF16 conv transpose2d, returns the INT32 accumulator.

    ``pads`` is (top, bottom, left, right); ``strides`` / ``dilations``
    are (h, w) pairs; ``output_shape`` is the expected NCHW output.
    """
    batch_size, in_channels, in_h, in_w = x.shape
    in_channels_w, out_channels_per_group, kernel_h, kernel_w = w.shape
    out_channels = out_channels_per_group * groups
    _, _, out_h, out_w = output_shape
    pad_top, pad_bottom, pad_left, pad_right = pads
    stride_h, stride_w = strides
    dilation_h, dilation_w = dilations

    # output_padding = the extra extent the caller asked for beyond the
    # natural conv transpose output
    output_padding_h = max(
        0,
        out_h
        - (
            (in_h - 1) * stride_h
            + (kernel_h - 1) * dilation_h
            + 1
            - pad_top
            - pad_bottom
        ),
    )
    output_padding_w = max(
        0,
        out_w
        - (
            (in_w - 1) * stride_w
            + (kernel_w - 1) * dilation_w
            + 1
            - pad_left
            - pad_right
        ),
    )

    in_channels_per_group = in_channels // groups
    aligned_in_channels_per_group = ((in_channels_per_group + 15) // 16) * 16

    result = torch.zeros(
        batch_size,
        out_channels,
        out_h,
        out_w,
        dtype=torch.int32,
        device=x.device,
    )
    if bias is not None:
        # Convt semantics: the scaled bias, whose LSB is 2**(sv-58) of
        # the real value, is broadcast into the result before the tile
        # loop, unlike conv2d which applies it after the last tile.
        bias_f32 = bias.to(torch.float32)
        bias_int32 = _saturate_to_int32(
            torch.round(bias_f32 * float(1 << (58 - fixed_shift_value)))
        )
        result = result + bias_int32.view(1, out_channels, 1, 1)

    # groups are reduced independently; every (batch, group) pair becomes
    # one im2col matmul written back into its output channel slice
    for batch_idx in range(batch_size):
        for group_idx in range(groups):
            group_start_channel = group_idx * in_channels_per_group
            input_group = x[
                batch_idx,
                group_start_channel : group_start_channel + in_channels_per_group,
            ]
            weight_group = w[
                group_start_channel : group_start_channel + in_channels_per_group
            ]
            if in_channels_per_group < aligned_in_channels_per_group:
                # zero pad the channel axis of both operands up to the
                # multiple of 16 (pad pairs run from the last dim backwards)
                channel_pad = aligned_in_channels_per_group - in_channels_per_group
                input_group = F.pad(input_group, (0, 0, 0, 0, 0, channel_pad))
                weight_group = F.pad(weight_group, (0, 0, 0, 0, 0, 0, 0, channel_pad))
            upsampled_input = _upsample_pad(
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
                output_padding_h,
                output_padding_w,
            )
            input_columns = _im2col(
                upsampled_input,
                kernel_h,
                kernel_w,
                dilation_h,
                dilation_w,
                out_h,
                out_w,
            )
            weight_matrix = _convt_weight_to_matrix(weight_group)
            # seed the matmul with the current result so the pre-added
            # bias rides along through the tile loop
            initial_accumulator = (
                result[
                    batch_idx,
                    group_idx * out_channels_per_group : (group_idx + 1)
                    * out_channels_per_group,
                ]
                .permute(1, 2, 0)
                .reshape(out_h * out_w, out_channels_per_group)
            )
            matmul_result = _matmul_f16f16_with_accumulator(
                input_columns,
                weight_matrix,
                fixed_shift_value,
                initial_accumulator,
            )
            # [out_h*out_w, cout_g] -> [cout_g, out_h, out_w]
            matmul_result = matmul_result.T.reshape(
                out_channels_per_group, out_h, out_w
            )
            result[
                batch_idx,
                group_idx * out_channels_per_group : (group_idx + 1)
                * out_channels_per_group,
            ] = matmul_result.to(torch.int32)
    result = torch.clamp(result, INT32_MIN, INT32_MAX)
    return result


def convt_f16f16(
    x,
    w,
    bias,
    pads,
    strides,
    dilations,
    groups,
    output_shape,
    fixed_shift_value=42,
):
    """fp16fixed F16xF16 conv transpose2d, functional entry (legacy compatible)."""
    int32_result = convt_f16f16_int32(
        x,
        w,
        bias,
        pads,
        strides,
        dilations,
        groups,
        output_shape,
        fixed_shift_value,
    )
    # back to real value: divide by 2**(58 - sv), the accumulator grid
    out = (int32_result.to(torch.float32) / float(1 << (58 - fixed_shift_value))).to(
        torch.float16
    )
    return out


class Fp16FixedConvTranspose2d(ShiftValueSearchMixin, torch.nn.ConvTranspose2d):
    """ConvTranspose2d running the fp16fixed F16xF16 fixed point simulation."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        output_padding=0,
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
            output_padding=output_padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.fixed_shift_value = int(fixed_shift_value)
        self._sse_sum = None

    @classmethod
    def from_module(cls, module, fixed_shift_value=FIXED_SHIFT_VALUE_FP16_FP16_DEFAULT):
        """Build from an existing ConvTranspose2d module (state dict copied)."""
        new_module = cls(
            module.in_channels,
            module.out_channels,
            module.kernel_size,
            stride=module.stride,
            padding=module.padding,
            output_padding=module.output_padding,
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

    def _output_shape(self, x):
        batch_size, _, in_h, in_w = x.shape
        out_h = (
            (in_h - 1) * self.stride[0]
            - 2 * self.padding[0]
            + self.dilation[0] * (self.kernel_size[0] - 1)
            + self.output_padding[0]
            + 1
        )
        out_w = (
            (in_w - 1) * self.stride[1]
            - 2 * self.padding[1]
            + self.dilation[1] * (self.kernel_size[1] - 1)
            + self.output_padding[1]
            + 1
        )
        return (batch_size, self.out_channels, out_h, out_w)

    def _forward_impl(self, x, fixed_shift_value):
        x = x.to(torch.float16)
        return convt_f16f16(
            x,
            self.weight.to(torch.float16),
            self.bias,
            self._pads(),
            tuple(self.stride),
            tuple(self.dilation),
            self.groups,
            self._output_shape(x),
            fixed_shift_value,
        )

    def forward(self, x):
        return self._forward_impl(x, self.fixed_shift_value)

    def _reference_output(self, x):
        return F.conv_transpose2d(
            x.to(torch.float32),
            self.weight.to(torch.float32),
            bias=self.bias.to(torch.float32) if self.bias is not None else None,
            stride=self.stride,
            padding=self.padding,
            output_padding=self.output_padding,
            dilation=self.dilation,
            groups=self.groups,
        )


def to_fixed_point_source_with_convt_clamp(
    exponent_raw, mantissa_raw, sign_bit, is_normal, tile_max_exponent
):
    """convtranspose engine's alignment: clamp [-16, 15]."""
    return to_fixed_point_source(
        exponent_raw,
        mantissa_raw,
        sign_bit,
        is_normal,
        tile_max_exponent,
        -16,
        15,
    )
