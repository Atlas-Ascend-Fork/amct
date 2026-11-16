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
"""fp16fixed F16xS8 ConvTranspose2d reference (weight-only regime).

Refactored from the frozen F16xS8 conv transpose reference (bit-exact,
parity-pinned contract): a fp16 activation meets an int8
weight, the regime of the weight-only shift value path (candidates
{5, ..., 14}). ``Fp16FixedConvTranspose2dS8`` wraps the functional entry for
the ``ShiftValueSearchMixin`` based per-layer shift value search.

A conv transpose is evaluated as a regular conv (same equivalence as the
F16xF16 family): the input is upsampled by inserting (stride - 1) zeros
between elements, the kernel is spatially flipped, and the matmul
reduces over (kh, kw, in_channels) with the convt output channels as
rows. The K axis channels are zero padded to a multiple of 16 so every
16-element Cube tile spans 16 consecutive channels at one tap.

The accumulator LSB is worth ``weight_scale * 2 ** (sv - 29)`` of real
value, which is exactly the ``antiquant_scale`` argument: it depends on
the shift value. ``Fp16FixedConvTranspose2dS8`` derives it from the stored
weight scale per shift value, the functional entry takes it as an
argument.

Semantics kept from the legacy reference:

- the pre-scaled bias seeds the accumulator before the first tile (the
  engine's accumulator init), unlike conv's after-the-last-tile add;
- a zero ``antiquant_scale`` (dead output channel) is treated as one
  when scaling the bias.

``x`` must be fp16 and ``w`` int8; the dequantized fp16 output layout
matches ``torch.nn.functional.conv_transpose2d``.
"""

import torch
import torch.nn.functional as F

from ....utils.vars import FIXED_SHIFT_VALUE_FP16_INT8_CANDIDATES
from ....utils.vars import FIXED_SHIFT_VALUE_FP16_INT8_DEFAULT
from .conv2d import _ceil_divide
from .convtranspose2d import _convt_weight_to_matrix
from .convtranspose2d import _im2col as _convt_im2col
from .convtranspose2d import _upsample_pad
from .matmul import _check_int8_weight
from .matmul import _normalize_ascale
from .matmul import _s8_bias_to_int32
from .matmul import f16_matmul_s8
from .matmul import int32_dequant_to_fp16_s8
from .shift_value_search import ShiftValueSearchMixin

CHANNEL_BLOCK_SIZE = 16


def convt_f16s8(
    x,
    w,
    bias,
    antiquant_scale,
    pads,
    strides,
    dilations,
    groups,
    output_shape,
    fixed_shift_value=FIXED_SHIFT_VALUE_FP16_INT8_DEFAULT,
):
    """fp16fixed F16xS8 conv transpose2d, functional entry (legacy-compatible).

    ``w`` is the int8 weight ``[Cin, Cout // groups, kh, kw]``,
    ``output_shape`` the expected NCHW output. ``antiquant_scale`` is
    the accumulator-LSB real value ``weight_scale * 2 ** (sv - 29)``;
    ``pads`` is (top, bottom, left, right), ``strides`` /
    ``dilations`` are (h, w) pairs. Returns the fp16 output, layout as
    ``F.conv_transpose2d``.
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

    ascale = _normalize_ascale(antiquant_scale, out_channels)
    in_channels_per_group = in_channels // groups
    aligned_in_channels = (
        _ceil_divide(in_channels_per_group, CHANNEL_BLOCK_SIZE) * CHANNEL_BLOCK_SIZE
    )

    out = torch.zeros(
        batch_size,
        out_channels,
        out_h,
        out_w,
        dtype=torch.float16,
        device=x.device,
    )
    for batch_idx in range(batch_size):
        for group_idx in range(groups):
            group_start = group_idx * in_channels_per_group
            input_group = x[
                batch_idx, group_start : group_start + in_channels_per_group
            ]
            weight_group = w[group_start : group_start + in_channels_per_group]
            if in_channels_per_group < aligned_in_channels:
                # zero pad the channel axis of both operands up to the
                # multiple of 16 (pad pairs run from the last dim backwards)
                channel_pad = aligned_in_channels - in_channels_per_group
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
            input_columns = _convt_im2col(
                upsampled_input,
                kernel_h,
                kernel_w,
                dilation_h,
                dilation_w,
                out_h,
                out_w,
            )
            weight_matrix = _convt_weight_to_matrix(weight_group)
            out_start = group_idx * out_channels_per_group
            ascale_group = ascale[out_start : out_start + out_channels_per_group]
            # convt semantics: the pre-scaled bias seeds the accumulator
            # before the first tile (passed pre-scaled, ascale=None)
            bias_group_int = (
                _s8_bias_to_int32(
                    bias[out_start : out_start + out_channels_per_group],
                    ascale_group,
                )
                if bias is not None
                else None
            )
            accumulator = f16_matmul_s8(
                input_columns,
                weight_matrix,
                bias_group_int,
                fixed_shift_value,
                None,
            )
            dequantized = int32_dequant_to_fp16_s8(accumulator, ascale_group)
            out[batch_idx, out_start : out_start + out_channels_per_group] = (
                dequantized.T.reshape(out_channels_per_group, out_h, out_w)
            )

    return out


class Fp16FixedConvTranspose2dS8(ShiftValueSearchMixin, torch.nn.ConvTranspose2d):
    """ConvTranspose2d running the fp16fixed F16xS8 fixed point simulation.

    The class owns the antiquant scale derivation: it stores the
    symmetric weight scale and derives the accumulator-LSB real value
    ``weight_scale * 2 ** (sv - 29)`` per shift value. The search
    primitives (``ShiftValueSearchMixin``) compare against the fp32 conv
    transpose run with the dequantized int8 weight, bias included on
    both sides.
    """

    candidates = list(FIXED_SHIFT_VALUE_FP16_INT8_CANDIDATES)

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
        fixed_shift_value=FIXED_SHIFT_VALUE_FP16_INT8_DEFAULT,
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
        # registered as buffers so .to(device) / .cuda() moves them
        # together with the module parameters
        self.register_buffer('int8_weight', None)
        self.register_buffer('weight_scale', None)
        self._sse_sum = None

    @classmethod
    def from_module(
        cls,
        module,
        int8_weight,
        weight_scale,
        fixed_shift_value=FIXED_SHIFT_VALUE_FP16_INT8_DEFAULT,
    ):
        """Build from an existing ConvTranspose2d module plus its int8 weight.

        Args:
            module: the wrapped ``torch.nn.ConvTranspose2d``.
            int8_weight: the deployed int8 weight with the torch layout
                ``[Cin, Cout // groups, kh, kw]``.
            weight_scale: a scalar tensor or one float entry per output
                channel (``Cout`` entries in total).
        """
        _check_int8_weight(int8_weight, module.weight)
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
        new_module.int8_weight = int8_weight
        new_module.weight_scale = _normalize_ascale(
            torch.as_tensor(weight_scale, dtype=torch.float32),
            module.out_channels,
        )
        # keep on the source module device
        new_module = new_module.to(module.weight.device)
        return new_module

    def _accumulator_scale(self, fixed_shift_value):
        """Accumulator-LSB real value: ``weight_scale * 2 ** (sv - 29)``."""
        return self.weight_scale * float(2.0 ** (fixed_shift_value - 29))

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
        return convt_f16s8(
            x,
            self.int8_weight,
            self.bias,
            self._accumulator_scale(fixed_shift_value),
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
        # the torch weight layout is [Cin, Cout // groups, kh, kw]: the
        # dim-1 columns are shared by every group but each output
        # channel has its own scale, so the scale is expanded per group
        weight = self.int8_weight.to(torch.float32)
        if self.groups > 1:
            in_channels, channels_per_group, kernel_h, kernel_w = weight.shape
            channels_in_group = in_channels // self.groups
            weight = (
                weight.reshape(
                    self.groups,
                    channels_in_group,
                    channels_per_group,
                    kernel_h,
                    kernel_w,
                )
                * self.weight_scale.reshape(self.groups, 1, channels_per_group, 1, 1)
            ).reshape(in_channels, channels_per_group, kernel_h, kernel_w)
        else:
            weight = weight * self.weight_scale.reshape(1, -1, 1, 1)
        return F.conv_transpose2d(
            x.to(torch.float32),
            weight,
            bias=self.bias.to(torch.float32) if self.bias is not None else None,
            stride=self.stride,
            padding=self.padding,
            output_padding=self.output_padding,
            dilation=self.dilation,
            groups=self.groups,
        )
