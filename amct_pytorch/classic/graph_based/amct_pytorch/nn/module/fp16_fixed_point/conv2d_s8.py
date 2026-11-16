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
"""fp16fixed F16xS8 Conv2d reference (weight-only regime).

Refactored from the frozen F16xS8 conv reference (bit-exact,
parity-pinned contract): a fp16 activation meets an int8
weight, the regime of the weight-only shift value path (candidates
{5, ..., 14}). ``Fp16FixedConv2dS8`` wraps the functional entry for the
``ShiftValueSearchMixin`` based per-layer shift value search.

Pipeline of one F16xS8 Cube tile (engine: ``matmul.f16_matmul_s8``,
shared with the matmul reference):

1. only the fp16 activation is reinterpreted and decoded into sign /
   raw exponent / raw mantissa; the int8 weight joins the MAC as-is and
   takes part in no exponent search;
2. within a 16-element K tile every activation element is aligned onto
   the tile max raw exponent and converted to a fixed point mantissa
   (grid ``value = fixed_point * 2 ** (tile_max_exponent - 29)``,
   alignment clamp ``[-16, 15]``);
3. the 16-element integer MAC is rescaled by ``tile_shift =
   fixed_shift_value - input_tile_max_exponent`` -- no clamp, the S8
   shifter range differs from the F16xF16 one;
4. the accumulator is rounded and saturated to INT32 after every tile.

The accumulator LSB is worth ``weight_scale * 2 ** (sv - 29)`` of real
value, which is exactly the ``antiquant_scale`` argument: it is the
accumulator-LSB real value, so it depends on the shift value and must
be recomputed when ``fixed_shift_value`` changes. ``Fp16FixedConv2dS8``
derives it from the stored weight scale per shift value, the functional
entry takes it as an argument.

Layout and semantics kept from the legacy reference:

- K axis: (channel_block, kh, kw, 16 channels) via the shared
  ``conv2d`` im2col helpers, the channel axis zero padded up to a
  multiple of 16 so no tile is partially populated;
- the pre-scaled bias is added after the last tile (one clamp), unlike
  conv transpose's pre-loaded accumulator;
- a zero ``antiquant_scale`` (dead output channel) is treated as one
  when scaling the bias.

``x`` must be fp16 and ``w`` int8; the dequantized fp16 output layout
matches ``torch.nn.functional.conv2d``.
"""

import torch
import torch.nn.functional as F

from ....utils.vars import FIXED_SHIFT_VALUE_FP16_INT8_CANDIDATES
from ....utils.vars import FIXED_SHIFT_VALUE_FP16_INT8_DEFAULT
from .conv2d import _ceil_divide
from .conv2d import _conv_im2col
from .conv2d import _conv_weight_to_matrix
from .matmul import _check_int8_weight
from .matmul import _normalize_ascale
from .matmul import _s8_bias_to_int32
from .matmul import f16_matmul_s8
from .matmul import int32_dequant_to_fp16_s8
from .shift_value_search import ShiftValueSearchMixin

INT32_MIN, INT32_MAX = -(2**31), 2**31 - 1
CHANNEL_BLOCK_SIZE = 16


def conv_f16s8(
    x,
    w,
    bias,
    antiquant_scale,
    pads,
    strides,
    dilations,
    groups,
    fixed_shift_value=FIXED_SHIFT_VALUE_FP16_INT8_DEFAULT,
):
    """fp16fixed F16xS8 conv2d, functional entry (legacy-compatible signature).

    ``x`` is the fp16 activation ``[N, Cin, H, W]``, ``w`` the int8
    weight ``[Cout, Cin // groups, kh, kw]``, ``antiquant_scale`` the
    accumulator-LSB real value (see the module docstring); ``pads`` is
    (top, bottom, left, right), ``strides`` / ``dilations`` are (h, w)
    pairs. Returns the fp16 output, layout as ``F.conv2d``.
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

    ascale = _normalize_ascale(antiquant_scale, out_channels)

    in_channels_per_group = in_channels // groups
    out_channels_per_group = out_channels // groups
    num_channel_blocks = _ceil_divide(in_channels_per_group, CHANNEL_BLOCK_SIZE)

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
            in_start = group_idx * in_channels_per_group
            input_group = x[batch_idx, in_start : in_start + in_channels_per_group]
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
                CHANNEL_BLOCK_SIZE,
            )
            out_start = group_idx * out_channels_per_group
            weight_group = w[out_start : out_start + out_channels_per_group]
            weight_matrix = _conv_weight_to_matrix(
                weight_group,
                in_channels_per_group,
                kernel_h,
                kernel_w,
                num_channel_blocks,
                CHANNEL_BLOCK_SIZE,
            )
            ascale_group = ascale[out_start : out_start + out_channels_per_group]
            # the engine runs without bias: conv adds the pre-scaled
            # bias after the last tile (legacy semantics)
            accumulator = f16_matmul_s8(
                input_columns, weight_matrix, None, fixed_shift_value
            )
            if bias is not None:
                bias_group_int = _s8_bias_to_int32(
                    bias[out_start : out_start + out_channels_per_group],
                    ascale_group,
                )
                accumulator = torch.clamp(
                    accumulator.to(torch.int64) + bias_group_int.unsqueeze(0),
                    INT32_MIN,
                    INT32_MAX,
                ).to(torch.int32)
            dequantized = int32_dequant_to_fp16_s8(accumulator, ascale_group)
            # [out_h * out_w, cout_g] -> [cout_g, out_h, out_w]
            out[batch_idx, out_start : out_start + out_channels_per_group] = (
                dequantized.T.reshape(out_channels_per_group, out_h, out_w)
            )

    return out


class Fp16FixedConv2dS8(ShiftValueSearchMixin, torch.nn.Conv2d):
    """Conv2d running the fp16fixed F16xS8 (weight-only) fixed point simulation.

    The class owns the antiquant scale derivation: it stores the
    symmetric weight scale and derives the accumulator-LSB real value
    ``weight_scale * 2 ** (sv - 29)`` per shift value. The search
    primitives (``ShiftValueSearchMixin``) compare against the fp32 conv
    run with the dequantized int8 weight, bias included on both sides.
    """

    # the S8 regime searches over {5, ..., 14}, not the F16xF16 range
    candidates = list(FIXED_SHIFT_VALUE_FP16_INT8_CANDIDATES)

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
        fixed_shift_value=FIXED_SHIFT_VALUE_FP16_INT8_DEFAULT,
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
        """Build from an existing Conv2d module plus its deployed int8 weight.

        Args:
            module: the wrapped ``torch.nn.Conv2d`` (geometry and bias
                are copied).
            int8_weight: the deployed int8 weight with the Conv2d layout
                ``[Cout, Cin // groups, kh, kw]``.
            weight_scale: a scalar tensor or one float entry per output
                channel (symmetric weight quantization scale).
        """
        _check_int8_weight(int8_weight, module.weight)
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

    def _forward_impl(self, x, fixed_shift_value):
        # the functional entry scales the bias into the accumulator
        # domain itself (with the zero-scale guard)
        return conv_f16s8(
            x.to(torch.float16),
            self.int8_weight,
            self.bias,
            self._accumulator_scale(fixed_shift_value),
            self._pads(),
            tuple(self.stride),
            tuple(self.dilation),
            self.groups,
            fixed_shift_value,
        )

    def forward(self, x):
        return self._forward_impl(x, self.fixed_shift_value)

    def _reference_output(self, x):
        dequant_weight = self.int8_weight.to(torch.float32) * (
            self.weight_scale.reshape(-1, 1, 1, 1)
        )
        return F.conv2d(
            x.to(torch.float32),
            dequant_weight,
            bias=self.bias.to(torch.float32) if self.bias is not None else None,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )
