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
"""fp16fixed Linear class implementation of the F16xS8 (weight-only) regime.

``Fp16FixedLinearS8`` subclasses ``torch.nn.Linear`` and runs the fp16fixed F16xS8
fixed point Cube Unit simulation on top of the validated engine
``matmul.f16_matmul_s8``: a fp16 activation meets the deployed int8
weight, the regime of the weight-only shift value path (candidates
{5, ..., 14}).

The class owns the antiquant scale derivation: it stores the symmetric
weight scale and derives the accumulator-LSB real value
``weight_scale * 2 ** (sv - 29)`` per shift value, so the caller never
has to rescale by hand when the shift value changes (the S8 accumulator
LSB depends on ``sv``, an easy caller mistake of the functional API).

The search primitives (``ShiftValueSearchMixin``) compare against the
fp32 linear run with the dequantized int8 weight -- the bias is part of
both the candidate output and the reference, the same convention as the
F16xF16 classes.
"""

import torch

from ....utils.vars import FIXED_SHIFT_VALUE_FP16_INT8_CANDIDATES
from ....utils.vars import FIXED_SHIFT_VALUE_FP16_INT8_DEFAULT
from .matmul import _check_int8_weight
from .matmul import _normalize_ascale
from .matmul import _s8_bias_to_int32
from .matmul import f16_matmul_s8
from .matmul import int32_dequant_to_fp16_s8
from .shift_value_search import ShiftValueSearchMixin


class Fp16FixedLinearS8(ShiftValueSearchMixin, torch.nn.Linear):
    """Linear running the fp16fixed F16xS8 fixed point simulation."""

    # the S8 regime searches over {5, ..., 14}, not the F16xF16 range
    candidates = list(FIXED_SHIFT_VALUE_FP16_INT8_CANDIDATES)

    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        fixed_shift_value=FIXED_SHIFT_VALUE_FP16_INT8_DEFAULT,
    ):
        super().__init__(in_features, out_features, bias=bias)
        self.fixed_shift_value = int(fixed_shift_value)
        # the deployed int8 weight and its symmetric weight scale, set
        # by from_module; self.weight keeps the fp module weight for
        # reference purposes only
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
        """Build from an existing Linear module plus its deployed int8 weight.

        Args:
            module: the wrapped ``torch.nn.Linear`` (bias is copied).
            int8_weight: the deployed int8 weight ``[out_features,
                in_features]`` (symmetric weight quantization).
            weight_scale: a scalar tensor or one float entry per output
                channel.
        """
        _check_int8_weight(int8_weight, module.weight)
        new_module = cls(
            module.in_features,
            module.out_features,
            bias=module.bias is not None,
            fixed_shift_value=fixed_shift_value,
        )
        new_module.load_state_dict(module.state_dict())
        new_module.int8_weight = int8_weight
        new_module.weight_scale = _normalize_ascale(
            torch.as_tensor(weight_scale, dtype=torch.float32),
            module.out_features,
        )
        # keep on the source module device
        new_module = new_module.to(module.weight.device)
        return new_module

    def _accumulator_scale(self, fixed_shift_value):
        """Accumulator-LSB real value: ``weight_scale * 2 ** (sv - 29)``."""
        return self.weight_scale * float(2.0 ** (fixed_shift_value - 29))

    def _forward_impl(self, x, fixed_shift_value):
        # the bias is pre-scaled into the accumulator domain with the
        # zero-scale guard of the S8 family, then seeds the accumulator
        ascale = self._accumulator_scale(fixed_shift_value)
        bias_int = (
            _s8_bias_to_int32(self.bias, ascale) if self.bias is not None else None
        )
        accumulator = f16_matmul_s8(
            x.to(torch.float16),
            self.int8_weight,
            bias_int,
            fixed_shift_value,
            None,
        )
        return int32_dequant_to_fp16_s8(accumulator, ascale)

    def forward(self, x):
        return self._forward_impl(x, self.fixed_shift_value)

    def _reference_output(self, x):
        dequant_weight = self.int8_weight.to(torch.float32) * (
            self.weight_scale.reshape(-1, 1)
        )
        return torch.nn.functional.linear(
            x.to(torch.float32),
            dequant_weight,
            self.bias.to(torch.float32) if self.bias is not None else None,
        )
