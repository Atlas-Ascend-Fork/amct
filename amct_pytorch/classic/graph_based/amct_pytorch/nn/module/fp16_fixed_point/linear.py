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
"""fp16fixed Linear class implementation.

``Fp16FixedLinear`` subclasses ``torch.nn.Linear`` and runs the fp16fixed F16xF16
fixed point Cube Unit simulation on top of the validated functional
reference in ``matmul.py``.
"""

import torch

from ....utils.vars import FIXED_SHIFT_VALUE_FP16_FP16_DEFAULT
from .matmul import f16_matmul_f16
from .matmul import int32_dequant_to_fp16
from .shift_value_search import ShiftValueSearchMixin


class Fp16FixedLinear(ShiftValueSearchMixin, torch.nn.Linear):
    """Linear running the fp16fixed F16xF16 fixed point simulation."""

    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        fixed_shift_value=FIXED_SHIFT_VALUE_FP16_FP16_DEFAULT,
    ):
        super().__init__(in_features, out_features, bias=bias)
        self.fixed_shift_value = int(fixed_shift_value)
        self._sse_sum = None

    @classmethod
    def from_module(cls, module, fixed_shift_value=FIXED_SHIFT_VALUE_FP16_FP16_DEFAULT):
        """Build from an existing Linear module (state dict copied)."""
        new_module = cls(
            module.in_features,
            module.out_features,
            bias=module.bias is not None,
            fixed_shift_value=fixed_shift_value,
        )
        new_module.load_state_dict(module.state_dict())
        # keep the fp16fixed module on the same device as the source module
        # so device-resident models can run the simulation without a
        # device mismatch
        new_module = new_module.to(module.weight.device)
        return new_module

    def _forward_impl(self, x, fixed_shift_value):
        # step 1: INT32 accumulator on the 2**(sv-58) fixed point grid
        # (the bias is pre-scaled onto the grid by f16_matmul_f16);
        # step 2: divide the grid back out to fp16
        int32_accumulator = f16_matmul_f16(
            x.to(torch.float16),
            self.weight.to(torch.float16),
            self.bias,
            fixed_shift_value,
        )
        return int32_dequant_to_fp16(int32_accumulator, fixed_shift_value)

    def forward(self, x):
        return self._forward_impl(x, self.fixed_shift_value)

    def _reference_output(self, x):
        return torch.nn.functional.linear(
            x.to(torch.float32),
            self.weight.to(torch.float32),
            self.bias.to(torch.float32) if self.bias is not None else None,
        )
