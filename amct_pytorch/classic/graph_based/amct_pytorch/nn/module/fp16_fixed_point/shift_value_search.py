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
"""Shift value search mixin shared by the fp16fixed operator classes.

Provides the per-candidate SSE search on top of two primitives a concrete
fp16fixed operator class has to implement:

- ``_forward_impl(x, fixed_shift_value)``: the fp16fixed forward;
- ``_reference_output(x)``: the fp32 reference output.

Two consumption styles:

- single shot: ``search_fixed_shift_value(x)``;
- multi batch: ``start_calibration()`` / ``accumulate_sse(x)`` * batch_num
  / ``finalize()`` -- per-candidate SSE of every batch is accumulated and
  the argmin is taken at the end, the same semantics as
  ``ShiftCalibrationModule``.
"""

import torch

from ....utils.vars import FIXED_SHIFT_VALUE_FP16_FP16_CANDIDATES


class ShiftValueSearchMixin:
    """Per-candidate SSE based fixed_shift_value search.

    The best candidate is the one whose fp16fixed forward output
    stays closest, in sum of squared error, to the fp32 reference output
    over the calibration data (one batch or accumulated batches).
    """

    candidates = list(FIXED_SHIFT_VALUE_FP16_FP16_CANDIDATES)

    def _forward_impl(self, x, fixed_shift_value):
        """fp16fixed forward with an explicit shift value."""
        raise NotImplementedError

    def _reference_output(self, x):
        """fp32 reference output of the operator."""
        raise NotImplementedError

    def compute_candidate_sse(self, x):
        """Per-candidate SSE of one batch against the fp32 reference.

        Returns a ``[len(candidates)]`` float64 tensor ordered as
        ``self.candidates``.
        """
        with torch.no_grad():
            reference = self._reference_output(x)
            sse = torch.zeros(len(self.candidates), dtype=torch.float64)
            for idx, candidate in enumerate(self.candidates):
                out = self._forward_impl(x, candidate)
                # squared error over all output elements, compared in
                # the fp32 domain
                diff = out.to(torch.float32) - reference
                sse[idx] = diff.pow(2).sum().to(torch.float64)
            return sse

    def search_fixed_shift_value(self, x):
        """Single shot search; writes the best candidate back to self."""
        best_index = int(torch.argmin(self.compute_candidate_sse(x)).item())
        self.fixed_shift_value = int(self.candidates[best_index])
        return self.fixed_shift_value

    def start_calibration(self):
        """Reset the multi-batch SSE accumulation."""
        self._sse_sum = None

    def accumulate_sse(self, x):
        """Accumulate the per-candidate SSE of one calibration batch."""
        # element-wise sum of the per-batch SSE vectors; the argmin is
        # taken once at finalize(), the same semantics as
        # ShiftCalibrationModule
        sse = self.compute_candidate_sse(x)
        if getattr(self, '_sse_sum', None) is None:
            self._sse_sum = sse
        else:
            self._sse_sum = self._sse_sum + sse
        return sse

    def finalize(self):
        """Take the argmin over the accumulated SSE and write it back."""
        if getattr(self, '_sse_sum', None) is None:
            raise RuntimeError(
                'no SSE accumulated, call start_calibration() and '
                'accumulate_sse(x) first'
            )
        best_index = int(torch.argmin(self._sse_sum).item())
        self.fixed_shift_value = int(self.candidates[best_index])
        self._sse_sum = None
        return self.fixed_shift_value
