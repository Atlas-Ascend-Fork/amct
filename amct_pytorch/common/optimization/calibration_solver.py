# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
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
# ----------------------------------------------------------------------------

from __future__ import annotations

from typing import Any

import torch

from amct_pytorch.common.optimization import SOLVER_REGISTRY
from amct_pytorch.common.optimization.base_solver import BaseSolver


@SOLVER_REGISTRY.register(
    name="calibrate",
    description="Forward-only calibration solver for statistic-based algorithms",
)
class CalibrationSolver(BaseSolver):
    """Drive a calibration module for the configured number of batches."""

    granularity = "calibrate"

    def solve(self, calibration_data, forward_kwargs=None):
        kwargs = forward_kwargs or {}
        self.model = self.model.to(self.args.device)
        with torch.no_grad():
            for data_batch in calibration_data:
                inputs = self._unpack_inputs(data_batch)
                self.model(inputs, **kwargs)
        return None

    def finalize(self):
        export_fn = getattr(self.model, "export_quant_params", None)
        if callable(export_fn):
            return export_fn()
        export_fn = getattr(self.model, "export_ptq_params", None)
        if callable(export_fn):
            return export_fn()
        return {}

    @staticmethod
    def _unpack_inputs(data_batch: Any):
        if isinstance(data_batch, (tuple, list)):
            if len(data_batch) != 1:
                raise ValueError(
                    "CalibrationSolver expects one input tensor per batch."
                )
            return data_batch[0]
        return data_batch
