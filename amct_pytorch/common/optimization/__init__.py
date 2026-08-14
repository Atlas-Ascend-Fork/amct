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

__all__ = ["register_solvers", "SOLVER_REGISTRY", "CalibrateBuildContext"]

from dataclasses import dataclass
from typing import Any

from amct_pytorch.common.utils.registry_factory import Registry

SOLVER_REGISTRY = Registry("solver")

_REGISTERED = False


def register_solvers():
    global _REGISTERED
    if _REGISTERED:
        return

    from .blockwise_solver import BlockwiseSolver  # noqa: F401
    from .calibration_solver import CalibrationSolver  # noqa: F401

    _REGISTERED = True


@dataclass
class CalibrateBuildContext:
    """Build contract for any ALGO_REGISTRY entry with solver="calibrate": the registered
    target is called as target(quant_linear, layer_name, ctx) -> algo, where algo exposes
    .forward(inputs). Mirrors AlgoBuildContext (amct_pytorch/algorithms/quant/__init__.py),
    the analogous build-context dataclass for solver="gradient" algorithms."""

    args: Any
    num_batches: int
