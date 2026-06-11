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
"""Structured pruning subpackage: an AMCT-style facade exposed via amct.prune, aligned with amct.quantize.

The public surface is deliberately minimal (entry functions + predefined configs only).
Internal components (AutoPruner / MethodRegistry / method classes / result and config
dataclasses) stay out of the package namespace and are imported by path when needed.
"""

__version__ = "0.4.0"

__all__ = [
    # entry points
    "prune",
    "PrunePass",
    "get_last_report",
    "get_last_auto_prune",
    "accuracy_based_auto_prune",
    "accuracy_based_auto_prune_with_budget",
    "size_budget_prune",
    "guarded_variance_hybrid_prune",
    "guarded_recovery_hybrid_prune",
    "finetune",
    "diagnose",
    # pre-defined configs
    "CNN_RECONSTRUCT_PRUNE_CFG",
    "CNN_VARIANCE_PRUNE_CFG",
    "DENSE_LOWVAR_PRUNE_CFG",
    "FULL_STRUCTURED_PRUNE_CFG",
    "MOE_MASSVAR_PRUNE_CFG",
    "MOE_OUTPUT_MERGE_PRUNE_CFG",
    "SENSITIVITY_ALLOC_PRUNE_CFG",
]

from .api import prune, PrunePass, get_last_report, get_last_auto_prune
from .accuracy_based_auto_prune import (
    accuracy_based_auto_prune,
    accuracy_based_auto_prune_with_budget,
    size_budget_prune,
)
from .variance_hybrid import guarded_variance_hybrid_prune
from .recovery_hybrid import guarded_recovery_hybrid_prune
from .finetune import finetune
from .diagnostics import diagnose
from .presets import (
    CNN_RECONSTRUCT_PRUNE_CFG,
    CNN_VARIANCE_PRUNE_CFG,
    DENSE_LOWVAR_PRUNE_CFG,
    FULL_STRUCTURED_PRUNE_CFG,
    MOE_OUTPUT_MERGE_PRUNE_CFG,
    MOE_MASSVAR_PRUNE_CFG,
    SENSITIVITY_ALLOC_PRUNE_CFG,
)
