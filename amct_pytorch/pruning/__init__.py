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
Internal components (the pruner, method registry, method classes, and result/config
dataclasses) stay out of the package namespace and are imported by path when needed.
"""


__all__ = [
    # entry points (minimal: mirror amct.quantize -- prune + its accuracy-search twin)
    "prune",
    "accuracy_based_auto_prune",
    # pre-defined configs
    "CNN_RECONSTRUCT_PRUNE_CFG",
    "CNN_VARIANCE_PRUNE_CFG",
    "DENSE_LOWVAR_PRUNE_CFG",
    "FULL_STRUCTURED_PRUNE_CFG",
    "MOE_MASSVAR_PRUNE_CFG",
    "MOE_OUTPUT_MERGE_PRUNE_CFG",
    "SENSITIVITY_ALLOC_PRUNE_CFG",
]

# Minimal advertised surface = {prune, accuracy_based_auto_prune} + the *_PRUNE_CFG presets,
# mirroring amct.quantize. Everything below stays importable (kept defined here) but is
# intentionally absent from __all__ -- advanced/secondary entry points: PrunePass and the
# PruneReport sink (pass ``prune(..., report=PruneReport())`` to inspect stats; no process-global
# / model-attached state is kept, mirroring amct.quantize), size_budget_prune / prune_finetune /
# prune_diagnose (size_budget_prune is also reachable as the prune(size_budget=...) mode), and the
# advanced menu-mode configs (MOE_VARIANCE_MENU_CFG / DENSE_RECOVERY_MENU_CFG) that drive
# accuracy_based_auto_prune's guarded best-of-menu search. Reach them by name via
# ``from amct_pytorch.pruning import ...`` (or ``amct.<name>``) when needed.
from .api import (
    prune,
    PrunePass as PrunePass,
)
from .report import PruneReport as PruneReport
from .accuracy_based_auto_prune import (
    accuracy_based_auto_prune,
    size_budget_prune as size_budget_prune,
)
from .finetune import prune_finetune as prune_finetune
from .diagnostics import prune_diagnose as prune_diagnose
from .presets import (
    CNN_RECONSTRUCT_PRUNE_CFG,
    CNN_VARIANCE_PRUNE_CFG,
    DENSE_LOWVAR_PRUNE_CFG,
    FULL_STRUCTURED_PRUNE_CFG,
    MOE_OUTPUT_MERGE_PRUNE_CFG,
    MOE_MASSVAR_PRUNE_CFG,
    SENSITIVITY_ALLOC_PRUNE_CFG,
    MOE_VARIANCE_MENU_CFG as MOE_VARIANCE_MENU_CFG,
    DENSE_RECOVERY_MENU_CFG as DENSE_RECOVERY_MENU_CFG,
)
