# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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
"""Structured pruning subpackage exposed via amct.prune.

This is the eager torch-module pruning path (rebuilds nn.Modules in place); it is
distinct from the existing graph/ONNX channel-prune in
amct_pytorch/common/auto_channel_prune, which operates on a traced graph.
"""


__all__ = [
    # entry points
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

# __all__ is the advertised surface; the names below stay importable but are kept out of
# __all__ (reach them via ``from amct_pytorch.pruning import ...``).
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
