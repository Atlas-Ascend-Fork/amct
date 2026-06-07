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
"""结构化剪枝子包：保留 AutoPruner 原生接口，并通过 amct.prune 暴露 AMCT 风格 façade。"""

__version__ = "0.3.0"

__all__ = [
    "__version__",
    "AutoPruner",
    "MethodRegistry",
    "MethodSpec",
    "PruneConfig",
    "PruneContext",
    "PruneReport",
    "BatchAdapter",
    "BasePruningMethod",
    "VarianceChannelPruningMethod",
    "LowVarianceDensePruningMethod",
    "MassMoEPruningMethod",
    "MassVarianceMoEPruningMethod",
    "create_default_registry",
    "collect_runtime_environment",
    "save_runtime_environment",
    "save_tokenizer",
    "prune",
    "PrunePass",
    "get_last_report",
    "get_last_auto_prune",
    "accuracy_based_auto_prune",
    "AutoTuneResult",
    "TrialResult",
    "diagnose",
    "DiagnosisReport",
    "CNN_VARIANCE_PRUNE_CFG",
    "CNN_DEPGRAPH_PRUNE_CFG",
    "DENSE_LOWVAR_PRUNE_CFG",
    "MOE_MASSVAR_PRUNE_CFG",
    "FULL_STRUCTURED_PRUNE_CFG",
]

from .api import prune, PrunePass, get_last_report, get_last_auto_prune
from .accuracy_based_auto_prune import AutoTuneResult, TrialResult, accuracy_based_auto_prune
from .diagnostics import DiagnosisReport, diagnose
from .config import MethodSpec, PruneConfig
from .context import BatchAdapter, PruneContext
from .methods import (
    BasePruningMethod,
    LowVarianceDensePruningMethod,
    MassMoEPruningMethod,
    MassVarianceMoEPruningMethod,
    VarianceChannelPruningMethod,
)
from .presets import (
    CNN_DEPGRAPH_PRUNE_CFG,
    CNN_VARIANCE_PRUNE_CFG,
    DENSE_LOWVAR_PRUNE_CFG,
    FULL_STRUCTURED_PRUNE_CFG,
    MOE_MASSVAR_PRUNE_CFG,
)
from .pruner import AutoPruner
from .registry import MethodRegistry, create_default_registry
from .report import PruneReport
from .runtime import collect_runtime_environment, save_runtime_environment, save_tokenizer
