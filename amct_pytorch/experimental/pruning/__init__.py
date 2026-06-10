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
"""结构化剪枝子包：通过 amct.prune 暴露与 amct.quantize 对齐的 AMCT 风格 façade。

公开接口刻意精简（与顶层 amct_pytorch 一致：只暴露入口函数 + 预定义配置）。
内部构件（AutoPruner / MethodRegistry / 各方法类 / 结果与配置数据类等）不进入包级公开命名空间，
需要时从对应子模块按路径导入。
"""

__version__ = "0.3.0"

__all__ = [
    # entry points
    "prune",
    "PrunePass",
    "get_last_report",
    "get_last_auto_prune",
    "accuracy_based_auto_prune",
    "accuracy_based_auto_prune_with_budget",
    "size_budget_prune",
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
