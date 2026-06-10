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

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass(slots=True)
class StageEvent:
    stage: str
    method: str
    module: str
    detail: str


@dataclass(slots=True)
class PruneReport:
    backend: str
    params_before: int
    params_after: int = 0
    events: List[StageEvent] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    per_layer_sparsity: Dict[str, float] = field(default_factory=dict)
    #: 预算诚实性：尺寸预算低于"不可剪质量"下限时置 True（见 size_budget_prune）。
    budget_unreachable: bool = False
    #: 可剪参数占总参数比例（未计算时为 None）。
    prunable_fraction: float | None = None
    #: 按层分配剪枝时的最终选择（'sensitivity' / 'uniform'）；非分配路径为 None。
    allocation_choice: Optional[str] = None

    def add(self, stage: str, method: str, module: str, detail: str) -> None:
        self.events.append(
            StageEvent(stage=stage, method=method, module=module, detail=detail)
        )

    def add_warning(self, message: str) -> None:
        self.warnings.append(message)

    def record_layer_sparsity(
        self, module_path: str, params_before: int, params_after: int
    ) -> None:
        if params_before > 0:
            self.per_layer_sparsity[module_path] = 1.0 - (params_after / params_before)

    def as_dict(self) -> Dict[str, object]:
        return {
            "backend": self.backend,
            "params_before": self.params_before,
            "params_after": self.params_after,
            "events": [
                {
                    "stage": e.stage,
                    "method": e.method,
                    "module": e.module,
                    "detail": e.detail,
                }
                for e in self.events
            ],
            "budget_unreachable": self.budget_unreachable,
            "prunable_fraction": self.prunable_fraction,
            "warnings": list(self.warnings),
            "per_layer_sparsity": dict(self.per_layer_sparsity),
            "allocation_choice": self.allocation_choice,
        }
