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

from dataclasses import dataclass, field, fields
from typing import Dict, List, Optional


@dataclass
class StageEvent:
    stage: str
    method: str
    module: str
    detail: str


@dataclass
class PruneReport:
    backend: str = ""
    params_before: int = 0
    params_after: int = 0
    events: List[StageEvent] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    per_layer_sparsity: Dict[str, float] = field(default_factory=dict)
    budget_unreachable: bool = False
    prunable_fraction: float | None = None
    allocation_choice: Optional[str] = None

    def copy_from(self, other: "PruneReport") -> None:
        """Absorb another report's contents into this caller-provided sink (in place)."""
        for f in fields(other):
            setattr(self, f.name, getattr(other, f.name))

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
