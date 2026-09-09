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

from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class BlockMetric:
    name: str
    index: int
    qsnr: float
    nmse: float
    score: float
    precision: int = 4


@dataclass
class LayerMetric:
    name: str
    index: int
    kind: str
    qsnr: float
    mse: float
    relative_error: float
    cosine_similarity: float
    max_error: float
    score: float
    precision: int


@dataclass
class InputOutlierMetric:
    name: str
    index: int
    kind: str
    max_abs: float
    p99_abs: float
    p999_abs: float
    token_ratio_p99: float
    token_ratio_max: float
    max_to_p999: float
    sample_count: int
    token_count: int


@dataclass
class MoEMetric:
    name: str
    index: int
    expert_frequency: dict[str, float]
    expert_error: dict[str, float]
    routing_mismatch: float
    score: float
    precision: int


@dataclass
class QuantDiagnosisReport:
    block_metrics: list[BlockMetric] = field(default_factory=list)
    layer_metrics: list[LayerMetric] = field(default_factory=list)
    input_outlier_metrics: list[InputOutlierMetric] = field(default_factory=list)
    moe_metrics: list[MoEMetric] = field(default_factory=list)
    mixed_precision_plan: dict[str, int] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"blocks={len(self.block_metrics)}",
            f"layers={len(self.layer_metrics)}",
            f"input_outliers={len(self.input_outlier_metrics)}",
            f"moe={len(self.moe_metrics)}",
        ]
        if self.mixed_precision_plan:
            lines.append(
                "plan="
                + ", ".join(
                    f"{name}:{bits}" for name, bits in sorted(self.mixed_precision_plan.items())
                )
            )
        for note in self.notes:
            lines.append(f"- {note}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
