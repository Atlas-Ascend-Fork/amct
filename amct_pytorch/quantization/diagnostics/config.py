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

from dataclasses import dataclass, field


@dataclass
class QuantDiagnosisConfig:
    layer_limit: int = 1
    sample_limit: int = 1
    top_k_ratio: float = 0.1
    top_k_min: int = 4
    top_k_max: int = 8
    block_qsnr_percentile: float = 10.0
    block_nmse_percentile: float = 90.0
    layer_upgrade_min_hit_count: int = 2
    cosine_similarity_floor: float = 0.99
    relative_error_floor: float = 0.05
    max_error_floor: float = 0.1
    moe_enable: bool = True
    early_stop_enabled: bool = True
    precision_map: dict[str, int] = field(
        default_factory=lambda: {
            "normal": 4,
            "sensitive": 8,
            "critical": 16,
        }
    )
