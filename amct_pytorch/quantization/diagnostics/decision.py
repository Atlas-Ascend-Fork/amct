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

from .config import QuantDiagnosisConfig
from .report import BlockMetric, LayerMetric, MoEMetric


def pick_top_k(total: int, cfg: QuantDiagnosisConfig) -> int:
    if total <= 0:
        return 0
    return min(total, cfg.top_k_max, max(1, cfg.top_k_min, int(total * cfg.top_k_ratio + 0.9999)))


def decide_block_precision(metric: BlockMetric, cfg: QuantDiagnosisConfig) -> int:
    precision = cfg.precision_map["normal"]
    if metric.qsnr <= 20.0:
        precision = max(precision, cfg.precision_map["sensitive"])
    if metric.qsnr <= 10.0:
        precision = max(precision, cfg.precision_map["critical"])
    return precision


def decide_layer_precision(metric: LayerMetric, cfg: QuantDiagnosisConfig) -> int:
    hits = 0
    if metric.relative_error >= cfg.relative_error_floor:
        hits += 1
    if metric.cosine_similarity <= cfg.cosine_similarity_floor:
        hits += 1
    if metric.max_error >= cfg.max_error_floor:
        hits += 1
    if metric.mse >= cfg.max_error_floor * cfg.max_error_floor:
        hits += 1
    precision = cfg.precision_map["normal"]
    if hits >= cfg.layer_upgrade_min_hit_count:
        precision = max(precision, cfg.precision_map["sensitive"])
    if hits >= cfg.layer_upgrade_min_hit_count + 1 or metric.qsnr <= 10.0:
        precision = max(precision, cfg.precision_map["critical"])
    return precision


def decide_moe_precision(metric: MoEMetric, cfg: QuantDiagnosisConfig) -> int:
    precision = cfg.precision_map["sensitive"]
    if metric.routing_mismatch >= 0.2:
        precision = cfg.precision_map["critical"]
    return precision
