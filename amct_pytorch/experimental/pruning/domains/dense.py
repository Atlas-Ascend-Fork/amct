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

from dataclasses import dataclass
from typing import List, Optional

import torch.nn as nn

from ..config import PruneConfig
from ..utils import (
    get_submodule,
    is_activation_like,
    is_linear_like,
    is_norm_like,
    iter_direct_named_modules,
    linear_like_in_features,
    linear_like_out_features,
    prune_linear_in_features,
    prune_linear_like_in_features,
    prune_linear_like_out_features,
    prune_linear_out_features,
    replace_submodule,
)
from .base import BasePruningDomain


@dataclass(slots=True)
class TwoLayerDenseTarget:
    producer_path: str
    consumer_path: str


@dataclass(slots=True)
class GatedDenseTarget:
    gate_path: str
    up_path: str
    down_path: str


def _find_gated_by_canonical_names(
    parent_name: str, children: dict, config: PruneConfig
) -> Optional[GatedDenseTarget]:
    gate = children.get("gate_proj")
    up = children.get("up_proj")
    down = children.get("down_proj")
    if not (
        isinstance(gate, nn.Linear)
        and isinstance(up, nn.Linear)
        and isinstance(down, nn.Linear)
    ):
        return None
    if gate.out_features != up.out_features or up.out_features != down.in_features:
        return None
    if down.in_features <= config.min_neurons:
        return None
    return GatedDenseTarget(
        gate_path=f"{parent_name}.gate_proj" if parent_name else "gate_proj",
        up_path=f"{parent_name}.up_proj" if parent_name else "up_proj",
        down_path=f"{parent_name}.down_proj" if parent_name else "down_proj",
    )


def _find_gated_by_shape_inference(
    parent_name: str, children: dict, config: PruneConfig
) -> Optional[GatedDenseTarget]:
    linear_children = [(n, m) for n, m in children.items() if isinstance(m, nn.Linear)]
    if len(linear_children) < 3:
        return None
    for down_name, down_mod in linear_children:
        if down_mod.in_features <= config.min_neurons:
            continue
        if _looks_like_attention_proj(down_name):
            continue
        # gate/up：输出维 == down 的输入维（中间维），且其输入维 == down 的输出维
        # （门控 FFN 复原残差流维度 in→inter→in）。后一条可排除注意力 q/o→k 的误配。
        same_hidden = [
            (n, m)
            for n, m in linear_children
            if m.out_features == down_mod.in_features
            and n != down_name
            and m.in_features == down_mod.out_features
            and not _looks_like_attention_proj(n)
        ]
        if len(same_hidden) >= 2:
            (gate_name, _), (up_name, _) = same_hidden[:2]
            return GatedDenseTarget(
                gate_path=f"{parent_name}.{gate_name}" if parent_name else gate_name,
                up_path=f"{parent_name}.{up_name}" if parent_name else up_name,
                down_path=f"{parent_name}.{down_name}" if parent_name else down_name,
            )
    return None


def _find_gated_pattern(
    parent_name: str, parent: nn.Module, config: PruneConfig
) -> Optional[GatedDenseTarget]:
    children = dict(iter_direct_named_modules(parent))
    gated = _find_gated_by_canonical_names(parent_name, children, config)
    if gated is not None:
        return gated
    return _find_gated_by_shape_inference(parent_name, children, config)


# 注意力投影命名（用于额外的安全排除；结构判据为主，命名为辅）
_ATTENTION_PROJ_HINTS = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "query",
    "key",
    "value",
    "out_proj",
    "wq",
    "wk",
    "wv",
    "wo",
    "qkv",
)


def _looks_like_attention_proj(name: str) -> bool:
    low = name.lower()
    return any(h in low for h in _ATTENTION_PROJ_HINTS)


def _find_two_layer_patterns(
    parent_name: str, parent: nn.Module, config: PruneConfig
) -> List[TwoLayerDenseTarget]:
    items = iter_direct_named_modules(parent)
    # 并列投影判据：注意力的 q/k/v/o 都读取同一输入(hidden)，故父模块内会有多个
    # in_features 相同的 Linear；它们不是“producer→consumer 顺序 FFN”，必须排除，
    # 否则会把 q_proj→k_proj 误当两层 FFN 剪掉而破坏注意力。
    # linear-like = nn.Linear 或 GPT-2 风格 Conv1D（等价线性层，weight 转置存储）
    in_feature_counts: dict = {}
    for _, m in items:
        if is_linear_like(m):
            inf = linear_like_in_features(m)
            in_feature_counts[inf] = in_feature_counts.get(inf, 0) + 1

    patterns: List[TwoLayerDenseTarget] = []
    for i, (name1, mod1) in enumerate(items):
        if not is_linear_like(mod1):
            continue
        if linear_like_out_features(mod1) <= config.min_neurons:
            continue
        # mod1 与至少一个兄弟共享输入 → 并列投影（如注意力），跳过
        if in_feature_counts.get(linear_like_in_features(mod1), 0) >= 2:
            continue
        if _looks_like_attention_proj(name1):
            continue
        for j in range(i + 1, len(items)):
            name2, mod2 = items[j]
            if is_activation_like(mod2) or is_norm_like(mod2):
                continue
            if (
                is_linear_like(mod2)
                and linear_like_out_features(mod1) == linear_like_in_features(mod2)
                and not _looks_like_attention_proj(name2)
            ):
                producer_path = f"{parent_name}.{name1}" if parent_name else name1
                consumer_path = f"{parent_name}.{name2}" if parent_name else name2
                patterns.append(
                    TwoLayerDenseTarget(
                        producer_path=producer_path, consumer_path=consumer_path
                    )
                )
            break
    return patterns


class DensePruningDomain(BasePruningDomain):
    name = "dense"

    def find_targets(
        self, model: nn.Module, config: PruneConfig
    ) -> List[TwoLayerDenseTarget | GatedDenseTarget]:
        targets: List[TwoLayerDenseTarget | GatedDenseTarget] = []
        for parent_name, parent in list(model.named_modules()):
            gated = _find_gated_pattern(parent_name, parent, config)
            if gated is not None:
                targets.append(gated)
                continue
            targets.extend(_find_two_layer_patterns(parent_name, parent, config))
        return targets

    def apply_keep_indices(
        self,
        model: nn.Module,
        target: TwoLayerDenseTarget | GatedDenseTarget,
        keep_idx: List[int],
    ) -> None:
        if isinstance(target, TwoLayerDenseTarget):
            producer = get_submodule(model, target.producer_path)
            consumer = get_submodule(model, target.consumer_path)
            if not is_linear_like(producer) or not is_linear_like(consumer):
                raise TypeError("Dense target modules must both be Linear/Conv1D.")
            replace_submodule(
                model,
                target.producer_path,
                prune_linear_like_out_features(producer, keep_idx),
            )
            replace_submodule(
                model,
                target.consumer_path,
                prune_linear_like_in_features(consumer, keep_idx),
            )
            return

        gate = get_submodule(model, target.gate_path)
        up = get_submodule(model, target.up_path)
        down = get_submodule(model, target.down_path)
        if (
            not isinstance(gate, nn.Linear)
            or not isinstance(up, nn.Linear)
            or not isinstance(down, nn.Linear)
        ):
            raise TypeError("Gated dense target must be composed of Linear layers.")
        replace_submodule(
            model, target.gate_path, prune_linear_out_features(gate, keep_idx)
        )
        replace_submodule(
            model, target.up_path, prune_linear_out_features(up, keep_idx)
        )
        replace_submodule(
            model, target.down_path, prune_linear_in_features(down, keep_idx)
        )

    def hidden_size(
        self, model: nn.Module, target: TwoLayerDenseTarget | GatedDenseTarget
    ) -> int:
        if isinstance(target, TwoLayerDenseTarget):
            producer = get_submodule(model, target.producer_path)
            if not is_linear_like(producer):
                raise TypeError(
                    f"Producer '{target.producer_path}' is not Linear/Conv1D."
                )
            return linear_like_out_features(producer)
        down = get_submodule(model, target.down_path)
        if not isinstance(down, nn.Linear):
            raise TypeError(f"Down projection '{target.down_path}' is not Linear.")
        return int(down.in_features)
