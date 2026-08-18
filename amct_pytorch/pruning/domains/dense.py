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

import torch
import torch.nn as nn

from .. import simulate
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


@dataclass
class TwoLayerDenseTarget:
    producer_path: str
    consumer_path: str


@dataclass
class GatedDenseTarget:
    gate_path: str
    up_path: str
    down_path: str


@dataclass
class FusedGatedDenseTarget:
    """gate and up fused into one Linear (first half gate, second half up, e.g. Phi-3/GLM-4 gate_up_proj)."""

    gate_up_path: str
    down_path: str


def _find_fused_gated(
    parent_name: str, children: dict, config: PruneConfig
) -> Optional[FusedGatedDenseTarget]:
    for du_name in ("down_proj", "dense_4h_to_h", "fc2", "w2"):
        down = children.get(du_name)
        if isinstance(down, nn.Linear):
            break
    else:
        return None
    if down.in_features <= config.min_neurons:
        return None
    for gu_name, gu in children.items():
        if gu_name == du_name or not isinstance(gu, nn.Linear):
            continue
        if (
            gu.out_features == 2 * down.in_features
            and gu.in_features == down.out_features
            and not _looks_like_attention_proj(gu_name)
        ):
            prefix = f"{parent_name}." if parent_name else ""
            return FusedGatedDenseTarget(
                gate_up_path=f"{prefix}{gu_name}", down_path=f"{prefix}{du_name}"
            )
    return None


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
        if _looks_like_attention_proj(down_name) or _looks_like_embed_proj(down_name):
            continue
        same_hidden = [
            (n, m)
            for n, m in linear_children
            if m.out_features == down_mod.in_features
            and n != down_name
            and m.in_features == down_mod.out_features
            and not _looks_like_attention_proj(n)
            and not _looks_like_embed_proj(n)
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
    fused = _find_fused_gated(parent_name, children, config)
    if fused is not None:
        return fused
    return _find_gated_by_shape_inference(parent_name, children, config)


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


_EMBED_PROJ_HINTS = (
    "project_in",
    "project_out",
    "embed_in",
    "embed_out",
)


def _looks_like_embed_proj(name: str) -> bool:
    low = name.lower()
    return any(h in low for h in _EMBED_PROJ_HINTS)


def _is_shape_transparent(module: nn.Module, num_features: int) -> bool:
    """Transparency probe for paramless modules: transparent only if a zero-tensor forward keeps the last dim."""
    probe = torch.zeros(1, int(num_features))
    try:
        with torch.no_grad():
            out = module(probe)
    except (RuntimeError, ValueError, TypeError, AttributeError, KeyError, IndexError):
        return False
    if not torch.is_tensor(out):
        return False
    return out.shape[-1] == num_features


def _match_two_layer_pair(
    parent_name: str, items: list, idx: int
) -> Optional[TwoLayerDenseTarget]:
    """Take items[idx] as the producer and decide whether the next non-transparent module forms a two-layer FFN."""
    name1, mod1 = items[idx]
    out1 = linear_like_out_features(mod1)
    for j in range(idx + 1, len(items)):
        name2, mod2 = items[j]
        if is_activation_like(mod2) or is_norm_like(mod2):
            continue
        paramless = not any(True for _ in mod2.parameters(recurse=False))
        if not is_linear_like(mod2) and paramless:
            if _is_shape_transparent(mod2, out1):
                continue
            return None
        if (
            is_linear_like(mod2)
            and out1 == linear_like_in_features(mod2)
            and not _looks_like_attention_proj(name2)
            and not _looks_like_embed_proj(name2)
        ):
            prefix = f"{parent_name}." if parent_name else ""
            return TwoLayerDenseTarget(
                producer_path=f"{prefix}{name1}", consumer_path=f"{prefix}{name2}"
            )
        return None
    return None


def _find_two_layer_patterns(
    parent_name: str, parent: nn.Module, config: PruneConfig
) -> List[TwoLayerDenseTarget]:
    items = iter_direct_named_modules(parent)
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
        if in_feature_counts.get(linear_like_in_features(mod1), 0) >= 2:
            continue
        if _looks_like_attention_proj(name1) or _looks_like_embed_proj(name1):
            continue
        pair = _match_two_layer_pair(parent_name, items, i)
        if pair is not None:
            patterns.append(pair)
    return patterns


class DensePruningDomain(BasePruningDomain):
    name = "dense"

    def find_targets(
        self, model: nn.Module, config: PruneConfig
    ) -> List[TwoLayerDenseTarget | GatedDenseTarget | FusedGatedDenseTarget]:
        targets: List[
            TwoLayerDenseTarget | GatedDenseTarget | FusedGatedDenseTarget
        ] = []
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
        target: TwoLayerDenseTarget | GatedDenseTarget | FusedGatedDenseTarget,
        keep_idx: List[int],
    ) -> None:
        session = simulate.active()
        if session is None:
            self._apply_keep_indices(model, target, keep_idx)
        else:
            session.record_dense(
                model, target, self.hidden_size(model, target), keep_idx
            )

    def _apply_keep_indices(
        self,
        model: nn.Module,
        target: TwoLayerDenseTarget | GatedDenseTarget | FusedGatedDenseTarget,
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

        if isinstance(target, FusedGatedDenseTarget):
            gate_up = get_submodule(model, target.gate_up_path)
            down = get_submodule(model, target.down_path)
            if not isinstance(gate_up, nn.Linear) or not isinstance(down, nn.Linear):
                raise TypeError("Fused gated dense target must be Linear layers.")
            inter = down.in_features
            fused_keep = list(keep_idx) + [inter + i for i in keep_idx]
            replace_submodule(
                model,
                target.gate_up_path,
                prune_linear_out_features(gate_up, fused_keep),
            )
            replace_submodule(
                model, target.down_path, prune_linear_in_features(down, keep_idx)
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
        self,
        model: nn.Module,
        target: TwoLayerDenseTarget | GatedDenseTarget | FusedGatedDenseTarget,
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
