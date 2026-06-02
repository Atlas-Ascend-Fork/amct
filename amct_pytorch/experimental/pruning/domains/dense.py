from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch.nn as nn

from ..config import PruneConfig
from ..utils import (
    get_submodule,
    is_activation_like,
    is_norm_like,
    iter_direct_named_modules,
    prune_linear_in_features,
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


class DensePruningDomain(BasePruningDomain):
    name = "dense"

    def find_targets(self, model: nn.Module, config: PruneConfig) -> List[TwoLayerDenseTarget | GatedDenseTarget]:
        targets: List[TwoLayerDenseTarget | GatedDenseTarget] = []
        for parent_name, parent in list(model.named_modules()):
            gated = self._find_gated_pattern(parent_name, parent, config)
            if gated is not None:
                targets.append(gated)
                continue
            targets.extend(self._find_two_layer_patterns(parent_name, parent, config))
        return targets

    def hidden_size(self, model: nn.Module, target: TwoLayerDenseTarget | GatedDenseTarget) -> int:
        if isinstance(target, TwoLayerDenseTarget):
            producer = get_submodule(model, target.producer_path)
            if not isinstance(producer, nn.Linear):
                raise TypeError(f"Producer '{target.producer_path}' is not Linear.")
            return int(producer.out_features)
        down = get_submodule(model, target.down_path)
        if not isinstance(down, nn.Linear):
            raise TypeError(f"Down projection '{target.down_path}' is not Linear.")
        return int(down.in_features)

    def apply_keep_indices(
        self,
        model: nn.Module,
        target: TwoLayerDenseTarget | GatedDenseTarget,
        keep_idx: List[int],
    ) -> None:
        if isinstance(target, TwoLayerDenseTarget):
            producer = get_submodule(model, target.producer_path)
            consumer = get_submodule(model, target.consumer_path)
            if not isinstance(producer, nn.Linear) or not isinstance(consumer, nn.Linear):
                raise TypeError("Dense target modules must both be Linear.")
            replace_submodule(model, target.producer_path, prune_linear_out_features(producer, keep_idx))
            replace_submodule(model, target.consumer_path, prune_linear_in_features(consumer, keep_idx))
            return

        gate = get_submodule(model, target.gate_path)
        up = get_submodule(model, target.up_path)
        down = get_submodule(model, target.down_path)
        if not isinstance(gate, nn.Linear) or not isinstance(up, nn.Linear) or not isinstance(down, nn.Linear):
            raise TypeError("Gated dense target must be composed of Linear layers.")
        replace_submodule(model, target.gate_path, prune_linear_out_features(gate, keep_idx))
        replace_submodule(model, target.up_path, prune_linear_out_features(up, keep_idx))
        replace_submodule(model, target.down_path, prune_linear_in_features(down, keep_idx))

    def _find_two_layer_patterns(
        self, parent_name: str, parent: nn.Module, config: PruneConfig
    ) -> List[TwoLayerDenseTarget]:
        items = iter_direct_named_modules(parent)
        patterns: List[TwoLayerDenseTarget] = []
        for i, (name1, mod1) in enumerate(items):
            if not isinstance(mod1, nn.Linear):
                continue
            if mod1.out_features <= config.min_neurons:
                continue
            for j in range(i + 1, len(items)):
                name2, mod2 = items[j]
                if is_activation_like(mod2) or is_norm_like(mod2):
                    continue
                if isinstance(mod2, nn.Linear) and mod1.out_features == mod2.in_features:
                    producer_path = f"{parent_name}.{name1}" if parent_name else name1
                    consumer_path = f"{parent_name}.{name2}" if parent_name else name2
                    patterns.append(TwoLayerDenseTarget(producer_path=producer_path, consumer_path=consumer_path))
                break
        return patterns

    def _find_gated_pattern(
        self, parent_name: str, parent: nn.Module, config: PruneConfig
    ) -> Optional[GatedDenseTarget]:
        children = dict(iter_direct_named_modules(parent))
        gated = self._find_gated_by_canonical_names(parent_name, children, config)
        if gated is not None:
            return gated
        return self._find_gated_by_shape_inference(parent_name, children, config)

    @staticmethod
    def _find_gated_by_canonical_names(
        parent_name: str, children: dict, config: PruneConfig
    ) -> Optional[GatedDenseTarget]:
        gate = children.get("gate_proj")
        up = children.get("up_proj")
        down = children.get("down_proj")
        if not (isinstance(gate, nn.Linear) and isinstance(up, nn.Linear) and isinstance(down, nn.Linear)):
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

    @staticmethod
    def _find_gated_by_shape_inference(
        parent_name: str, children: dict, config: PruneConfig
    ) -> Optional[GatedDenseTarget]:
        linear_children = [(n, m) for n, m in children.items() if isinstance(m, nn.Linear)]
        if len(linear_children) < 3:
            return None
        for down_name, down_mod in linear_children:
            if down_mod.in_features <= config.min_neurons:
                continue
            same_hidden = [
                (n, m)
                for n, m in linear_children
                if m.out_features == down_mod.in_features and n != down_name
            ]
            if len(same_hidden) >= 2:
                (gate_name, _), (up_name, _) = same_hidden[:2]
                return GatedDenseTarget(
                    gate_path=f"{parent_name}.{gate_name}" if parent_name else gate_name,
                    up_path=f"{parent_name}.{up_name}" if parent_name else up_name,
                    down_path=f"{parent_name}.{down_name}" if parent_name else down_name,
                )
        return None
