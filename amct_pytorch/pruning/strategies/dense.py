from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from ..report import PruneReport
from ..utils import (
    is_activation_like,
    is_norm_like,
    iter_direct_named_modules,
    prune_linear_in_features,
    prune_linear_out_features,
    replace_submodule,
    topk_keep_indices,
)
from .base import BasePruningStage


@dataclass(slots=True)
class TwoLayerPattern:
    producer_name: str
    producer: nn.Linear
    consumer_name: str
    consumer: nn.Linear


@dataclass(slots=True)
class GatedPattern:
    gate_name: str
    gate: nn.Linear
    up_name: str
    up: nn.Linear
    down_name: str
    down: nn.Linear


class DensePruningStage(BasePruningStage):
    stage_name = "dense"

    def _linear_score(self, linear: nn.Linear, dim: str) -> torch.Tensor:
        if dim == "out":
            return linear.weight.detach().abs().mean(dim=1).cpu()
        if dim == "in":
            return linear.weight.detach().abs().mean(dim=0).cpu()
        raise ValueError(dim)

    def _find_two_layer_patterns(self, parent: nn.Module) -> List[TwoLayerPattern]:
        items = iter_direct_named_modules(parent)
        patterns: List[TwoLayerPattern] = []
        for i, (name1, mod1) in enumerate(items):
            if not isinstance(mod1, nn.Linear):
                continue
            for j in range(i + 1, len(items)):
                name2, mod2 = items[j]
                if is_activation_like(mod2) or is_norm_like(mod2):
                    continue
                if isinstance(mod2, nn.Linear) and mod1.out_features == mod2.in_features:
                    patterns.append(TwoLayerPattern(name1, mod1, name2, mod2))
                break
        return patterns

    def _find_gated_pattern(self, parent: nn.Module) -> Optional[GatedPattern]:
        children = dict(iter_direct_named_modules(parent))
        gate = children.get("gate_proj")
        up = children.get("up_proj")
        down = children.get("down_proj")
        if isinstance(gate, nn.Linear) and isinstance(up, nn.Linear) and isinstance(down, nn.Linear):
            if gate.out_features == up.out_features == down.in_features:
                return GatedPattern("gate_proj", gate, "up_proj", up, "down_proj", down)

        linear_children = [(n, m) for n, m in children.items() if isinstance(m, nn.Linear)]
        if len(linear_children) < 3:
            return None
        producers = [(n, m) for n, m in linear_children if any(m.out_features == x[1].in_features for x in linear_children)]
        for down_name, down_mod in linear_children:
            same_hidden = [(n, m) for n, m in linear_children if m.out_features == down_mod.in_features and n != down_name]
            if len(same_hidden) >= 2:
                (gate_name, gate_mod), (up_name, up_mod) = same_hidden[:2]
                return GatedPattern(gate_name, gate_mod, up_name, up_mod, down_name, down_mod)
        return None

    def _apply_two_layer(
        self,
        model: nn.Module,
        parent_name: str,
        pattern: TwoLayerPattern,
        report: PruneReport,
    ) -> Optional[int]:
        producer = pattern.producer
        consumer = pattern.consumer
        hidden = producer.out_features
        if hidden <= self.config.min_neurons:
            return None

        score = self._linear_score(producer, "out") + self._linear_score(consumer, "in")
        keep_idx = topk_keep_indices(score, self.config.dense_ratio, self.config.min_neurons)
        if len(keep_idx) == hidden:
            return None

        producer_path = f"{parent_name}.{pattern.producer_name}" if parent_name else pattern.producer_name
        consumer_path = f"{parent_name}.{pattern.consumer_name}" if parent_name else pattern.consumer_name
        replace_submodule(model, producer_path, prune_linear_out_features(producer, keep_idx))
        replace_submodule(model, consumer_path, prune_linear_in_features(consumer, keep_idx))
        report.add(
            self.stage_name,
            parent_name or "<root>",
            f"MLP hidden {hidden} -> {len(keep_idx)} via {producer_path} / {consumer_path}",
        )
        return len(keep_idx)

    def _apply_gated(
        self,
        model: nn.Module,
        parent_name: str,
        pattern: GatedPattern,
        report: PruneReport,
    ) -> Optional[int]:
        hidden = pattern.down.in_features
        if hidden <= self.config.min_neurons:
            return None

        score = (
            self._linear_score(pattern.gate, "out")
            + self._linear_score(pattern.up, "out")
            + self._linear_score(pattern.down, "in")
        )
        keep_idx = topk_keep_indices(score, self.config.dense_ratio, self.config.min_neurons)
        if len(keep_idx) == hidden:
            return None

        for name, module, mode in (
            (pattern.gate_name, pattern.gate, "out"),
            (pattern.up_name, pattern.up, "out"),
            (pattern.down_name, pattern.down, "in"),
        ):
            path = f"{parent_name}.{name}" if parent_name else name
            if mode == "out":
                replace_submodule(model, path, prune_linear_out_features(module, keep_idx))
            else:
                replace_submodule(model, path, prune_linear_in_features(module, keep_idx))

        report.add(
            self.stage_name,
            parent_name or "<root>",
            f"Gated MLP hidden {hidden} -> {len(keep_idx)}",
        )
        return len(keep_idx)

    def apply(self, model: nn.Module, report: PruneReport) -> None:
        ratio = self.config.dense_ratio
        if ratio <= 0:
            return

        last_hidden: Optional[int] = None
        for parent_name, parent in list(model.named_modules()):
            gated_pattern = self._find_gated_pattern(parent)
            if gated_pattern is not None:
                new_hidden = self._apply_gated(model, parent_name, gated_pattern, report)
                if new_hidden is not None:
                    last_hidden = new_hidden
                    continue

            for pattern in self._find_two_layer_patterns(parent):
                new_hidden = self._apply_two_layer(model, parent_name, pattern, report)
                if new_hidden is not None:
                    last_hidden = new_hidden

        if last_hidden is not None:
            meta = getattr(model, "_ascend_prune_meta", {})
            meta["dense_hidden_size"] = int(last_hidden)
            setattr(model, "_ascend_prune_meta", meta)
