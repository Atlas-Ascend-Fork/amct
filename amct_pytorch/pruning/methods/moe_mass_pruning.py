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
"""
MoE Mass-Based Expert Pruning — proven method from LLaDA2.1-mini experiments.

Algorithm:
1. Run one epoch of data, collect per-expert activation counts (mass).
2. Per layer, keep top-N experts by mass (activation frequency).
3. Prune router (gate) output dimension to match kept experts.
4. Override group_limited_topk with simple topk (for models that use it).
5. No expert merging, no fine-tuning — pruning only.

Key findings (LLaDA2.1-mini, 256 -> 128 experts, GSM8K zero-shot):
- Mass-based pruning only: ACC 80-90% (baseline 88%)
- Any form of fine-tuning after pruning degrades ACC
- N_eff (effective number of experts) determines optimal keep count
- Gate weight parameter variance can identify specialized experts in deep layers,
  but mass alone performs as well or better in practice
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from ..config import MethodSpec, PruneConfig
from ..context import PruneContext
from ..domains.base import BasePruningDomain
from ..domains.moe import MoEPruningDomain, MoETarget, route_topk_from_output
from ..report import PruneReport
from ..utils import (
    run_calibration_epoch,
    get_submodule,
    infer_model_device,
    move_batch_to_device,
    topk_keep_indices,
)
from ._moe_mass_common import patch_router_module, prune_one_layer
from .base import BasePruningMethod


@dataclass(slots=True)
class ActivationCounter:
    """Counts how often each expert is selected in top-k routing."""

    counts: torch.Tensor | None = None
    num_experts: int = 0

    def update(self, router_output: torch.Tensor, top_k: int = 8) -> None:
        """Update counts from router output (logits / tuple / fused TopKRouter)."""
        # structure-agnostic parse of top-k expert indices (handles fused router (logits, wt, idx) tuple)
        topk_idx, _ = route_topk_from_output(router_output, self.num_experts, top_k)
        if topk_idx is None:
            return
        flat_idx = topk_idx.reshape(-1)

        if self.counts is None:
            n = self.num_experts or (int(flat_idx.max().item()) + 1)
            self.num_experts = n
            self.counts = torch.zeros(n, dtype=torch.long)

        max_idx = int(flat_idx.max().item())
        if max_idx >= len(self.counts):
            new_counts = torch.zeros(max_idx + 1, dtype=torch.long)
            new_counts[: len(self.counts)] = self.counts
            self.counts = new_counts
            self.num_experts = max_idx + 1

        for idx in flat_idx.cpu().tolist():
            if 0 <= idx < len(self.counts):
                self.counts[idx] += 1


def _make_router_hook(collector: ActivationCounter, k: int):
    def hook(module, inputs, output):
        _ = module
        _ = inputs
        collector.update(output, top_k=k)

    return hook


def _register_router_hooks(
    model: nn.Module,
    domain: MoEPruningDomain,
    targets: List[MoETarget],
    top_k: int,
):
    collectors: Dict[str, ActivationCounter] = {}
    hooks: List[torch.utils.hooks.RemovableHandle] = []
    for target in targets:
        if target.router_path is None:
            continue
        router = get_submodule(model, target.router_path)
        collector = ActivationCounter()
        # preset expert count (fused structures can't infer the full count from indices;
        # also aids router-output disambiguation)
        try:
            collector.num_experts = domain.num_experts(model, target)
        except (TypeError, ValueError):
            pass
        collectors[target.router_path] = collector
        hooks.append(router.register_forward_hook(_make_router_hook(collector, top_k)))
    return collectors, hooks


def _select_keep_indices(
    target: MoETarget,
    collectors: Dict[str, ActivationCounter],
    domain: MoEPruningDomain,
    model: nn.Module,
    config: PruneConfig,
    prune_ratio: float,
):
    if target.router_path is None or target.router_path not in collectors:
        return None, 0
    collector = collectors[target.router_path]
    if collector.counts is None:
        return None, 0
    num_experts = domain.num_experts(model, target)
    if num_experts <= config.min_experts:
        return None, num_experts
    mass = collector.counts[:num_experts].float()
    keep_idx = topk_keep_indices(mass, prune_ratio, config.min_experts)
    if len(keep_idx) == num_experts:
        return None, num_experts
    return keep_idx, num_experts


def _prune_layers(
    method_name: str,
    model: nn.Module,
    domain: MoEPruningDomain,
    targets: List[MoETarget],
    collectors: Dict[str, ActivationCounter],
    config: PruneConfig,
    report: PruneReport,
    prune_ratio: float,
    top_k: int,
) -> Optional[int]:
    last_num_experts: Optional[int] = None
    for target in targets:
        keep_idx, num_experts = _select_keep_indices(
            target, collectors, domain, model, config, prune_ratio
        )
        if keep_idx is None:
            continue
        prune_one_layer(model, domain, target, keep_idx, top_k)
        report.add(
            domain.name,
            method_name,
            target.module_path or target.experts_path,
            f"Mass-based prune experts {num_experts} -> {len(keep_idx)}",
        )
        report.record_layer_sparsity(
            target.module_path or target.experts_path, num_experts, len(keep_idx)
        )
        last_num_experts = len(keep_idx)
    return last_num_experts


class MassMoEPruningMethod(BasePruningMethod):
    """
    Prune MoE experts by activation mass (frequency).

    Keeps the most frequently activated experts per layer.
    No merging — removed experts are simply dropped.
    Router output dimension is pruned to match.

    Proven on LLaDA2.1-mini: 256->128 experts, ACC 80-90% (baseline 88%).

    kwargs:
        prune_ratio (float): Fraction of experts to remove. Default 0.50.
        top_k (int): Number of experts selected per token. Default 8.
    """

    domain = "moe"
    name = "activation_count"
    requires_data = True

    def apply(
        self,
        model: nn.Module,
        domain: BasePruningDomain,
        targets: List[MoETarget],
        context: PruneContext,
        report: PruneReport,
        config: PruneConfig,
        spec: MethodSpec,
    ) -> None:
        if context.data is None:
            raise ValueError(
                "MoE mass pruning requires input data via prune(..., data=...)."
            )
        if not isinstance(domain, MoEPruningDomain):
            raise TypeError("MassMoEPruningMethod requires MoEPruningDomain.")

        prune_ratio = float(spec.kwargs.get("prune_ratio", 0.50))
        top_k = int(spec.kwargs.get("top_k", 8))
        if prune_ratio <= 0:
            return

        collectors, hooks = _register_router_hooks(model, domain, targets, top_k)
        try:
            num_batches = run_calibration_epoch(model, context)
        finally:
            for h in hooks:
                h.remove()

        if num_batches == 0:
            msg = "MoE mass pruning received empty dataset. Skipped."
            warnings.warn(msg, RuntimeWarning, stacklevel=2)
            report.add_warning(msg)
            return

        last_num_experts = _prune_layers(
            self.name,
            model,
            domain,
            targets,
            collectors,
            config,
            report,
            prune_ratio,
            top_k,
        )

        if last_num_experts is not None:
            meta = getattr(model, "_amct_prune_meta", {})
            meta["moe_num_experts"] = int(last_num_experts)
            setattr(model, "_amct_prune_meta", meta)
