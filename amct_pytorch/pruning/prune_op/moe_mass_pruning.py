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
from typing import Dict, List

import torch
import torch.nn as nn

from ...common.utils.log import LOGGER

from ..config import MethodSpec, PruneConfig
from ..context import PruneContext
from ..domains.base import BasePruningDomain
from ..domains.moe import (
    DEFAULT_TOP_K,
    MoEPruningDomain,
    MoETarget,
    route_topk_from_output,
)
from ..report import PruneReport
from ..utils import (
    record_prune_size,
    run_calibration_epoch,
    topk_keep_indices,
)
from ._moe_mass_common import prune_moe_layers, register_router_hooks
from .base import BasePruningMethod


@dataclass
class ActivationCounter:
    """Counts how often each expert is selected in top-k routing."""

    counts: torch.Tensor | None = None
    num_experts: int = 0

    def update(self, router_output: torch.Tensor, top_k: int = DEFAULT_TOP_K) -> None:
        """Update counts from router output (logits / tuple / fused TopKRouter)."""
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


def _make_collector(
    domain: MoEPruningDomain, model: nn.Module, target: MoETarget, top_k: int
) -> ActivationCounter:
    _ = top_k
    collector = ActivationCounter()
    try:
        collector.num_experts = domain.num_experts(model, target)
    except (TypeError, ValueError):
        pass
    return collector


def _register_router_hooks(
    model: nn.Module,
    domain: MoEPruningDomain,
    targets: List[MoETarget],
    top_k: int,
):
    return register_router_hooks(
        model, domain, targets, top_k, _make_collector, _make_router_hook
    )


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


class MassMoEPruningMethod(BasePruningMethod):
    """Prune MoE experts by activation mass (keep the most frequently activated per layer).

    kwargs:
        prune_ratio (float): Fraction of experts to remove. Default 0.50.
        top_k (int): Experts selected per token. Default: read from the
            router / model config (see MoEPruningDomain.resolve_top_k).
    """

    domain = "moe"
    name = "activation_count"
    accepted_kwargs = frozenset({"prune_ratio", "top_k"})
    supports_masked_trial = True
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
        top_k = spec.kwargs.get("top_k")
        top_k = None if top_k is None else int(top_k)
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
            LOGGER.logw(msg, "amct_prune")
            report.add_warning(msg)
            return

        def _select(_idx, target):
            keep_idx, num_experts = _select_keep_indices(
                target, collectors, domain, model, config, prune_ratio
            )
            if keep_idx is None:
                return None, num_experts, None
            message = f"Mass-based prune experts {num_experts} -> {len(keep_idx)}"
            return keep_idx, num_experts, message

        last_num_experts = prune_moe_layers(
            self.name, model, domain, targets, report, top_k, _select
        )

        if last_num_experts is not None:
            record_prune_size(model, "moe_num_experts", last_num_experts)
