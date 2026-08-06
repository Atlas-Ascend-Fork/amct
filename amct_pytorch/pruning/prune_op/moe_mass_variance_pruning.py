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
class LayerRoutingStats:
    """Per-layer routing statistics collected during one epoch."""

    mass: torch.Tensor
    expert_var_sum: torch.Tensor
    expert_select_count: torch.Tensor
    expert_var: torch.Tensor
    expert_max: torch.Tensor
    num_experts: int = 0
    top_k: int = DEFAULT_TOP_K


def _make_router_hook(stats: "LayerRoutingStats", k: int):
    def hook(module, inputs, output):
        _ = module
        _ = inputs
        topk_idx, topk_wt = route_topk_from_output(output, stats.num_experts, k)
        if topk_idx is None:
            return

        idx = topk_idx.reshape(-1).to(device="cpu", dtype=torch.long)
        wt = topk_wt.reshape(-1).to(device="cpu", dtype=torch.float32)
        valid = (idx >= 0) & (idx < stats.num_experts)
        if not bool(valid.any()):
            return
        idx, wt = idx[valid], wt[valid]
        stats.mass.scatter_add_(0, idx, wt.double())
        stats.expert_var_sum.scatter_add_(0, idx, (wt * wt).double())
        stats.expert_select_count.scatter_add_(
            0, idx, torch.ones_like(idx, dtype=torch.long)
        )
        stats.expert_max.scatter_reduce_(0, idx, wt, reduce="amax")

    return hook


def _make_stats(
    domain: MoEPruningDomain, model: nn.Module, target: MoETarget, top_k: int
) -> LayerRoutingStats:
    num_experts = domain.num_experts(model, target)
    return LayerRoutingStats(
        mass=torch.zeros(num_experts, dtype=torch.float64),
        expert_var_sum=torch.zeros(num_experts, dtype=torch.float64),
        expert_select_count=torch.zeros(num_experts, dtype=torch.long),
        expert_var=torch.zeros(num_experts, dtype=torch.float32),
        expert_max=torch.zeros(num_experts, dtype=torch.float32),
        num_experts=num_experts,
        top_k=top_k,
    )


def _register_router_hooks(
    model: nn.Module,
    domain: MoEPruningDomain,
    targets: List[MoETarget],
    top_k: int,
):
    return register_router_hooks(
        model, domain, targets, top_k, _make_stats, _make_router_hook
    )


def _finalize_variance(layer_stats: Dict[str, "LayerRoutingStats"]) -> None:
    for st in layer_stats.values():
        observed = st.expert_select_count > 0
        count = st.expert_select_count.double()
        mean_w = torch.where(observed, st.mass / count, torch.zeros_like(st.mass))
        mean_w_sq = torch.where(
            observed, st.expert_var_sum / count, torch.zeros_like(st.expert_var_sum)
        )
        var = (mean_w_sq - mean_w * mean_w).clamp(min=0).float()
        if observed.any() and (~observed).any():
            var[~observed] = var[observed].mean()
            st.expert_max[~observed] = st.expert_max[observed].mean()
        st.expert_var = var


VARIANCE_SCORES = ("cond", "peak", "cvxpeak")


def _variance_regime_score(
    st: "LayerRoutingStats", variance_score: str
) -> torch.Tensor:
    """Per-expert keep score for the variance regime: cond Var(w|sel) / peak max(w) / cvxpeak (product)."""
    if variance_score == "peak":
        return st.expert_max
    if variance_score == "cvxpeak":
        return st.expert_var * st.expert_max
    return st.expert_var


def _dispersion(x: torch.Tensor) -> float:
    """Across-expert coefficient of variation (std/mean); 0 when the mean is non-positive."""
    x = x.float()
    mean = x.mean()
    if mean <= 0:
        return 0.0
    return float(x.std(unbiased=False) / mean)


def _auto_select_score(st: "LayerRoutingStats", variance_score: str):
    """Per-layer auto: mass vs variance by whichever is more discriminative (higher CV)."""
    mass = st.mass.float()
    var = _variance_regime_score(st, variance_score)
    if _dispersion(var) > _dispersion(mass):
        return var, "variance(auto)"
    return mass, "mass(auto)"


def _select_keep_indices(
    idx: int,
    target: MoETarget,
    layer_stats: Dict[str, "LayerRoutingStats"],
    config: PruneConfig,
    prune_ratio: float,
    boundary,
    variance_score: str,
):
    if target.router_path is None or target.router_path not in layer_stats:
        return None, None, 0
    st = layer_stats[target.router_path]
    num_experts = st.num_experts
    if num_experts <= config.min_experts:
        return None, None, num_experts
    var_score = _variance_regime_score(st, variance_score)
    if boundary == "auto":
        score, method_desc = _auto_select_score(st, variance_score)
    elif isinstance(boundary, (list, tuple, set, frozenset)):
        if idx in boundary:
            score, method_desc = var_score, "variance(explicit)"
        else:
            score, method_desc = st.mass.float(), "mass(explicit)"
    elif idx <= boundary:
        score = st.mass.float()
        method_desc = "mass"
    else:
        score = var_score
        method_desc = "variance"
    keep_idx = topk_keep_indices(score, prune_ratio, config.min_experts)
    if len(keep_idx) == num_experts:
        return None, None, num_experts
    return keep_idx, method_desc, num_experts


class MassVarianceMoEPruningMethod(BasePruningMethod):
    """Adaptive mass + variance expert pruning (shallow layers use mass, deep use variance).

    kwargs:
        prune_ratio (float): Fraction of experts to remove. Default 0.50.
        top_k (int): Experts per token. Default: read from the router /
            model config (see MoEPruningDomain.resolve_top_k).
        boundary (int | "auto" | list[int]): fixed split (default 10), "auto", or explicit
                        variance-layer indices; -1 makes every layer use variance.
        variance_score (str): "cond" (default), "peak", or "cvxpeak".
    """

    domain = "moe"
    name = "mass_variance"
    accepted_kwargs = frozenset({"prune_ratio", "top_k", "boundary", "variance_score"})
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
            raise ValueError("MoE mass+variance pruning requires input data.")
        if not isinstance(domain, MoEPruningDomain):
            raise TypeError("Requires MoEPruningDomain.")

        prune_ratio = float(spec.kwargs.get("prune_ratio", 0.50))
        top_k = spec.kwargs.get("top_k")
        top_k = None if top_k is None else int(top_k)
        boundary_raw = spec.kwargs.get("boundary", 10)
        if boundary_raw == "auto" or isinstance(
            boundary_raw, (list, tuple, set, frozenset)
        ):
            boundary = boundary_raw
        else:
            boundary = int(boundary_raw)
        variance_score = str(spec.kwargs.get("variance_score", "cond"))
        if variance_score not in VARIANCE_SCORES:
            raise ValueError(
                f"variance_score must be one of {VARIANCE_SCORES}, got {variance_score!r}."
            )
        if prune_ratio <= 0:
            return

        layer_stats, hooks = _register_router_hooks(model, domain, targets, top_k)
        try:
            num_batches = run_calibration_epoch(model, context)
        finally:
            for h in hooks:
                h.remove()

        if num_batches == 0:
            msg = "Empty dataset. Skipped."
            LOGGER.logw(msg, "amct_prune")
            report.add_warning(msg)
            return

        _finalize_variance(layer_stats)

        def _select(idx, target):
            keep_idx, method_desc, num_experts = _select_keep_indices(
                idx, target, layer_stats, config, prune_ratio, boundary, variance_score
            )
            if keep_idx is None:
                return None, num_experts, None
            message = f"{method_desc}-based prune {num_experts} -> {len(keep_idx)}"
            return keep_idx, num_experts, message

        last_num_experts = prune_moe_layers(
            self.name, model, domain, targets, report, top_k, _select
        )

        if last_num_experts is not None:
            record_prune_size(model, "moe_num_experts", last_num_experts)
