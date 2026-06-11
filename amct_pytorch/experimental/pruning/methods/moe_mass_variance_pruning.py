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
MoE Mass + Variance Expert Pruning.

Per-layer adaptive expert selection:
- Shallow layers (idx <= boundary): mass-based selection (activation frequency)
- Deep layers (idx > boundary): variance-based selection (per-expert routing weight variance)

Variance metric: for each expert i, collect all routing weights assigned to
expert i across all tokens where i is in the top-k. Compute Var(w) = E[w^2] - E[w]^2.
High variance = the expert receives very different weights depending on context
= specialized/important. Low variance = always gets similar weight = interchangeable.

Experts never selected during calibration are assigned the global mean variance
so they are not automatically pruned (they may be rare but critical).

Boundary between the mass and variance regimes:
- ``boundary=<int>`` (default 10): fixed split — layers idx <= boundary use mass, idx > boundary
  use variance. boundary=10 was validated on LLaDA2.1-mini.
- ``boundary="auto"`` (opt-in): per-layer, pick whichever signal is more *discriminative* for
  top-k selection (larger scale-free across-expert dispersion / coefficient of variation). Removes
  the fixed depth split (which silently never reaches variance on shallow models, L <= boundary).
  NOTE: on deep toy MoEs this did NOT improve max-cut and the CV proxy agreed poorly with a per-layer
  oracle — kept opt-in pending validation on a real MoE (see prune_bench/moe_split/FINDINGS_moe_split.md).
- ``boundary=list[int]`` (opt-in): use variance on exactly those layer indices, mass elsewhere.
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
    topk_keep_indices,
)
from ._moe_mass_common import prune_one_layer
from .base import BasePruningMethod


@dataclass(slots=True)
class LayerRoutingStats:
    """Per-layer routing statistics collected during one epoch."""

    mass: torch.Tensor
    expert_var_sum: torch.Tensor
    expert_select_count: torch.Tensor
    expert_var: torch.Tensor
    expert_max: torch.Tensor
    num_experts: int = 0
    top_k: int = 8


def _make_router_hook(stats: "LayerRoutingStats", k: int):
    def hook(module, inputs, output):
        _ = module
        _ = inputs
        # 结构无关地解析 (topk_idx, topk_wt)：兼容 logits 张量 / 旧式 (idx,wt) /
        # 现代 *TopKRouter 的 (logits, topk_wt, topk_idx) 元组。
        topk_idx, topk_wt = route_topk_from_output(output, stats.num_experts, k)
        if topk_idx is None:
            return

        # 向量化散射累加（替代逐 token 的 .item() 双重循环）：在大模型 / 长校准集
        # 上快几个量级，且为 peak（per-expert 最大路由权重）提供 amax 归约。
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
        # 路由权重非负（softmax/sigmoid 概率），初值 0 的 amax 归约即逐专家峰值权重。
        stats.expert_max.scatter_reduce_(0, idx, wt, reduce="amax")

    return hook


def _register_router_hooks(
    model: nn.Module,
    domain: MoEPruningDomain,
    targets: List[MoETarget],
    top_k: int,
):
    layer_stats: Dict[str, LayerRoutingStats] = {}
    hooks: List[torch.utils.hooks.RemovableHandle] = []
    for target in targets:
        if target.router_path is None:
            continue
        router = get_submodule(model, target.router_path)
        num_experts = domain.num_experts(model, target)
        stats = LayerRoutingStats(
            mass=torch.zeros(num_experts, dtype=torch.float64),
            expert_var_sum=torch.zeros(num_experts, dtype=torch.float64),
            expert_select_count=torch.zeros(num_experts, dtype=torch.long),
            expert_var=torch.zeros(num_experts, dtype=torch.float32),
            expert_max=torch.zeros(num_experts, dtype=torch.float32),
            num_experts=num_experts,
            top_k=top_k,
        )
        layer_stats[target.router_path] = stats
        hooks.append(router.register_forward_hook(_make_router_hook(stats, top_k)))
    return layer_stats, hooks


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
            # 未被选中的专家也给全局均值峰值，避免（可能稀有但关键）被自动剪掉。
            st.expert_max[~observed] = st.expert_max[observed].mean()
        st.expert_var = var


#: 方差族打分模式 —— variance_score kwarg 的取值。
VARIANCE_SCORES = ("cond", "peak", "cvxpeak")


def _variance_regime_score(
    st: "LayerRoutingStats", variance_score: str
) -> torch.Tensor:
    """方差区间的逐专家保留打分（越大越保留）。

    - ``cond``（默认，向后兼容）：Var(w | 被选中) —— 专家收到的路由权重的条件方差。
    - ``peak``：max(w) —— 专家收到过的峰值路由权重。修正 cond 的盲点：始终强权重的
      专家条件方差≈0 会被 cond 误剪，peak 把它们识别为重要。
    - ``cvxpeak``：Var(w|sel) × max(w) —— 方差与峰值的乘积（仍属"方差"分支，二者兼顾）。
    """
    if variance_score == "peak":
        return st.expert_max
    if variance_score == "cvxpeak":
        return st.expert_var * st.expert_max
    return st.expert_var


def _dispersion(x: torch.Tensor) -> float:
    """Scale-free across-expert dispersion (coefficient of variation = std/mean).

    Higher = the score separates experts more = more discriminative for top-k selection.
    Returns 0 when the mean is non-positive (uninformative signal).
    """
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


def _prune_layers(
    method_name: str,
    model: nn.Module,
    domain: MoEPruningDomain,
    targets: List[MoETarget],
    layer_stats: Dict[str, "LayerRoutingStats"],
    config: PruneConfig,
    report: PruneReport,
    prune_ratio: float,
    top_k: int,
    boundary: int,
    variance_score: str,
) -> Optional[int]:
    last_num_experts: Optional[int] = None
    for idx, target in enumerate(targets):
        keep_idx, method_desc, num_experts = _select_keep_indices(
            idx, target, layer_stats, config, prune_ratio, boundary, variance_score
        )
        if keep_idx is None:
            continue
        prune_one_layer(model, domain, target, keep_idx, top_k)
        report.add(
            domain.name,
            method_name,
            target.module_path or target.experts_path,
            f"{method_desc}-based prune {num_experts} -> {len(keep_idx)}",
        )
        report.record_layer_sparsity(
            target.module_path or target.experts_path, num_experts, len(keep_idx)
        )
        last_num_experts = len(keep_idx)
    return last_num_experts


class MassVarianceMoEPruningMethod(BasePruningMethod):
    """
    Adaptive mass + variance expert pruning.

    Shallow layers use mass (frequency), deep layers use variance
    (per-expert routing weight variance after softmax).

    kwargs:
        prune_ratio (float): Fraction of experts to remove. Default 0.50.
        top_k (int): Number of experts per token. Default 8.
        boundary (int | "auto" | list[int]): default int 10 — fixed split, layers <= boundary
                        use mass, the rest use variance. "auto" (opt-in) picks mass vs variance
                        per layer by which is more discriminative (coefficient of variation).
                        A list/set of layer indices (opt-in) uses variance on exactly those
                        layers (mass elsewhere) — explicit per-layer control. ``boundary=-1``
                        makes every layer use the variance score.
        variance_score (str): which per-expert score the variance regime uses —
                        "cond" (default, Var(w|sel); backward-compatible), "peak" (max routing
                        weight; keeps consistently-strong specialists cond misses), or "cvxpeak"
                        (Var(w|sel) x max(w)). Has no effect on mass-regime layers.
    """

    domain = "moe"
    name = "mass_variance"
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
        top_k = int(spec.kwargs.get("top_k", 8))
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
            warnings.warn(msg, RuntimeWarning, stacklevel=2)
            report.add_warning(msg)
            return

        _finalize_variance(layer_stats)
        last_num_experts = _prune_layers(
            self.name,
            model,
            domain,
            targets,
            layer_stats,
            config,
            report,
            prune_ratio,
            top_k,
            boundary,
            variance_score,
        )

        if last_num_experts is not None:
            meta = getattr(model, "_amct_prune_meta", {})
            meta["moe_num_experts"] = int(last_num_experts)
            setattr(model, "_amct_prune_meta", meta)
