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
"""Guarded recovery-HYBRID FFN / CNN pruning.

Like :func:`guarded_variance_hybrid_prune` but searches the post-prune compensation
method (recovery) rather than the neuron-selection criterion. Fixes the saliency
criterion, prunes one copy per recovery in the MENU
{none (fallback), bias, ls}, and picks the best on a small validation set. Only
switches off fallback(none) on strict validation improvement, so it is never worse
than the no-compensation baseline on validation.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from torch import nn

from .accuracy_based_auto_prune import QualityFn, _resolve_quality
from .config import PruneConfig
from .context import BatchAdapter
from .pruner import AutoPruner
from .report import PruneReport
from .utils import count_parameters

#: Default recovery MENU; item 0 "none" (naive slice) is the fallback.
DEFAULT_RECOVERY_MENU: Tuple[str, ...] = ("none", "bias", "ls")

#: Supported prune domain -> its reconstruct method name (dense FFN / CNN channels).
_DOMAIN_METHOD = {"dense": "reconstruct", "cnn": "reconstruct"}


@dataclass(slots=True)
class RecoveryHybridResult:
    """Result of guarded recovery-HYBRID; ``applied`` flags in-place model change."""

    chosen_recovery: str
    fallback_recovery: str
    recovery_quality: Dict[str, float]
    chosen_quality: float
    fallback_quality: float
    gain_over_fallback: float
    prune_ratio: float
    domain: str
    params_before: int
    params_after: int
    weight_reduction: float
    applied: bool
    report: Optional[PruneReport]
    trials: List[Tuple[str, float]] = field(default_factory=list)

    def summary(self) -> str:
        verb = "已应用" if self.applied else "建议（未应用）"
        ranking = ", ".join(
            f"{name}={q:.4f}"
            for name, q in sorted(
                self.recovery_quality.items(), key=lambda kv: kv[1], reverse=True
            )
        )
        return (
            f"[recovery-hybrid:{self.domain}] {verb} recovery={self.chosen_recovery}"
            f"（fallback={self.fallback_recovery}，验证增益 {self.gain_over_fallback:+.4f}）："
            f"剪枝率={self.prune_ratio:.2f}，权重 {self.params_before:,} → {self.params_after:,}"
            f"（削减 {self.weight_reduction * 100:.1f}%）。 验证质量 [{ranking}]。"
        )


def _candidate_config(
    domain: str, recovery: str, prune_ratio: float, ridge: float, copy_model: bool
) -> PruneConfig:
    """Use reconstruct(recovery=...) only on the target domain; other prunable
    domains set to prune_ratio=0 (early return, never mispruned).
    """
    base = {
        "dense": {"name": "low_variance", "kwargs": {"prune_ratio": 0.0}},
        "cnn": {"name": "variance_channel", "kwargs": {"prune_ratio": 0.0}},
        "moe": {"name": "activation_count", "kwargs": {"prune_ratio": 0.0}},
    }
    base[domain] = {
        "name": _DOMAIN_METHOD[domain],
        "kwargs": {"prune_ratio": prune_ratio, "recovery": recovery, "ridge": ridge},
    }
    return PruneConfig(
        methods=base,
        copy_model=copy_model,
        verbose=False,
        missing_data_policy="warn_skip",
    )


def _prune_with(
    model, domain, recovery, prune_ratio, ridge, *, data, batch_adapter, copy_model
):
    cfg = _candidate_config(domain, recovery, prune_ratio, ridge, copy_model)
    pruner = AutoPruner(cfg)
    pruned = pruner(model, data=data, batch_adapter=batch_adapter)
    return pruned, pruner.last_report


def _trial_menu(
    model: nn.Module,
    menu: List[str],
    domain: str,
    prune_ratio: float,
    ridge: float,
    quality_fn: QualityFn,
    params_before: int,
    *,
    data: Any,
    batch_adapter: Optional[BatchAdapter],
) -> Tuple[Dict[str, float], Dict[str, int]]:
    """Prune one copy per recovery and score on validation. Failed entries are
    recorded as -inf with a warning, without affecting others or raising.
    Returns (recovery -> quality, recovery -> params_after).
    """
    quality: Dict[str, float] = {}
    pa_by_rec: Dict[str, int] = {}
    for rec in menu:
        try:
            pruned, _ = _prune_with(
                model,
                domain,
                rec,
                prune_ratio,
                ridge,
                data=data,
                batch_adapter=batch_adapter,
                copy_model=True,
            )
            quality[rec] = float(quality_fn(pruned))
            pa_by_rec[rec] = count_parameters(pruned)
            del pruned
        except Exception as exc:
            warnings.warn(
                f"[recovery-hybrid] recovery {rec!r} 试探失败（非正常拒绝）："
                f"{type(exc).__name__}: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            quality[rec] = float("-inf")
            pa_by_rec[rec] = params_before
    return quality, pa_by_rec


def _finalize(
    model: nn.Module,
    domain: str,
    quality: Dict[str, float],
    pa_by_rec: Dict[str, int],
    fallback: str,
    prune_ratio: float,
    ridge: float,
    params_before: int,
    apply: bool,
    *,
    data: Any,
    batch_adapter: Optional[BatchAdapter],
) -> RecoveryHybridResult:
    """Guarded selection -> apply -> assemble :class:`RecoveryHybridResult`."""
    chosen, fb_q, gain = _select_chosen(quality, fallback)
    report, params_after = _apply_chosen(
        model,
        domain,
        chosen,
        prune_ratio,
        ridge,
        pa_by_rec,
        params_before,
        apply,
        data=data,
        batch_adapter=batch_adapter,
    )
    return RecoveryHybridResult(
        chosen_recovery=chosen,
        fallback_recovery=fallback,
        recovery_quality=quality,
        chosen_quality=quality[chosen],
        fallback_quality=fb_q,
        gain_over_fallback=gain,
        prune_ratio=prune_ratio,
        domain=domain,
        params_before=params_before,
        params_after=params_after,
        weight_reduction=(1.0 - params_after / params_before) if params_before else 0.0,
        applied=apply,
        report=report,
        trials=sorted(quality.items(), key=lambda kv: kv[1], reverse=True),
    )


def _select_chosen(
    quality: Dict[str, float], fallback: str
) -> Tuple[str, float, float]:
    """Guarded selection: take the highest validation quality, but only switch when
    strictly better than fallback(none), else keep fallback.
    Returns (chosen, fallback_quality, gain_over_fallback).
    """
    fb_q = quality[fallback]
    chosen = max(quality, key=lambda r: quality[r])
    if not (quality[chosen] > fb_q):
        chosen = fallback
    gain = 0.0 if chosen == fallback else quality[chosen] - fb_q
    return chosen, fb_q, gain


def _apply_chosen(
    model: nn.Module,
    domain: str,
    chosen: str,
    prune_ratio: float,
    ridge: float,
    pa_by_rec: Dict[str, int],
    params_before: int,
    apply: bool,
    *,
    data: Any,
    batch_adapter: Optional[BatchAdapter],
) -> Tuple[Optional[PruneReport], int]:
    """If ``apply=True``, prune ``model`` in place with the chosen recovery and read
    params_after; otherwise reuse the trial count. Returns (report, params_after).
    """
    report = None
    params_after = params_before
    if apply:
        _, report = _prune_with(
            model,
            domain,
            chosen,
            prune_ratio,
            ridge,
            data=data,
            batch_adapter=batch_adapter,
            copy_model=False,
        )
        params_after = count_parameters(model)
    else:
        params_after = pa_by_rec.get(chosen, params_before)
    return report, params_after


def guarded_recovery_hybrid_prune(
    model: nn.Module,
    *,
    data: Any = None,
    prune_ratio: float = 0.50,
    domain: str = "dense",
    recovery_menu: Sequence[str] = DEFAULT_RECOVERY_MENU,
    ridge: float = 1e-2,
    eval_data: Optional[Sequence[Any]] = None,
    quality_fn: Optional[QualityFn] = None,
    evaluator: Optional[Any] = None,
    eval_iterations: int = 1,
    batch_adapter: Optional[BatchAdapter] = None,
    apply: bool = True,
) -> RecoveryHybridResult:
    """Guarded recovery-HYBRID: prune one copy per recovery in the MENU, select the
    best on a validation set, then apply the best compensation in place.

    Parameters
    ----------
    model : nn.Module
        Model with dense FFN (or CNN sequential conv); modified in place when
        ``apply=True``.
    data : Any
        Calibration data (one run per recovery); also the default validation set
        when ``eval_data`` is not given.
    prune_ratio : float
        Intermediate neuron / channel prune ratio (same ratio and saliency criterion
        for all recoveries; only the compensation differs).
    domain : str
        "dense" (FFN) or "cnn" (sequential conv channels).
    recovery_menu : Sequence[str]
        Candidate compensations. Default ``("none", "bias", "ls")``; item 0 is the
        fallback. For test-side robustness under small calibration, pass
        ``("none", "ls")`` to drop bias (bias folding overfits most under small calib).
    quality_fn / evaluator / eval_data
        Validation quality source (higher is better). None uses default top-1
        fidelity. Priority: quality_fn > evaluator > fidelity.
    apply : bool
        True applies the chosen recovery to ``model`` in place.

    Notes
    -----
    The guard selects on the validation set only and switches off fallback(none) only
    on strict improvement, so it is never worse than the no-compensation baseline on
    validation. A failing recovery is recorded as -inf and warned, without affecting
    others.
    """
    if not (0.0 <= prune_ratio < 1.0):
        raise ValueError(f"prune_ratio must be in [0.0, 1.0). Got {prune_ratio}.")
    if domain not in _DOMAIN_METHOD:
        raise ValueError(
            f"domain must be one of {tuple(_DOMAIN_METHOD)}, got {domain!r}."
        )
    menu = list(dict.fromkeys(recovery_menu))  # de-dup, keep order
    if not menu:
        raise ValueError("recovery_menu must contain at least the fallback recovery.")
    fallback = menu[0]

    eval_batches = (
        list(eval_data)
        if eval_data is not None
        else (list(data) if data is not None else [])
    )
    quality_fn, _ = _resolve_quality(
        model, quality_fn, evaluator, eval_iterations, eval_batches, batch_adapter
    )

    params_before = count_parameters(model)
    quality, pa_by_rec = _trial_menu(
        model,
        menu,
        domain,
        prune_ratio,
        ridge,
        quality_fn,
        params_before,
        data=data,
        batch_adapter=batch_adapter,
    )

    return _finalize(
        model,
        domain,
        quality,
        pa_by_rec,
        fallback,
        prune_ratio,
        ridge,
        params_before,
        apply,
        data=data,
        batch_adapter=batch_adapter,
    )
