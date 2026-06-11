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
"""Guarded variance-HYBRID MoE expert pruning.

Calibrate once, prune one copy per criterion in a MENU, pick the best on a small
validation set, and apply it. Fallback (mass) is always in the MENU and is only
replaced when a variance criterion is strictly better on validation, so HYBRID is
never worse than fallback on the validation set.
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

#: Default criterion MENU: (display name, MoE MethodSpec). Item 0 is the fallback
#: (mass). The three variance entries use boundary=-1 so every layer scores by
#: variance (rather than the default depth split).
DEFAULT_MENU: Tuple[Tuple[str, Dict[str, Any]], ...] = (
    ("mass", {"name": "activation_count", "kwargs": {}}),
    (
        "cond_var",
        {"name": "mass_variance", "kwargs": {"boundary": -1, "variance_score": "cond"}},
    ),
    (
        "peak",
        {"name": "mass_variance", "kwargs": {"boundary": -1, "variance_score": "peak"}},
    ),
    (
        "cvxpeak",
        {
            "name": "mass_variance",
            "kwargs": {"boundary": -1, "variance_score": "cvxpeak"},
        },
    ),
)


@dataclass(slots=True)
class HybridResult:
    """Result of guarded variance-HYBRID; ``applied`` flags in-place model change."""

    chosen_criterion: str
    fallback_criterion: str
    #: Per-criterion validation quality (higher is better); failed criteria are -inf.
    criterion_quality: Dict[str, float]
    chosen_quality: float
    fallback_quality: float
    #: Validation gain of chosen criterion over fallback (guard ensures >= 0).
    gain_over_fallback: float
    prune_ratio: float
    params_before: int
    params_after: int
    weight_reduction: float
    applied: bool
    report: Optional[PruneReport]
    trials: List[Tuple[str, float]] = field(default_factory=list)

    def summary(self) -> str:
        verb = "applied" if self.applied else "suggested (not applied)"
        ranking = ", ".join(
            f"{name}={q:.4f}"
            for name, q in sorted(
                self.criterion_quality.items(), key=lambda kv: kv[1], reverse=True
            )
        )
        return (
            f"[variance-hybrid] {verb} criterion={self.chosen_criterion}"
            f" (fallback={self.fallback_criterion}, validation gain {self.gain_over_fallback:+.4f}): "
            f"expert prune_ratio={self.prune_ratio:.2f}, weights {self.params_before:,} → "
            f"{self.params_after:,} (cut {self.weight_reduction * 100:.1f}%)."
            f" Validation quality [{ranking}]."
        )


def _candidate_config(
    spec: Dict[str, Any], prune_ratio: float, top_k: int, copy_model: bool
) -> PruneConfig:
    """Build a MoE-only candidate config: cnn/dense set to prune_ratio=0 (early
    return, never mispruned) so the only variable across criteria is the MoE
    expert-selection signal.
    """
    kwargs = dict(spec.get("kwargs", {}))
    kwargs["prune_ratio"] = prune_ratio
    kwargs["top_k"] = top_k
    return PruneConfig(
        methods={
            "cnn": {"name": "variance_channel", "kwargs": {"prune_ratio": 0.0}},
            "dense": {"name": "low_variance", "kwargs": {"prune_ratio": 0.0}},
            "moe": {"name": str(spec["name"]), "kwargs": kwargs},
        },
        copy_model=copy_model,
        verbose=False,
        missing_data_policy="warn_skip",
    )


def _prune_with(
    model: nn.Module,
    spec: Dict[str, Any],
    prune_ratio: float,
    top_k: int,
    *,
    data: Any,
    batch_adapter: Optional[BatchAdapter],
    copy_model: bool,
) -> Tuple[nn.Module, Optional[PruneReport]]:
    cfg = _candidate_config(spec, prune_ratio, top_k, copy_model)
    pruner = AutoPruner(cfg)
    pruned = pruner(model, data=data, batch_adapter=batch_adapter)
    return pruned, pruner.last_report


def _trial_menu(
    model: nn.Module,
    menu: List[Tuple[str, Dict[str, Any]]],
    prune_ratio: float,
    top_k: int,
    quality_fn: QualityFn,
    params_before: int,
    *,
    data: Any,
    batch_adapter: Optional[BatchAdapter],
) -> Tuple[Dict[str, float], Dict[str, int]]:
    """Prune one copy per criterion and score on validation. Failed criteria are
    recorded as -inf with a warning, without affecting others or raising.
    Returns (criterion -> quality, criterion -> params_after).
    """
    quality: Dict[str, float] = {}
    pa_by_name: Dict[str, int] = {}
    for name, spec in menu:
        try:
            pruned, _ = _prune_with(
                model,
                spec,
                prune_ratio,
                top_k,
                data=data,
                batch_adapter=batch_adapter,
                copy_model=True,
            )
            quality[name] = float(quality_fn(pruned))
            pa_by_name[name] = count_parameters(pruned)
            del pruned
        except Exception as exc:  # never silent: record -inf and warn, keep others
            warnings.warn(
                f"[variance-hybrid] criterion {name!r} trial failed (not a normal rejection): "
                f"{type(exc).__name__}: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            quality[name] = float("-inf")
            pa_by_name[name] = params_before
    return quality, pa_by_name


def _finalize(
    model: nn.Module,
    menu: List[Tuple[str, Dict[str, Any]]],
    quality: Dict[str, float],
    pa_by_name: Dict[str, int],
    fallback_name: str,
    prune_ratio: float,
    top_k: int,
    params_before: int,
    apply: bool,
    *,
    data: Any,
    batch_adapter: Optional[BatchAdapter],
) -> HybridResult:
    """Guarded selection -> apply -> assemble :class:`HybridResult`. Raises
    RuntimeError if the fallback is missing from the evaluation.
    """
    if fallback_name not in quality:
        raise RuntimeError("fallback criterion missing from menu evaluation.")
    chosen, fb_q, gain = _select_chosen(quality, fallback_name)
    report, params_after = _apply_chosen(
        model,
        menu,
        chosen,
        prune_ratio,
        top_k,
        pa_by_name,
        params_before,
        apply,
        data=data,
        batch_adapter=batch_adapter,
    )
    return HybridResult(
        chosen_criterion=chosen,
        fallback_criterion=fallback_name,
        criterion_quality=quality,
        chosen_quality=quality[chosen],
        fallback_quality=fb_q,
        gain_over_fallback=gain,
        prune_ratio=prune_ratio,
        params_before=params_before,
        params_after=params_after,
        weight_reduction=(1.0 - params_after / params_before) if params_before else 0.0,
        applied=apply,
        report=report,
        trials=sorted(quality.items(), key=lambda kv: kv[1], reverse=True),
    )


def _select_chosen(
    quality: Dict[str, float], fallback_name: str
) -> Tuple[str, float, float]:
    """Guarded selection: take the highest validation quality, but only switch when
    strictly better than fallback, else keep fallback.
    Returns (chosen, fallback_quality, gain_over_fallback).
    """
    fb_q = quality[fallback_name]
    chosen = max(quality, key=lambda n: quality[n])
    if not (quality[chosen] > fb_q):
        chosen = fallback_name
    gain = 0.0 if chosen == fallback_name else quality[chosen] - fb_q
    return chosen, fb_q, gain


def _apply_chosen(
    model: nn.Module,
    menu: List[Tuple[str, Dict[str, Any]]],
    chosen: str,
    prune_ratio: float,
    top_k: int,
    pa_by_name: Dict[str, int],
    params_before: int,
    apply: bool,
    *,
    data: Any,
    batch_adapter: Optional[BatchAdapter],
) -> Tuple[Optional[PruneReport], int]:
    """If ``apply=True``, prune ``model`` in place with the chosen criterion (via
    __dict__ update) and read params_after; otherwise reuse the trial count.
    Returns (report, params_after).
    """
    report: Optional[PruneReport] = None
    params_after = params_before
    if apply:
        chosen_spec = dict(menu)[chosen]
        pruned, report = _prune_with(
            model,
            chosen_spec,
            prune_ratio,
            top_k,
            data=data,
            batch_adapter=batch_adapter,
            copy_model=False,
        )
        if pruned is not model:
            model.__dict__.update(pruned.__dict__)
        params_after = count_parameters(model)
    else:
        params_after = pa_by_name.get(chosen, params_before)
    return report, params_after


def guarded_variance_hybrid_prune(
    model: nn.Module,
    *,
    data: Any = None,
    prune_ratio: float = 0.50,
    top_k: int = 8,
    eval_data: Optional[Sequence[Any]] = None,
    quality_fn: Optional[QualityFn] = None,
    evaluator: Optional[Any] = None,
    eval_iterations: int = 1,
    batch_adapter: Optional[BatchAdapter] = None,
    menu: Optional[Sequence[Tuple[str, Dict[str, Any]]]] = None,
    apply: bool = True,
) -> HybridResult:
    """
    Function: Guarded variance-HYBRID: calibrate once, evaluate the criterion MENU on a validation set,
              then apply the best criterion in place
    Parameter: model: nn.Module with MoE layers (modified in place when apply=True)
               data: calibration data (one calibration per criterion); also the default validation set
                     when eval_data is not given
               prune_ratio: float fraction of experts removed per layer (same ratio for all criteria)
               top_k: int experts selected per token
               eval_data: Sequence | None validation set for the guard (ideally distinct from data); falls back to data
               quality_fn: Callable[[nn.Module], float] | None validation quality (higher is better); None uses
                           default top-1 fidelity. Priority: quality_fn > evaluator > default fidelity
               evaluator: object | None evaluate(model[, iters]) -> float abstraction shared with the quant side
               eval_iterations: int evaluation iterations passed to the evaluator
               batch_adapter: BatchAdapter | None adapts raw batches for the model
               menu: Sequence[(name, MethodSpec)] | None criterion MENU; None uses DEFAULT_MENU. Item 0 is the
                     fallback and must always be present
               apply: bool, True applies the chosen criterion in place; False only searches
    Return: HybridResult with the chosen criterion, validation qualities, param counts, and the applied flag

    The guard selects on the validation set only and switches off fallback only on
    strict improvement, so HYBRID is never worse than fallback on validation. A
    failing criterion is recorded as -inf and warned, without affecting others.
    """
    if not (0.0 <= prune_ratio < 1.0):
        raise ValueError(f"prune_ratio must be in [0.0, 1.0). Got {prune_ratio}.")
    menu = list(menu) if menu is not None else list(DEFAULT_MENU)
    if not menu:
        raise ValueError("menu must contain at least the fallback criterion.")
    fallback_name = menu[0][0]

    eval_batches = (
        list(eval_data)
        if eval_data is not None
        else (list(data) if data is not None else [])
    )
    quality_fn, _ = _resolve_quality(
        model, quality_fn, evaluator, eval_iterations, eval_batches, batch_adapter
    )

    params_before = count_parameters(model)
    quality, pa_by_name = _trial_menu(
        model,
        menu,
        prune_ratio,
        top_k,
        quality_fn,
        params_before,
        data=data,
        batch_adapter=batch_adapter,
    )

    return _finalize(
        model,
        menu,
        quality,
        pa_by_name,
        fallback_name,
        prune_ratio,
        top_k,
        params_before,
        apply,
        data=data,
        batch_adapter=batch_adapter,
    )
