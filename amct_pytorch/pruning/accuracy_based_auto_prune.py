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
"""Tolerance-driven automatic prune-ratio search: given an acceptable accuracy loss
(tolerance), pick the *largest prune ratio that meets the tolerance* and apply it in place.

Core idea
---------
- "Accuracy" is abstracted as ``quality_fn(model) -> float`` (larger is better).
  * When the user provides ``quality_fn``: baseline = quality_fn(original), and each
    candidate's loss = baseline - quality_fn(pruned copy). tolerance shares quality_fn's
    units (e.g. accuracy in 0~1 means tolerance=0.02 = "at most 2 points").
  * Otherwise the default *fidelity* metric is used: take the original model's top-1
    predictions on calibration data as reference; quality = top-1 agreement of the pruned
    model with the reference (in [0,1], baseline=1.0). tolerance then means the max
    fraction of predictions allowed to flip.
- Binary-search the candidate ratio grid under the common assumption that loss is
  monotone non-decreasing in ratio, finding the largest ratio with loss <= tolerance;
  if none qualifies, nothing is pruned (applied=False).
- Each trial runs on a ``copy.deepcopy`` (AutoPruner(copy_model=True)); the original is
  unchanged until the final apply.
"""

from __future__ import annotations

import copy
import warnings
from collections import namedtuple
from dataclasses import dataclass, field, fields
from typing import Any, Callable, List, Mapping, Optional, Sequence

import torch
from torch import nn

from .config import MethodSpec, PruneConfig
from .context import BatchAdapter
from .pruner import AutoPruner
from .report import PruneReport
from .utils import count_parameters

QualityFn = Callable[[nn.Module], float]

DEFAULT_RATIO_GRID: tuple = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)

#: Default tolerance -- a conservative starting point for users new to pruning (at most 2% quality loss).
DEFAULT_TOLERANCE: float = 0.02

#: Attention submodule name fragments to auto-skip. The dense domain's "adjacent
#: Linear = two-layer FFN" heuristic would misread parallel q/k/v projections as a
#: prunable FFN and break attention; excluding these paths by default lets users prune
#: LLM FFNs safely without knowing the internal structure.
ATTENTION_NAME_HINTS: tuple = ("self_attn", "attention", "attn")

#: Size-budget scope: 'total' = budget over all model params (default);
#: 'prunable' = budget over prunable mass only (embeddings etc. excluded from the denominator).
BUDGET_SCOPES: tuple = ("total", "prunable")


@dataclass(slots=True)
class TrialResult:
    prune_ratio: float
    quality: float
    quality_drop: float
    params_after: int
    weight_reduction: float
    accepted: bool


@dataclass(slots=True)
class AutoTuneResult:
    """Tolerance-search result. ``applied`` indicates whether pruning was applied to the original model in place."""

    chosen_ratio: Optional[float]
    tolerance: float
    baseline_quality: float
    # quality at the chosen ratio (after finetune + optional quant_fn); quantized-state quality
    # when quant_fn is passed, not the returned (prune-only) model
    pruned_quality: Optional[float]
    quality_drop: Optional[float]
    params_before: int
    params_after: int
    weight_reduction: float
    applied: bool
    report: Optional[PruneReport]
    trials: List[TrialResult] = field(default_factory=list)
    #: Budget honesty: True when the size budget is below the non-prunable-mass floor
    #: (unreachable even after pruning every target).
    budget_unreachable: bool = False
    #: Prunable mass as a fraction of total params (computed only by budget entries; None otherwise).
    prunable_fraction: Optional[float] = None

    def summary(self) -> str:
        """One-line human-readable summary."""
        if self.chosen_ratio is None:
            return (
                f"[auto-prune] 在容差 {self.tolerance:.3f} 内找不到可接受的剪枝率"
                f"（已试 {len(self.trials)} 个候选），模型未改动。"
                f" 可放宽容差或在 ratio_grid 中加入更小的剪枝率。"
            )
        verb = "已应用" if self.applied else "建议（未应用）"
        return (
            f"[auto-prune] {verb}剪枝率={self.chosen_ratio:.2f}："
            f"权重 {self.params_before:,} → {self.params_after:,} "
            f"(削减 {self.weight_reduction * 100:.1f}%)，"
            f"质量损失 {self.quality_drop:.3f} ≤ 容差 {self.tolerance:.3f}。"
        )


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------
def _config_to_dict(config) -> dict:
    if config is None:
        # Default to all three domains; an unmatched domain is naturally a no-op.
        return {
            "methods": {
                "cnn": {"name": "variance_channel", "kwargs": {}},
                "dense": {"name": "low_variance", "kwargs": {}},
                "moe": {"name": "mass_variance", "kwargs": {"boundary": 10}},
            }
        }
    if isinstance(config, PruneConfig):
        base = {f: getattr(config, f) for f in config.__slots__}
        return copy.deepcopy(base)
    if isinstance(config, Mapping):
        return copy.deepcopy(dict(config))
    raise TypeError(f"config must be dict / PruneConfig / None, got {type(config)!r}")


def _with_ratio(config_dict: dict, ratio: float) -> dict:
    """Return a config copy with every method's prune_ratio overridden to ``ratio``."""
    cfg = copy.deepcopy(config_dict)
    methods = cfg.get("methods", {})
    for name, spec in list(methods.items()):
        if isinstance(spec, MethodSpec):
            spec.kwargs["prune_ratio"] = ratio
        elif isinstance(spec, dict):
            kwargs = dict(spec.get("kwargs", {}))
            kwargs["prune_ratio"] = ratio
            spec["kwargs"] = kwargs
        elif isinstance(spec, str):
            # "name" shorthand -> expand to a dict carrying prune_ratio
            methods[name] = {"name": spec, "kwargs": {"prune_ratio": ratio}}
    cfg["methods"] = methods
    return cfg


def _merge_attention_skips(config_dict: dict) -> dict:
    """Merge attention submodule name fragments into skip_layers to avoid mis-pruning parallel q/k/v projections."""
    cfg = copy.deepcopy(config_dict)
    skips = list(cfg.get("skip_layers", []) or [])
    for hint in ATTENTION_NAME_HINTS:
        if hint not in skips:
            skips.append(hint)
    cfg["skip_layers"] = skips
    return cfg


# ---------------------------------------------------------------------------
# Budget honesty: prunable mass = sum of params in modules hit by the same three-domain find_targets as prune_diagnose
# ---------------------------------------------------------------------------
def _target_module_paths(target) -> List[str]:
    """Collect all module-path fields (``*_path``, non-empty) of a target dataclass."""
    paths: List[str] = []
    for f in fields(target):
        if not f.name.endswith("_path"):
            continue
        value = getattr(target, f.name)
        if value:
            paths.append(str(value))
    return paths


def _dedup_nested_paths(paths) -> List[str]:
    """Drop nested duplicates: if A is an ancestor path of B, keep only A to avoid double-counting params."""
    kept: List[str] = []
    for path in sorted(set(paths)):
        if not any(path == k or path.startswith(k + ".") for k in kept):
            kept.append(path)
    return kept


def _prunable_param_count(model: nn.Module, base_cfg: dict) -> int:
    """Estimate the model's prunable-param mass (upper bound) via the same 3-domain find_targets as prune_diagnose."""
    from .domains.cnn import CNNPruningDomain
    from .domains.dense import DensePruningDomain
    from .domains.moe import MoEPruningDomain
    from .utils import get_submodule

    cfg = PruneConfig(**copy.deepcopy(base_cfg))
    paths: List[str] = []
    for domain in (CNNPruningDomain(), DensePruningDomain(), MoEPruningDomain()):
        try:
            targets = domain.find_targets(model, cfg)
        except Exception as exc:  # not silent: detection failure means prunable mass is underestimated, tell the user
            warnings.warn(
                f"[budget] {domain.name} 域 find_targets 失败，可剪质量可能被低估："
                f"{type(exc).__name__}: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            continue
        for target in targets:
            paths.extend(_target_module_paths(target))
    return sum(count_parameters(get_submodule(model, p)) for p in _dedup_nested_paths(paths))


def _resolve_size_target(
    params_before, prunable_mass, target_keep_ratio, target_params, budget_scope, max_ratio
):
    """Convert the target param count per scope and do an honest reachability check.

    Returns (target_params, unreachable).

    - 'total': target = total params * keep_ratio; reachable floor = non-prunable mass.
    - 'prunable': target = non-prunable mass + prunable mass * keep_ratio (embeddings etc.
      excluded from the denominator); a directly given target_params is read as a prunable-part
      upper bound. The floor mirrors 'total' logic: even at the most aggressive grid ratio
      ``max_ratio`` about (1-max_ratio) of the prunable mass remains, so a lower keep is
      likewise unreachable and must warn rather than stay silent.
    """
    if budget_scope not in BUDGET_SCOPES:
        raise ValueError(f"budget_scope must be one of {BUDGET_SCOPES}, got {budget_scope!r}.")
    if target_params is None and target_keep_ratio is None:
        raise ValueError("give target_keep_ratio (0<r<=1) or target_params.")
    non_prunable = params_before - prunable_mass
    if target_params is None:
        if not (0.0 < float(target_keep_ratio) <= 1.0):
            raise ValueError("target_keep_ratio must be in (0, 1].")
        if budget_scope == "prunable":
            target_params = non_prunable + int(round(prunable_mass * float(target_keep_ratio)))
        else:
            target_params = int(round(params_before * float(target_keep_ratio)))
    elif budget_scope == "prunable":
        target_params = non_prunable + int(target_params)
    # Reachable floor: 'total' = non-prunable mass; 'prunable' also adds the prunable mass
    # still retained at the most aggressive grid ratio.
    floor = non_prunable
    if budget_scope == "prunable":
        floor = non_prunable + int(prunable_mass * (1.0 - float(max_ratio)))
    unreachable = target_params < floor
    if unreachable:
        warnings.warn(
            f"[budget] 尺寸预算不可达：目标 {target_params:,} 低于可达下限 {floor:,}"
            f"（不可剪质量 {non_prunable:,}，可剪质量 {prunable_mass:,}，占总参数 "
            f"{prunable_mass / params_before * 100 if params_before else 0.0:.2f}%；"
            f"'prunable' 口径按最激进网格率 {float(max_ratio):.2f} 估算剪后保留下限）。"
            f"即使按最激进可行率剪也无法达标；将继续搜索，不抛异常。"
            f"可放宽预算或调整 budget_scope / ratio_grid。",
            UserWarning,
            stacklevel=4,
        )
    return target_params, unreachable


def _attach_budget_honesty(result: AutoTuneResult, unreachable: bool, fraction) -> AutoTuneResult:
    """Write budget-honesty fields into the result (and its embedded PruneReport, if any)."""
    result.budget_unreachable = bool(unreachable)
    result.prunable_fraction = fraction
    if result.report is not None:
        result.report.budget_unreachable = bool(unreachable)
        result.report.prunable_fraction = fraction
    return result


# ---------------------------------------------------------------------------
# Default fidelity metric
# ---------------------------------------------------------------------------
def _model_logits(model: nn.Module, batch: Any, batch_adapter: Optional[BatchAdapter]):
    if batch_adapter is not None:
        args, kwargs = batch_adapter(batch)
    elif isinstance(batch, Mapping):
        args, kwargs = (), dict(batch)
    elif isinstance(batch, (tuple, list)):
        args, kwargs = tuple(batch), {}
    else:
        args, kwargs = (batch,), {}
    out = model(*args, **kwargs)
    if hasattr(out, "logits"):
        return out.logits
    if isinstance(out, torch.Tensor):
        return out
    if isinstance(out, (tuple, list)) and out:
        return out[0]
    raise TypeError(
        "Cannot extract logits from model output for fidelity metric; "
        "pass an explicit quality_fn."
    )


def _make_fidelity_quality(
    model: nn.Module,
    eval_batches: Sequence[Any],
    batch_adapter: Optional[BatchAdapter],
) -> QualityFn:
    """Build a top-1 agreement quality function using the original model's top-1 predictions as reference."""
    if not eval_batches:
        raise ValueError(
            "Default fidelity metric needs eval data; pass data=/eval_data= "
            "or provide a custom quality_fn."
        )
    refs = []
    model.eval()
    with torch.no_grad():
        for batch in eval_batches:
            refs.append(_model_logits(model, batch, batch_adapter).argmax(-1))

    def quality(candidate: nn.Module) -> float:
        candidate.eval()
        total = 0
        agree = 0
        with torch.no_grad():
            for batch, ref in zip(eval_batches, refs):
                pred = _model_logits(candidate, batch, batch_adapter).argmax(-1)
                agree += (pred == ref).sum().item()
                total += ref.numel()
        return agree / total if total else 1.0

    return quality


# ---------------------------------------------------------------------------
# Evaluator accuracy abstraction (shared with the quantization side)
# ---------------------------------------------------------------------------
def _make_evaluator_quality(evaluator: Any, eval_iterations: int) -> QualityFn:
    """Wrap any evaluator into ``quality_fn(model) -> float``, sharing the quantization-side abstraction.

    Accepts both ``evaluate`` signatures, so a subclass of :class:`AutoCalibrationEvaluatorBase`
    (the same base used by quantization's ``accuracy_based_auto_calibration``) can be passed
    directly without writing a separate metric:

      * ``evaluate(model)``             -- e.g. an AutoCalibrationEvaluatorBase subclass
      * ``evaluate(model, iterations)`` -- an iteration-aware implementation

    Note: the built-in graph ``ModelEvaluator.evaluate`` is a calibration data feeder and
    returns no accuracy; passing it raises clearly instead of failing silently.
    """
    import inspect

    if not hasattr(evaluator, "evaluate") or not callable(evaluator.evaluate):
        raise TypeError(
            "evaluator must expose a callable .evaluate(model[, iterations]) -> float "
            "(e.g. a subclass of amct AutoCalibrationEvaluatorBase whose evaluate returns a metric)."
        )

    # Decide whether evaluate accepts an iterations param beyond model (bound method excludes self).
    try:
        params = inspect.signature(evaluator.evaluate).parameters
        accepts_iterations = len(params) >= 2 or any(
            p.kind == inspect.Parameter.VAR_POSITIONAL for p in params.values()
        )
    except (TypeError, ValueError):
        accepts_iterations = True  # when the signature is unparseable, call the most compatible way

    def quality(m: nn.Module) -> float:
        score = (
            evaluator.evaluate(m, eval_iterations)
            if accepts_iterations
            else evaluator.evaluate(m)
        )
        if score is None:
            raise TypeError(
                "evaluator.evaluate(...) returned None. The accuracy abstraction must return a "
                "numeric metric (larger = better). The graph-based ModelEvaluator is a calibration "
                "data feeder (returns None) and cannot drive the prune search — implement evaluate() "
                "to return e.g. top-1 accuracy / mAP, as for accuracy_based_auto_calibration."
            )
        return float(score)

    return quality


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------
def _resolve_quality(
    model, quality_fn, evaluator, eval_iterations, eval_batches, batch_adapter
):
    """Accuracy-source priority: quality_fn > evaluator > default top-1 fidelity. Returns (quality_fn, baseline)."""
    if quality_fn is not None:
        return quality_fn, float(quality_fn(model))
    if evaluator is not None:
        qf = _make_evaluator_quality(evaluator, eval_iterations)
        return qf, qf(model)
    qf = _make_fidelity_quality(model, eval_batches, batch_adapter)
    return qf, 1.0


def _prepare_search_inputs(model, config, safe_skip_attention, ratio_grid, data, eval_data):
    """Shared setup for both search entries: validate grid, build base config, count params, collect eval batches."""
    grid = sorted(set(float(r) for r in ratio_grid if 0.0 <= float(r) < 1.0))
    if not grid:
        raise ValueError("ratio_grid must contain at least one ratio in [0.0, 1.0).")
    base_cfg = _config_to_dict(config)
    if safe_skip_attention:
        base_cfg = _merge_attention_skips(base_cfg)
    if eval_data is not None:
        eval_batches = list(eval_data)
    else:
        eval_batches = list(data) if data is not None else []
    return grid, base_cfg, count_parameters(model), eval_batches


def _make_trial_fn(
    model,
    base_cfg,
    params_before,
    *,
    data,
    batch_adapter,
    finetune_fn,
    quant_fn,
    quality_fn,
    baseline_quality,
    accept_fn,
    warn_tag,
):
    """Build the per-ratio trial function (shared by tolerance and size-budget modes).

    Each trial runs on a copy (copy_model=True); the original is untouched.
    - finetune_fn: a (light) finetune of the pruned copy before evaluation, so the search
      measures reachable accuracy "after prune + recovery" rather than the post-prune collapse
      (otherwise one-shot structural pruning crashes near random and the search is over-conservative).
    - quant_fn: applies the target quantization scheme to the pruned copy before evaluation,
      so the search decides in the quantized state.
    - prune/forward errors are not silent: warn clearly and mark unacceptable, never raise out.
    """

    def trial(ratio: float) -> TrialResult:
        cfg = PruneConfig(**_with_ratio(base_cfg, ratio))
        cfg.copy_model = True  # trial on a copy, original untouched
        cfg.verbose = False
        try:
            pruner = AutoPruner(cfg)
            pruned = pruner(model, data=data, batch_adapter=batch_adapter)
            pa = count_parameters(pruned)
            if finetune_fn is not None:
                finetune_fn(pruned)
            if quant_fn is not None:
                quant_fn(pruned)
            q = float(quality_fn(pruned))
        except Exception as exc:
            # Prune/forward at this ratio errored (not a normal rejection). Not silent: avoid
            # disguising an error as a normal rejection, else the user only sees "no feasible
            # ratio" without knowing it was a bug/incompatibility.
            warnings.warn(
                f"[{warn_tag}] ratio={ratio} 试探发生错误（非正常拒绝）：{type(exc).__name__}: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            return TrialResult(ratio, float("-inf"), float("inf"), params_before, 0.0, False)
        drop = baseline_quality - q
        return TrialResult(
            prune_ratio=ratio,
            quality=q,
            quality_drop=drop,
            params_after=pa,
            weight_reduction=1.0 - pa / params_before if params_before else 0.0,
            accepted=accept_fn(drop, pa),
        )

    return trial


def _search_max_ratio(grid, trial_fn, tolerance):
    """Binary-search the ascending grid for the largest ratio with loss <= tolerance.

    Returns (best_idx, ordered_trials, trials).
    """
    trials: dict = {}

    def cached(i):
        if i not in trials:
            trials[i] = trial_fn(grid[i])
        return trials[i]

    lo, hi, best_idx = 0, len(grid) - 1, None
    while lo <= hi:
        mid = (lo + hi) // 2
        if cached(mid).accepted:
            best_idx = mid
            lo = mid + 1  # can go more aggressive
        else:
            hi = mid - 1  # too aggressive, back off
    return best_idx, [trials[i] for i in sorted(trials)], trials


def _search_min_ratio(grid, trial_fn, target_params):
    """Binary-search the ascending grid for the smallest ratio with params <= target (stop once enough).

    Returns (best_idx, ordered, trials).
    """
    trials: dict = {}

    def cached(i):
        if i not in trials:
            trials[i] = trial_fn(grid[i])
        return trials[i]

    lo, hi, best_idx = 0, len(grid) - 1, None
    while lo <= hi:
        mid = (lo + hi) // 2
        if cached(mid).params_after <= target_params:
            best_idx = mid
            hi = mid - 1  # try a smaller ratio (prune less) and see if still within budget
        else:
            lo = mid + 1  # not enough yet, go more aggressive
    return best_idx, [trials[i] for i in sorted(trials)], trials


def _apply_chosen_ratio(
    model, base_cfg, chosen, *, data, batch_adapter, finetune_fn, params_before
):
    """Apply the chosen ratio in place (with recovery finetune matching the search). Returns (report, params_after)."""
    cfg = PruneConfig(**_with_ratio(base_cfg, chosen))
    cfg.copy_model = False
    cfg.verbose = False
    pruner = AutoPruner(cfg)
    pruned = pruner(model, data=data, batch_adapter=batch_adapter)
    if pruned is not model:
        model.__dict__.update(pruned.__dict__)
    if finetune_fn is not None:
        finetune_fn(model)  # recovery finetune matching the search
    return pruner.last_report, count_parameters(model)


def _finalize_autotune(
    model,
    base_cfg,
    grid,
    best_idx,
    trials,
    ordered_trials,
    *,
    tolerance,
    baseline_quality,
    params_before,
    data,
    batch_adapter,
    finetune_fn,
    apply,
):
    """Build the AutoTuneResult after the search (shared by tolerance/size modes).

    best_idx=None means no feasible ratio.
    """
    if best_idx is None:
        return AutoTuneResult(
            chosen_ratio=None,
            tolerance=tolerance,
            baseline_quality=baseline_quality,
            pruned_quality=None,
            quality_drop=None,
            params_before=params_before,
            params_after=params_before,
            weight_reduction=0.0,
            applied=False,
            report=None,
            trials=ordered_trials,
        )
    chosen = grid[best_idx]
    chosen_trial = trials[best_idx]
    report, params_after = None, params_before
    if apply:
        report, params_after = _apply_chosen_ratio(
            model,
            base_cfg,
            chosen,
            data=data,
            batch_adapter=batch_adapter,
            finetune_fn=finetune_fn,
            params_before=params_before,
        )
    return AutoTuneResult(
        chosen_ratio=chosen,
        tolerance=tolerance,
        baseline_quality=baseline_quality,
        pruned_quality=chosen_trial.quality,
        quality_drop=chosen_trial.quality_drop,
        params_before=params_before,
        params_after=params_after if apply else chosen_trial.params_after,
        weight_reduction=(1.0 - (params_after / params_before))
        if apply and params_before
        else chosen_trial.weight_reduction,
        applied=apply,
        report=report,
        trials=ordered_trials,
    )


def accuracy_based_auto_prune(
    model: nn.Module,
    config=None,
    *,
    data: Any = None,
    tolerance: float = DEFAULT_TOLERANCE,
    quality_fn: Optional[QualityFn] = None,
    evaluator: Optional[Any] = None,
    eval_iterations: int = 1,
    eval_data: Optional[Sequence[Any]] = None,
    ratio_grid: Sequence[float] = DEFAULT_RATIO_GRID,
    batch_adapter: Optional[BatchAdapter] = None,
    safe_skip_attention: bool = True,
    finetune_fn: Optional[Callable[[nn.Module], Any]] = None,
    quant_fn: Optional[Callable[[nn.Module], Any]] = None,
    apply: bool = True,
) -> AutoTuneResult:
    """One-click tolerance-driven pruning: given an acceptable accuracy loss, pick the
    *largest prune ratio that meets the tolerance* and apply it in place -- no need to know
    ratios/layer structure.

    Simplest use::

        from amct_pytorch.pruning import accuracy_based_auto_prune
        result = accuracy_based_auto_prune(model, data=calib, tolerance=0.02)
        print(result.summary())

    Parameters
    ----------
    model : nn.Module
        Model to prune (mutated in place when ``apply=True``).
    config : dict | PruneConfig | None
        Prune config; each method's ``prune_ratio`` is overridden by the searched value. None
        uses the three-domain default methods.
    data : Any
        Calibration data (needed by variance methods); also the default fidelity eval set when
        ``eval_data`` is not given.
    tolerance : float
        Upper bound on acceptable accuracy loss, in ``quality_fn`` units (top-1 flip fraction
        under the default fidelity metric). Default ``0.02`` (a conservative 2%).
    quality_fn : Callable[[nn.Module], float] | None
        Accuracy function (larger is better). None uses the default top-1 fidelity metric
        (agreement of the pruned model with the original's predictions), label-free.
    evaluator : object | None
        Shares the same accuracy abstraction as quantization's
        :func:`accuracy_based_auto_calibration`: any object with
        ``evaluate(model, iterations) -> float`` (e.g. ``ModelEvaluator``). When given, it is
        the accuracy source, so one evaluator can drive both prune and quant accuracy searches.
        Priority: ``quality_fn`` > ``evaluator`` > default fidelity.
    eval_iterations : int
        Iteration count passed to ``evaluator.evaluate``.
    eval_data : Sequence | None
        Eval batches for the default fidelity metric; falls back to ``data``.
    ratio_grid : Sequence[float]
        Candidate prune ratios (ascending).
    safe_skip_attention : bool
        Default True: merge attention submodules (self_attn/attention/attn) into skip_layers
        so the dense domain does not mistake parallel q/k/v projections for a two-layer FFN and
        break the model. No effect on pure-CNN / attention-free models.
    finetune_fn : Callable[[nn.Module], Any] | None
        Optional post-prune recovery callback (e.g. ``lambda m: prune_finetune(m, train_data, steps=300)``).
        When given, every candidate ratio finetunes the pruned copy before evaluation, so the
        search measures reachable accuracy "after prune + recovery". This corrects an important
        bias: one-shot structural pruning crashes near random, and without finetune the search
        picks almost no ratio (over-conservative); after finetune high ratios recover and the
        search can pick genuinely usable, more aggressive ratios. Cost: one finetune per candidate (slower).
    quant_fn : Callable[[nn.Module], Any] | None
        Optional quantization-aware callback (e.g. ``lambda m: (amct.quantize(m), amct.convert(m))``).
        When given, every candidate ratio applies the *to-be-deployed quantization scheme* to the
        pruned copy before evaluation (after ``finetune_fn``, in deployment order: prune -> recover
        -> quantize), so the search decides in the quantized state -- the chosen ratio is co-optimized
        with that scheme rather than decided in float then quantized. The returned model is only
        pruned (like ``finetune_fn``, quantization is left to deployment); reported quality loss includes quantization.
    apply : bool
        True applies the chosen ratio to ``model`` in place; False searches only and leaves the original.

    Returns
    -------
    AutoTuneResult
        Carries ``chosen_ratio`` / ``weight_reduction`` / ``quality_drop`` and ``summary()``.

    Notes
    -----
    Binary search over the grid assumes quality loss is monotone non-decreasing in ratio. If
    measured loss is non-monotone, the search may be *conservative* (prune a bit less) -- a safe
    direction for the user. Any ratio whose prune or forward fails is marked unacceptable and
    never raises out.
    """
    if not (0.0 <= tolerance):
        raise ValueError("tolerance must be >= 0.")
    grid, base_cfg, params_before, eval_batches = _prepare_search_inputs(
        model, config, safe_skip_attention, ratio_grid, data, eval_data
    )
    # Accuracy-source priority: quality_fn > evaluator > default top-1 fidelity
    quality_fn, baseline_quality = _resolve_quality(
        model, quality_fn, evaluator, eval_iterations, eval_batches, batch_adapter
    )
    trial = _make_trial_fn(
        model, base_cfg, params_before,
        data=data, batch_adapter=batch_adapter, finetune_fn=finetune_fn, quant_fn=quant_fn,
        quality_fn=quality_fn, baseline_quality=baseline_quality,
        accept_fn=lambda drop, pa: drop <= tolerance, warn_tag="auto-prune",
    )
    # Binary-search the grid for the largest ratio with loss <= tolerance (monotone assumption).
    best_idx, ordered_trials, trials = _search_max_ratio(grid, trial, tolerance)
    return _finalize_autotune(
        model, base_cfg, grid, best_idx, trials, ordered_trials,
        tolerance=tolerance, baseline_quality=baseline_quality, params_before=params_before,
        data=data, batch_adapter=batch_adapter, finetune_fn=finetune_fn, apply=apply,
    )


# ---------------------------------------------------------------------------
# Size-budget driven (the "budget = target size" mode of spec §1)
# ---------------------------------------------------------------------------
# Resolved inputs of the size-budget search (named wrapper to avoid a long tuple return)
_BudgetSetup = namedtuple(
    "_BudgetSetup",
    "grid base_cfg params_before eval_batches target_params unreachable prunable_fraction",
)


def _resolve_budget_config(
    model, config, safe_skip_attention, ratio_grid, data, eval_data,
    target_keep_ratio, target_params, budget_scope,
):
    """Resolve the size-budget entry: search setup + prunable-mass accounting + target conversion and reachability.

    Returns (grid, base_cfg, params_before, eval_batches, target_params, unreachable,
    prunable_fraction). Budget honesty: prunable mass comes from the same three-domain target
    detection as prune_diagnose.
    """
    grid, base_cfg, params_before, eval_batches = _prepare_search_inputs(
        model, config, safe_skip_attention, ratio_grid, data, eval_data
    )
    prunable_mass = _prunable_param_count(model, base_cfg)
    prunable_fraction = prunable_mass / params_before if params_before else 0.0
    target_params, unreachable = _resolve_size_target(
        params_before, prunable_mass, target_keep_ratio, target_params, budget_scope, grid[-1]
    )
    return _BudgetSetup(grid, base_cfg, params_before, eval_batches, target_params, unreachable, prunable_fraction)


def _build_budget_report(
    model, base_cfg, grid, best_idx, trials, ordered_trials, *,
    baseline_quality, params_before, data, batch_adapter, finetune_fn, apply,
    unreachable, prunable_fraction,
):
    """Assemble the size-budget result after the search: tolerance = -1 (non-tolerance mode) + budget-honesty fields."""
    result = _finalize_autotune(
        model,
        base_cfg,
        grid,
        best_idx,
        trials,
        ordered_trials,
        tolerance=-1.0,
        baseline_quality=baseline_quality,
        params_before=params_before,
        data=data,
        batch_adapter=batch_adapter,
        finetune_fn=finetune_fn,
        apply=apply,
    )
    return _attach_budget_honesty(result, unreachable, prunable_fraction)


def size_budget_prune(
    model: nn.Module,
    config=None,
    *,
    data: Optional[Sequence] = None,
    target_keep_ratio: Optional[float] = None,
    target_params: Optional[int] = None,
    quality_fn: Optional[Callable[[nn.Module], float]] = None,
    evaluator: Optional[object] = None,
    eval_iterations: int = 1,
    eval_data: Optional[Sequence] = None,
    ratio_grid: Sequence[float] = DEFAULT_RATIO_GRID,
    batch_adapter: Optional[BatchAdapter] = None,
    safe_skip_attention: bool = True,
    finetune_fn: Optional[Callable[[nn.Module], Any]] = None,
    quant_fn: Optional[Callable[[nn.Module], Any]] = None,
    apply: bool = True,
    budget_scope: str = "total",
) -> AutoTuneResult:
    """Prune to a **size budget**: find the **smallest** ratio fitting params within the target.

    Reports that point's (size, accuracy). Complements :func:`accuracy_based_auto_prune`
    (accuracy-tolerance driven): the "budget = target size" mode of spec §1.

    Parameters
    ----------
    target_keep_ratio : float | None
        Fraction of params to keep (0.7 = keep 70% = cut 30%). Use either this or ``target_params``.
    target_params : int | None
        Direct upper bound on the target param count.
    budget_scope : str
        Budget scope, default ``'total'`` (budget over all model params). ``'prunable'`` budgets
        only the *prunable mass* (sum of params in modules hit by the prune_diagnose three-domain
        detection), excluding non-prunable parts like embeddings from the denominator -- suited to
        small LMs dominated by embeddings. Under either scope, if the budget is below the
        non-prunable-mass floor (unreachable even after pruning every target), a warning is emitted
        and the result gets ``budget_unreachable=True`` with ``prunable_fraction`` recorded; no exception.
    Other parameters are as in :func:`accuracy_based_auto_prune` (quality_fn/evaluator/finetune_fn/quant_fn...).

    Returns
    -------
    AutoTuneResult
        ``weight_reduction`` is the actual cut; ``quality_drop`` is the measured accuracy loss at
        that size; ``tolerance`` is -1 (marks non-tolerance mode). ``chosen_ratio`` is None when no
        ratio meets the budget.
    """
    (grid, base_cfg, params_before, eval_batches, target_params, unreachable,
     prunable_fraction) = _resolve_budget_config(
        model, config, safe_skip_attention, ratio_grid, data, eval_data,
        target_keep_ratio, target_params, budget_scope,
    )
    quality_fn, baseline_quality = _resolve_quality(
        model, quality_fn, evaluator, eval_iterations, eval_batches, batch_adapter
    )
    trial = _make_trial_fn(
        model, base_cfg, params_before,
        data=data, batch_adapter=batch_adapter, finetune_fn=finetune_fn, quant_fn=quant_fn,
        quality_fn=quality_fn, baseline_quality=baseline_quality,
        accept_fn=lambda drop, pa: pa <= target_params,  # here accepted = "meets the size budget"
        warn_tag="size-budget",
    )
    # Monotone assumption: larger ratio -> fewer params. Binary-search the smallest ratio
    # with params <= target (stop once enough).
    best_idx, ordered_trials, trials = _search_min_ratio(grid, trial, target_params)
    return _build_budget_report(
        model, base_cfg, grid, best_idx, trials, ordered_trials,
        baseline_quality=baseline_quality, params_before=params_before,
        data=data, batch_adapter=batch_adapter, finetune_fn=finetune_fn, apply=apply,
        unreachable=unreachable, prunable_fraction=prunable_fraction,
    )


# ---------------------------------------------------------------------------
# Joint accuracy budget (prune -> quantize sequential split)
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class JointBudget:
    """Joint accuracy-budget allocation result for prune+quantize ("split and measure sequentially").

    Usage: feed ``remaining_tolerance`` as the budget bound of a subsequent quantization accuracy
    search (e.g. the evaluator accept threshold of ``accuracy_based_auto_calibration``) so the total
    loss stays within ``total_tolerance``.
    """

    total_tolerance: float
    prune_tolerance: float  # budget bound allotted to pruning
    prune_drop: float  # measured quality loss from pruning
    remaining_tolerance: float  # budget left for quantization = max(0, total - prune_drop)
    prune_result: AutoTuneResult

    def summary(self) -> str:
        return (
            f"[joint-budget] 总预算 {self.total_tolerance:.3f}：剪枝上界 {self.prune_tolerance:.3f}，"
            f"实测用掉 {self.prune_drop:.3f} → 量化剩余预算 {self.remaining_tolerance:.3f}。"
        )


def _joint_budget_honesty(model, config, safe_skip_attention, prune_budget):
    """Prunable-mass accounting for the joint budget (same source as size_budget_prune / prune_diagnose).

    Warns (no exception) when prunable mass is zero (the prune budget can never be spent).
    Returns (unreachable, prunable_fraction).
    """
    honesty_cfg = _config_to_dict(config)
    if safe_skip_attention:
        honesty_cfg = _merge_attention_skips(honesty_cfg)
    params_before = count_parameters(model)
    prunable_mass = _prunable_param_count(model, honesty_cfg)
    prunable_fraction = prunable_mass / params_before if params_before else 0.0
    unreachable = prunable_mass == 0
    if unreachable:
        warnings.warn(
            "[joint-budget] 预算不可达：模型无任何可剪质量（三域均未识别到目标），"
            f"分给剪枝的预算 {prune_budget:.3f} 永远花不出去；"
            "将照常返回（剪枝损失=0，剩余预算全给量化），不抛异常。",
            UserWarning,
            stacklevel=3,
        )
    return unreachable, prunable_fraction


def _split_remaining_budget(res, total_tolerance, prune_tol):
    """Give quantization the remaining budget from the measured prune loss.

    Not pruned (applied=False / no loss) counts as 0.
    """
    spent = res.quality_drop if (res.applied and res.quality_drop is not None) else 0.0
    spent = max(0.0, float(spent))
    return JointBudget(
        total_tolerance=total_tolerance,
        prune_tolerance=prune_tol,
        prune_drop=spent,
        remaining_tolerance=max(0.0, total_tolerance - spent),
        prune_result=res,
    )


def accuracy_based_auto_prune_with_budget(
    model: nn.Module,
    config=None,
    *,
    total_tolerance: float,
    prune_fraction: float = 0.5,
    data: Any = None,
    quality_fn: Optional[QualityFn] = None,
    evaluator: Optional[Any] = None,
    eval_iterations: int = 1,
    eval_data: Optional[Sequence[Any]] = None,
    ratio_grid: Sequence[float] = DEFAULT_RATIO_GRID,
    batch_adapter: Optional[BatchAdapter] = None,
    safe_skip_attention: bool = True,
    finetune_fn: Optional[Callable[[nn.Module], Any]] = None,
    quant_fn: Optional[Callable[[nn.Module], Any]] = None,
    apply: bool = True,
    budget_scope: str = "total",
) -> JointBudget:
    """Split a **joint accuracy budget** between *prune+quantize* and measure sequentially (code form of README flow).

    Steps:
      1. Allot a fraction (``prune_fraction``) of the total budget as the prune tolerance bound;
      2. Call :func:`accuracy_based_auto_prune` to prune in place and **measure** the quality loss ``prune_drop``;
      3. Budget available to quantization = ``max(0, total_tolerance - prune_drop)``.

    Returns :class:`JointBudget`. Feed ``remaining_tolerance`` to quantization-side
    ``accuracy_based_auto_calibration`` (its evaluator accept threshold) so the total accuracy loss
    of *prune-then-quantize* stays within ``total_tolerance`` -- using the same evaluator as quantization.

    Parameters
    ----------
    total_tolerance : float
        Upper bound on the **combined** acceptable accuracy loss of pruning and quantization.
    prune_fraction : float
        Share of the budget allotted to pruning (0~1). Default 0.5 (even split); 1.0 prunes greedily
        and leaves the remainder to quantization.
    budget_scope : str
        Budget-honesty scope (aligned with the same-named param of :func:`size_budget_prune`, default
        ``'total'``). This entry's budget is an accuracy tolerance, independent of scope; the param is
        used for *prunable-mass accounting*: ``prunable_fraction`` is always recorded on
        ``prune_result``, and when the model has no prunable mass (the prune budget can never be spent)
        a warning is emitted and ``budget_unreachable=True`` is set, with no exception.
    Other parameters are passed through to :func:`accuracy_based_auto_prune`.
    """
    if not (0.0 <= total_tolerance):
        raise ValueError("total_tolerance must be >= 0.")
    if not (0.0 <= prune_fraction <= 1.0):
        raise ValueError("prune_fraction must be in [0.0, 1.0].")
    if budget_scope not in BUDGET_SCOPES:
        raise ValueError(f"budget_scope must be one of {BUDGET_SCOPES}, got {budget_scope!r}.")

    prune_tol = total_tolerance * prune_fraction
    # Budget honesty: prunable-mass accounting (same source as size_budget_prune / prune_diagnose)
    unreachable, prunable_fraction = _joint_budget_honesty(
        model, config, safe_skip_attention, prune_tol
    )
    res = accuracy_based_auto_prune(
        model,
        config,
        data=data,
        tolerance=prune_tol,
        quality_fn=quality_fn,
        evaluator=evaluator,
        eval_iterations=eval_iterations,
        eval_data=eval_data,
        ratio_grid=ratio_grid,
        batch_adapter=batch_adapter,
        safe_skip_attention=safe_skip_attention,
        finetune_fn=finetune_fn,
        quant_fn=quant_fn,
        apply=apply,
    )
    _attach_budget_honesty(res, unreachable, prunable_fraction)
    # Measured prune loss: not pruned (applied=False / chosen_ratio=None) counts as 0.
    return _split_remaining_budget(res, total_tolerance, prune_tol)
