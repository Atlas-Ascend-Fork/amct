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
- "Accuracy" is abstracted via ``evaluator`` -- a ``Callable[[model], float]`` or an
  ``AutoCalibrationEvaluatorBase`` (larger is better).
  * When the user provides an ``evaluator``: baseline = evaluator(original), and each
    candidate's loss = baseline - evaluator(pruned copy). tolerance shares the evaluator's
    units (e.g. accuracy in 0~1 means tolerance=0.02 = "at most 2 points").
  * Otherwise the default *fidelity* metric is used: take the original model's top-1
    predictions on calibration data as reference; quality = top-1 agreement of the pruned
    model with the reference (in [0,1], baseline=1.0). tolerance then means the max
    fraction of predictions allowed to flip.
- Binary-search the candidate ratio grid under the common assumption that loss is
  monotone non-decreasing in ratio, finding the largest ratio with loss <= tolerance;
  if none qualifies, nothing is pruned (applied=False).
- Each trial runs on a ``copy.deepcopy`` of the model; the original is
  unchanged until the final apply.
"""

from __future__ import annotations

import copy
import warnings
from collections import namedtuple
from dataclasses import dataclass, field, fields
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

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
                f"[auto-prune] no acceptable prune ratio found within tolerance {self.tolerance:.3f}"
                f" (tried {len(self.trials)} candidates), model unchanged."
                f" Relax the tolerance or add a smaller prune ratio to ratio_grid."
            )
        verb = "applied" if self.applied else "suggested (not applied)"
        head = (
            f"[auto-prune] {verb} prune_ratio={self.chosen_ratio:.2f}: "
            f"weights {self.params_before:,} → {self.params_after:,} "
            f"(cut {self.weight_reduction * 100:.1f}%), "
            f"quality loss {self.quality_drop:.3f}"
        )
        # tolerance < 0 is the menu / size-budget sentinel (no acceptable-loss bound to report).
        if self.tolerance < 0:
            return head + "."
        return head + f" ≤ tolerance {self.tolerance:.3f}."


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
                f"[budget] {domain.name} domain find_targets failed, prunable mass may be underestimated: "
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
            f"[budget] size budget unreachable: target {target_params:,} is below the reachable floor {floor:,}"
            f" (non-prunable mass {non_prunable:,}, prunable mass {prunable_mass:,}, "
            f"{prunable_mass / params_before * 100 if params_before else 0.0:.2f}% of total params; "
            f"the 'prunable' scope estimates the post-prune retained floor at the most aggressive "
            f"grid ratio {float(max_ratio):.2f})."
            f" Even pruning at the most aggressive feasible ratio cannot meet it; the search continues without raising."
            f" Relax the budget or adjust budget_scope / ratio_grid.",
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
        "pass an explicit evaluator."
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
            "or provide a custom evaluator."
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

    # A plain quality function (Callable[[model], float]) is already in the target form;
    # pass it through untouched. Only object evaluators (with .evaluate) need wrapping.
    if callable(evaluator) and not hasattr(evaluator, "evaluate"):
        return evaluator

    if not hasattr(evaluator, "evaluate") or not callable(evaluator.evaluate):
        raise TypeError(
            "evaluator must be a Callable[[model], float] or expose a callable "
            ".evaluate(model[, iterations]) -> float (e.g. a subclass of amct "
            "AutoCalibrationEvaluatorBase whose evaluate returns a metric)."
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
    model, evaluator, eval_iterations, eval_batches, batch_adapter
):
    """Accuracy source: the given ``evaluator`` -- a ``Callable[[model], float]`` or an
    ``AutoCalibrationEvaluatorBase``-style object with ``.evaluate`` -- else the default
    top-1 fidelity. Returns (quality_fn, baseline)."""
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
                f"[{warn_tag}] ratio={ratio} trial failed (not a normal rejection): {type(exc).__name__}: {exc}",
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


# ---------------------------------------------------------------------------
# Menu mode: config-driven guarded "best-of-menu" search (folds in the former
# guarded variance / recovery HYBRIDs). A method spec may carry a "menu" key =
# a list of (display_name, method_spec); item 0 is always the fallback. At a
# fixed prune_ratio, prune one deepcopy-per-menu-item, evaluate on validation,
# and pick the highest-quality candidate -- but only switch off the item-0
# fallback on a STRICT improvement (so menu mode is never worse than fallback on
# the validation set). The winner is applied in place and returned as a normal
# AutoTuneResult (chosen_ratio = the fixed prune_ratio).
# ---------------------------------------------------------------------------
def _find_menu_domain(base_cfg: dict) -> Optional[str]:
    """Return the domain whose method spec carries a non-empty ``menu`` key, or None.

    Raises if more than one domain carries a menu (one menu per search).
    """
    methods = base_cfg.get("methods", {}) or {}
    menu_domains = []
    for domain, spec in methods.items():
        # Presence of the key (even an explicit empty menu) selects menu mode; an empty
        # menu then raises in _menu_spec rather than silently degrading to ratio search.
        if isinstance(spec, Mapping) and "menu" in spec:
            menu_domains.append(domain)
    if len(menu_domains) > 1:
        raise ValueError(
            f"menu mode supports a single menu-carrying domain, got {menu_domains!r}."
        )
    return menu_domains[0] if menu_domains else None


def _menu_spec(base_cfg: dict, domain: str) -> Tuple[
    List[Tuple[str, Dict[str, Any]]], float, Dict[str, Any]
]:
    """Extract (menu, prune_ratio, common_kwargs) from the menu-carrying domain.

    ``common_kwargs`` are the domain spec's own kwargs (minus prune_ratio), shared
    by every menu item (e.g. top_k for MoE, ridge for recovery).
    """
    spec = dict(base_cfg["methods"][domain])
    menu_raw = list(spec.get("menu", []))
    if not menu_raw:
        raise ValueError("menu must contain at least the fallback item.")
    menu: List[Tuple[str, Dict[str, Any]]] = []
    for item in menu_raw:
        if not (isinstance(item, (tuple, list)) and len(item) == 2):
            raise ValueError(
                "each menu item must be a (display_name, method_spec) pair."
            )
        name, ms = item
        if not (isinstance(ms, Mapping) and "name" in ms):
            raise ValueError(
                f"menu item {name!r} method_spec must be a dict with a 'name' key."
            )
        menu.append((str(name), {"name": str(ms["name"]), "kwargs": dict(ms.get("kwargs", {}))}))
    common = dict(spec.get("kwargs", {}))
    prune_ratio = float(common.pop("prune_ratio", 0.0))
    return menu, prune_ratio, common


def _menu_candidate_config(
    base_cfg: dict, domain: str, item_spec: Dict[str, Any],
    prune_ratio: float, common_kwargs: Dict[str, Any], copy_model: bool,
) -> PruneConfig:
    """Build a candidate config: the menu domain uses ``item_spec`` (common kwargs +
    prune_ratio merged in); every other prunable domain is neutralized to prune_ratio=0
    (early return, never mispruned) so the only variable is the menu item.
    """
    cfg = copy.deepcopy(base_cfg)
    methods = cfg.get("methods", {})
    neutral = {
        "cnn": {"name": "variance_channel", "kwargs": {"prune_ratio": 0.0}},
        "dense": {"name": "low_variance", "kwargs": {"prune_ratio": 0.0}},
        "moe": {"name": "activation_count", "kwargs": {"prune_ratio": 0.0}},
    }
    for other in list(methods):
        if other != domain and other in neutral:
            methods[other] = dict(neutral[other])
    kwargs = dict(common_kwargs)
    kwargs.update(item_spec.get("kwargs", {}))
    kwargs["prune_ratio"] = prune_ratio
    methods[domain] = {"name": item_spec["name"], "kwargs": kwargs}
    cfg["methods"] = methods
    cfg.pop("copy_model", None)
    return PruneConfig(copy_model=copy_model, **cfg)


def _menu_prune_with(
    model, base_cfg, domain, item_spec, prune_ratio, common_kwargs, *,
    data, batch_adapter, copy_model,
):
    cfg = _menu_candidate_config(
        base_cfg, domain, item_spec, prune_ratio, common_kwargs, copy_model
    )
    cfg.verbose = False
    pruner = AutoPruner(cfg)
    pruned = pruner(model, data=data, batch_adapter=batch_adapter)
    return pruned, pruner.last_report


def _menu_trial(
    model, base_cfg, domain, menu, prune_ratio, common_kwargs, quality_fn, params_before,
    *, data, batch_adapter,
) -> Tuple[Dict[str, float], Dict[str, int]]:
    """Prune one deepcopy per menu item and score on validation. A failed item is
    recorded as -inf with a warning, without affecting others or raising.
    Returns (name -> quality, name -> params_after).
    """
    quality: Dict[str, float] = {}
    pa_by_name: Dict[str, int] = {}
    for name, item_spec in menu:
        try:
            pruned, _ = _menu_prune_with(
                model, base_cfg, domain, item_spec, prune_ratio, common_kwargs,
                data=data, batch_adapter=batch_adapter, copy_model=True,
            )
            quality[name] = float(quality_fn(pruned))
            pa_by_name[name] = count_parameters(pruned)
            del pruned
        except Exception as exc:  # never silent: record -inf and warn, keep others
            warnings.warn(
                f"[menu-prune] variant {name!r} trial failed (not a normal rejection): "
                f"{type(exc).__name__}: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            quality[name] = float("-inf")
            pa_by_name[name] = params_before
    return quality, pa_by_name


def _menu_select(quality: Dict[str, float], fallback_name: str) -> Tuple[str, float, float]:
    """Guarded selection: take the highest validation quality, but only switch off the
    item-0 fallback on a STRICT improvement, else keep the fallback.
    Returns (chosen, fallback_quality, gain_over_fallback).
    """
    if fallback_name not in quality:
        raise RuntimeError("fallback item missing from menu evaluation.")
    fb_q = quality[fallback_name]
    chosen = max(quality, key=lambda n: quality[n])
    if not (quality[chosen] > fb_q):
        chosen = fallback_name
    gain = 0.0 if chosen == fallback_name else quality[chosen] - fb_q
    return chosen, fb_q, gain


def _menu_apply(
    model, base_cfg, domain, menu, chosen, prune_ratio, common_kwargs,
    pa_by_name, params_before, apply, *, data, batch_adapter,
) -> Tuple[Optional[PruneReport], int]:
    """If ``apply=True``, prune ``model`` in place with the chosen item and read
    params_after; otherwise reuse the trial count. Returns (report, params_after).
    """
    if not apply:
        return None, pa_by_name.get(chosen, params_before)
    item_spec = dict(menu)[chosen]
    pruned, report = _menu_prune_with(
        model, base_cfg, domain, item_spec, prune_ratio, common_kwargs,
        data=data, batch_adapter=batch_adapter, copy_model=False,
    )
    if pruned is not model:
        model.__dict__.update(pruned.__dict__)
    return report, count_parameters(model)


def _run_menu_mode(
    model, base_cfg, *, data, evaluator, eval_iterations, eval_batches,
    batch_adapter, apply,
) -> AutoTuneResult:
    """Config-driven guarded best-of-menu search returning a normal AutoTuneResult.

    chosen_ratio = the fixed prune_ratio; quality_drop = baseline - winner_quality;
    params / weight_reduction / applied come from the winner. The chosen variant name
    is recorded into the returned report (a StageEvent), also reachable via the prune(report=) sink.
    """
    domain = _find_menu_domain(base_cfg)
    menu, prune_ratio, common_kwargs = _menu_spec(base_cfg, domain)
    if not (0.0 <= prune_ratio < 1.0):
        raise ValueError(f"prune_ratio must be in [0.0, 1.0). Got {prune_ratio}.")
    fallback_name = menu[0][0]

    quality_fn, baseline_quality = _resolve_quality(
        model, evaluator, eval_iterations, eval_batches, batch_adapter
    )
    params_before = count_parameters(model)
    quality, pa_by_name = _menu_trial(
        model, base_cfg, domain, menu, prune_ratio, common_kwargs,
        quality_fn, params_before, data=data, batch_adapter=batch_adapter,
    )
    chosen, _fb_q, _gain = _menu_select(quality, fallback_name)
    report, params_after = _menu_apply(
        model, base_cfg, domain, menu, chosen, prune_ratio, common_kwargs,
        pa_by_name, params_before, apply, data=data, batch_adapter=batch_adapter,
    )
    if report is not None:
        report.add(
            "menu-select", domain, domain,
            f"chosen variant {chosen!r} (fallback {fallback_name!r}) at "
            f"prune_ratio={prune_ratio:.4f}",
        )
    winner_quality = quality[chosen]
    trials = [
        TrialResult(
            prune_ratio=prune_ratio,
            quality=quality[n],
            quality_drop=baseline_quality - quality[n],
            params_after=pa_by_name.get(n, params_before),
            weight_reduction=(1.0 - pa_by_name.get(n, params_before) / params_before)
            if params_before else 0.0,
            accepted=(n == chosen),
        )
        for n, _ in menu
    ]
    return AutoTuneResult(
        chosen_ratio=prune_ratio,
        tolerance=-1.0,
        baseline_quality=baseline_quality,
        pruned_quality=winner_quality,
        quality_drop=baseline_quality - winner_quality,
        params_before=params_before,
        params_after=params_after,
        weight_reduction=(1.0 - params_after / params_before) if params_before else 0.0,
        applied=apply,
        report=report,
        trials=trials,
    )


def accuracy_based_auto_prune(
    model: nn.Module,
    config=None,
    *,
    data: Any = None,
    tolerance: float = DEFAULT_TOLERANCE,
    evaluator: Optional[Union[QualityFn, Any]] = None,
    eval_iterations: int = 1,
    eval_data: Optional[Sequence[Any]] = None,
    ratio_grid: Sequence[float] = DEFAULT_RATIO_GRID,
    batch_adapter: Optional[BatchAdapter] = None,
    safe_skip_attention: bool = True,
    finetune_fn: Optional[Callable[[nn.Module], Any]] = None,
    quant_fn: Optional[Callable[[nn.Module], Any]] = None,
    apply: bool = True,
) -> AutoTuneResult:
    """
    Function: Tolerance-driven pruning -- search the grid for the largest prune ratio whose accuracy loss
    stays within tolerance, and apply it in place
    Parameter: model: model to prune (mutated in place when apply=True)
               config: dict / PruneConfig / None; each method's prune_ratio is overridden by the searched
                   value, None uses the three-domain defaults
               data: calibration data (needed by variance methods); also the default fidelity eval set when
                   eval_data is not given
               tolerance: upper bound on acceptable accuracy loss, in the evaluator's units (top-1 flip
                   fraction under the default fidelity metric); default 0.02
               evaluator: accuracy source -- a Callable[[model], float] OR an
                   AutoCalibrationEvaluatorBase object (shared with quantization); None uses
                   default top-1 fidelity.
               eval_iterations: iteration count passed to evaluator.evaluate
               eval_data: eval batches for the default fidelity metric; falls back to data
               ratio_grid: candidate prune ratios (ascending)
               batch_adapter: maps a data batch to (args, kwargs) for the model forward; None uses sensible defaults
               safe_skip_attention: default True; merges attention submodules (self_attn/attention/attn) into
                   skip_layers so the dense domain does not mistake parallel q/k/v projections for a two-layer FFN
               finetune_fn: optional post-prune recovery callback; when given, each candidate finetunes the
                   pruned copy before evaluation so the search measures accuracy after prune + recovery
                   (one finetune per candidate, slower)
               quant_fn: optional quantization-aware callback applied to the pruned copy before evaluation
                   (after finetune_fn) so the search decides in the quantized state; the returned model
                   is prune-only
               apply: True applies the chosen ratio to model in place; False searches only and leaves the original
    Return: AutoTuneResult carrying chosen_ratio / weight_reduction / quality_drop and summary()

    Binary search over the grid assumes quality loss is monotone non-decreasing in ratio; on non-monotone
    loss the search may be conservative (prune a bit less). Any ratio whose prune or forward fails is marked
    unacceptable and never raises out.
    """
    if not (0.0 <= tolerance):
        raise ValueError("tolerance must be >= 0.")
    # Menu mode: a method spec carrying a "menu" key switches the accuracy-driven entry
    # from ratio search to a guarded best-of-menu search at the config's fixed prune_ratio.
    menu_cfg = _config_to_dict(config)
    if safe_skip_attention:
        # Guard attention the same way the ratio path does, so the dense recovery menu
        # never mistakes parallel q/k/v projections for a two-layer FFN on real transformers.
        menu_cfg = _merge_attention_skips(menu_cfg)
    if _find_menu_domain(menu_cfg) is not None:
        eval_batches = (
            list(eval_data) if eval_data is not None
            else (list(data) if data is not None else [])
        )
        return _run_menu_mode(
            model, menu_cfg,
            data=data, evaluator=evaluator, eval_iterations=eval_iterations,
            eval_batches=eval_batches, batch_adapter=batch_adapter, apply=apply,
        )
    grid, base_cfg, params_before, eval_batches = _prepare_search_inputs(
        model, config, safe_skip_attention, ratio_grid, data, eval_data
    )
    # Accuracy: evaluator (Callable or AutoCalibrationEvaluatorBase) or default top-1 fidelity
    quality_fn, baseline_quality = _resolve_quality(
        model, evaluator, eval_iterations, eval_batches, batch_adapter
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
    evaluator: Optional[Union[QualityFn, Any]] = None,
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
    """
    Function: Prune to a size budget -- binary-search the smallest ratio whose param count fits within the
    target, and report that point's (size, accuracy)
    Parameter: model: model to prune (mutated in place when apply=True)
               config: dict / PruneConfig / None; prune config, None uses the three-domain defaults
               data: calibration data; also the default fidelity eval set when eval_data is not given
               target_keep_ratio: fraction of params to keep (0.7 = keep 70% = cut 30%); use either this
                   or target_params
               target_params: direct upper bound on the target param count
               evaluator: accuracy source -- a Callable[[model], float] OR an
                   AutoCalibrationEvaluatorBase object (shared with quantization); None uses
                   default top-1 fidelity.
               eval_iterations: iteration count passed to evaluator.evaluate
               eval_data: eval batches for the default fidelity metric; falls back to data
               ratio_grid: candidate prune ratios (ascending)
               batch_adapter: maps a data batch to (args, kwargs) for the model forward
               safe_skip_attention: default True; merges attention submodules into skip_layers to protect
                   parallel q/k/v projections
               finetune_fn: optional post-prune recovery callback run on each pruned copy before evaluation
               quant_fn: optional quantization-aware callback applied before evaluation so the search decides
                   in the quantized state
               apply: True applies the chosen ratio in place; False searches only
               budget_scope: 'total' (default, budget over all params) or 'prunable' (budget over the prunable
                   mass only, excluding embeddings etc.); if the budget is below the non-prunable-mass floor
                   a warning is emitted and budget_unreachable=True is recorded, no exception
    Return: AutoTuneResult; weight_reduction is the actual cut, quality_drop the measured accuracy loss at that
        size, tolerance is -1 (non-tolerance mode), chosen_ratio is None when no ratio meets the budget

    Complements accuracy_based_auto_prune (accuracy-tolerance driven): the "budget = target size" mode.
    """
    (grid, base_cfg, params_before, eval_batches, target_params, unreachable,
     prunable_fraction) = _resolve_budget_config(
        model, config, safe_skip_attention, ratio_grid, data, eval_data,
        target_keep_ratio, target_params, budget_scope,
    )
    quality_fn, baseline_quality = _resolve_quality(
        model, evaluator, eval_iterations, eval_batches, batch_adapter
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
