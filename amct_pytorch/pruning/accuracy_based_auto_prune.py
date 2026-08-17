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

import copy
from collections import namedtuple
from dataclasses import dataclass, field, fields
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import torch
from torch import nn

from ..common.utils.log import LOGGER

from .config import MethodSpec, PruneConfig, _normalize_method_spec
from .context import BatchAdapter
from .pruner import AutoPruner
from .report import PruneReport
from .utils import count_parameters

QualityFn = Callable[[nn.Module], float]

DEFAULT_RATIO_GRID: tuple = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)

DEFAULT_TOLERANCE: float = 0.02

ATTENTION_NAME_HINTS: tuple = ("self_attn", "attention", "attn")

BUDGET_SCOPES: tuple = ("total", "prunable")


@dataclass
class TrialResult:
    prune_ratio: float
    quality: float
    quality_drop: float
    params_after: int
    weight_reduction: float
    accepted: bool


@dataclass
class AutoTuneResult:
    """Tolerance-search result. ``applied`` indicates whether pruning was applied to the original model in place."""

    chosen_ratio: Optional[float]
    tolerance: float
    baseline_quality: float
    pruned_quality: Optional[float]
    quality_drop: Optional[float]
    params_before: int
    params_after: int
    weight_reduction: float
    applied: bool
    report: Optional[PruneReport]
    trials: List[TrialResult] = field(default_factory=list)
    budget_unreachable: bool = False
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
            f"weights {self.params_before:,} -> {self.params_after:,} "
            f"(cut {self.weight_reduction * 100:.1f}%), "
            f"quality loss {self.quality_drop:.3f}"
        )
        if self.tolerance < 0:
            return head + "."
        return head + f" <= tolerance {self.tolerance:.3f}."


def _config_to_dict(config) -> dict:
    if config is None:
        return {
            "methods": {
                "cnn": {"name": "variance_channel", "kwargs": {}},
                "dense": {"name": "low_variance", "kwargs": {}},
                "moe": {"name": "mass_variance", "kwargs": {"boundary": 10}},
            }
        }
    if isinstance(config, PruneConfig):
        base = {f.name: getattr(config, f.name) for f in fields(config)}
        return copy.deepcopy(base)
    if isinstance(config, Mapping):
        return copy.deepcopy(dict(config))
    raise TypeError(f"config must be dict / PruneConfig / None, got {type(config)!r}")


def _is_pinned_off(spec) -> bool:
    """A domain explicitly set to prune_ratio 0.0 opts out of the search entirely.

    Normalise first: a spec may carry ``prune_ratio`` flat alongside ``name`` instead of
    nested under ``kwargs``, and the flat form is the one that wins.
    """
    if spec is None:
        return False
    try:
        return _normalize_method_spec(spec).kwargs.get("prune_ratio") == 0.0
    except (TypeError, ValueError):
        return False


NEUTRAL_METHODS: dict = {
    "cnn": {"name": "variance_channel", "kwargs": {"prune_ratio": 0.0}},
    "dense": {"name": "low_variance", "kwargs": {"prune_ratio": 0.0}},
    "moe": {"name": "activation_count", "kwargs": {"prune_ratio": 0.0}},
}


def _with_ratio(config_dict: dict, ratio: float) -> dict:
    """Return a config copy with each named method's prune_ratio overridden to ``ratio``.

    Domains pinned to 0.0 keep that value. Domains the config never named are left out
    entirely; ``PruneConfig.resolved_methods`` pins them to 0.0 for every mode.
    """
    cfg = copy.deepcopy(config_dict)
    methods = cfg.get("methods", {})
    for name, spec in list(methods.items()):
        if _is_pinned_off(spec):
            continue
        if isinstance(spec, MethodSpec):
            spec.kwargs["prune_ratio"] = ratio
        elif isinstance(spec, dict):
            kwargs = dict(spec.get("kwargs", {}))
            kwargs["prune_ratio"] = ratio
            # A flat prune_ratio beside "name" outranks the nested one in
            # _normalize_method_spec, so drop it or the override never takes effect.
            spec.pop("prune_ratio", None)
            spec["kwargs"] = kwargs
        elif isinstance(spec, str):
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
    """Estimate the model's prunable-param mass via the three-domain find_targets."""
    from .domains.cnn import CNNPruningDomain
    from .domains.dense import DensePruningDomain
    from .domains.moe import MoEPruningDomain
    from .pruner import _is_skipped
    from .utils import get_submodule

    cfg = PruneConfig(**copy.deepcopy(base_cfg))
    # resolved_methods already pins every domain the config did not name, so this mirrors
    # exactly what the search will prune -- no domain is counted that stays at ratio 0.0.
    resolved = cfg.resolved_methods()
    paths: List[str] = []
    for domain in (CNNPruningDomain(), DensePruningDomain(), MoEPruningDomain()):
        spec = resolved.get(domain.name)
        if spec is None or spec.kwargs.get("prune_ratio") == 0.0:
            continue
        try:
            targets = domain.find_targets(model, cfg)
        except (
            RuntimeError,
            ValueError,
            TypeError,
            AttributeError,
            KeyError,
            IndexError,
        ) as exc:
            LOGGER.logw(
                f"[budget] {domain.name} domain find_targets did not return targets, "
                f"prunable mass may be underestimated: {type(exc).__name__}: {exc}",
                "amct_prune",
            )
            continue
        for target in targets:
            # skip_layers is applied by the pruner, not by find_targets; skipped structures
            # are never pruned, so they are not prunable mass either.
            if _is_skipped(target, cfg.skip_layers):
                continue
            paths.extend(_target_module_paths(target))
    return sum(
        count_parameters(get_submodule(model, p)) for p in _dedup_nested_paths(paths)
    )


def _resolve_size_target(
    params_before,
    prunable_mass,
    target_keep_ratio,
    target_params,
    budget_scope,
    max_ratio,
):
    """Convert the per-scope target param count and check reachability. Returns (target_params, unreachable)."""
    if budget_scope not in BUDGET_SCOPES:
        raise ValueError(
            f"budget_scope must be one of {BUDGET_SCOPES}, got {budget_scope!r}."
        )
    if target_params is None and target_keep_ratio is None:
        raise ValueError("give target_keep_ratio (0<r<=1) or target_params.")
    non_prunable = params_before - prunable_mass
    if target_params is None:
        if not (0.0 < float(target_keep_ratio) <= 1.0):
            raise ValueError("target_keep_ratio must be in (0, 1].")
        if budget_scope == "prunable":
            target_params = non_prunable + int(
                round(prunable_mass * float(target_keep_ratio))
            )
        else:
            target_params = int(round(params_before * float(target_keep_ratio)))
    elif budget_scope == "prunable":
        target_params = non_prunable + int(target_params)
    floor = non_prunable
    if budget_scope == "prunable":
        floor = non_prunable + int(prunable_mass * (1.0 - float(max_ratio)))
    # With no prunable mass the floor collapses onto params_before, so a target below the
    # current size is unreachable even though it never dips under the floor.
    unreachable = target_params < floor or (
        prunable_mass == 0 and target_params < params_before
    )
    if unreachable:
        LOGGER.logw(
            f"[budget] size budget unreachable: target {target_params:,} is below the reachable floor {floor:,}"
            f" (non-prunable mass {non_prunable:,}, prunable mass {prunable_mass:,}, "
            f"{prunable_mass / params_before * 100 if params_before else 0.0:.2f}% of total params; "
            f"the 'prunable' scope estimates the post-prune retained floor at the most aggressive "
            f"grid ratio {float(max_ratio):.2f})."
            f" Even pruning at the most aggressive feasible ratio cannot meet it; the search continues without raising."
            f" Relax the budget or adjust budget_scope / ratio_grid.",
            "amct_prune",
        )
    return target_params, unreachable


def _attach_budget_fields(
    result: AutoTuneResult, unreachable: bool, fraction
) -> AutoTuneResult:
    """Write the budget fields into the result (and its embedded PruneReport, if any)."""
    result.budget_unreachable = bool(unreachable)
    result.prunable_fraction = fraction
    if result.report is not None:
        result.report.budget_unreachable = bool(unreachable)
        result.report.prunable_fraction = fraction
    return result


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


def _make_evaluator_quality(evaluator: Any, eval_iterations: int) -> QualityFn:
    """Wrap any evaluator into ``quality_fn(model) -> float``."""
    import inspect

    if callable(evaluator) and not hasattr(evaluator, "evaluate"):
        return evaluator

    if not hasattr(evaluator, "evaluate") or not callable(evaluator.evaluate):
        raise TypeError(
            "evaluator must be a Callable[[model], float] or expose a callable "
            ".evaluate(model[, iterations]) -> float returning a metric."
        )

    try:
        params = inspect.signature(evaluator.evaluate).parameters
        accepts_iterations = len(params) >= 2 or any(
            p.kind == inspect.Parameter.VAR_POSITIONAL for p in params.values()
        )
    except (TypeError, ValueError):
        accepts_iterations = True

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
                "data feeder (returns None) and cannot drive the prune search -- implement evaluate() "
                "to return e.g. top-1 accuracy / mAP, as for accuracy_based_auto_calibration."
            )
        return float(score)

    return quality


def _resolve_quality(model, evaluator, eval_iterations, eval_batches, batch_adapter):
    """Resolve the accuracy source (evaluator or default top-1 fidelity). Returns (quality_fn, baseline)."""
    if evaluator is not None:
        qf = _make_evaluator_quality(evaluator, eval_iterations)
        return qf, qf(model)
    qf = _make_fidelity_quality(model, eval_batches, batch_adapter)
    return qf, 1.0


def _prepare_search_inputs(
    model, config, safe_skip_attention, ratio_grid, data, eval_data
):
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
    """Build the per-ratio trial function (shared by tolerance and size-budget modes)."""

    def trial(ratio: float) -> TrialResult:
        cfg = PruneConfig(**_with_ratio(base_cfg, ratio))
        cfg.copy_model = True
        try:
            pruner = AutoPruner(cfg)
            pruned = pruner(model, data=data, batch_adapter=batch_adapter)
            pa = count_parameters(pruned)
            if finetune_fn is not None:
                finetune_fn(pruned)
            if quant_fn is not None:
                quant_fn(pruned)
            q = float(quality_fn(pruned))
        except (
            RuntimeError,
            ValueError,
            TypeError,
            AttributeError,
            KeyError,
            IndexError,
        ) as exc:
            LOGGER.logw(
                f"[{warn_tag}] ratio={ratio} trial did not complete (not a normal rejection): "
                f"{type(exc).__name__}: {exc}",
                "amct_prune",
            )
            return TrialResult(
                ratio, float("-inf"), float("inf"), params_before, 0.0, False
            )
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
    """Binary-search the grid for the largest ratio with loss <= tolerance.
    Returns (best_idx, ordered_trials, trials)."""
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
            lo = mid + 1
        else:
            hi = mid - 1
    return best_idx, [trials[i] for i in sorted(trials)], trials


def _search_min_ratio(grid, trial_fn, target_params):
    """Binary-search the grid for the smallest ratio with params <= target. Returns (best_idx, ordered, trials)."""
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
            hi = mid - 1
        else:
            lo = mid + 1
    return best_idx, [trials[i] for i in sorted(trials)], trials


def _apply_chosen_ratio(
    model, base_cfg, chosen, *, data, batch_adapter, finetune_fn, params_before
):
    """Apply the chosen ratio in place (with recovery finetune matching the search). Returns (report, params_after)."""
    cfg = PruneConfig(**_with_ratio(base_cfg, chosen))
    cfg.copy_model = False
    pruner = AutoPruner(cfg)
    pruned = pruner(model, data=data, batch_adapter=batch_adapter)
    if pruned is not model:
        model.__dict__.update(pruned.__dict__)
    if finetune_fn is not None:
        finetune_fn(model)
    return pruner.last_report, count_parameters(model)


def _no_feasible_ratio(
    model,
    ordered_trials,
    *,
    tolerance,
    baseline_quality,
    params_before,
):
    """Result for a search where no candidate ratio qualified: model untouched, warning attached."""
    from .compat import detect_backend

    empty = PruneReport(
        backend=str(detect_backend(model)),
        params_before=params_before,
        params_after=params_before,
    )
    empty.add_warning(
        f"no prune ratio reached the requested size budget across "
        f"{len(ordered_trials)} candidates; model left unchanged. Loosen the budget "
        f"or add a larger prune ratio to ratio_grid."
        if tolerance < 0
        else f"no prune ratio met tolerance {tolerance:.3f} across {len(ordered_trials)} "
        f"candidates; model left unchanged. Relax the tolerance or add a smaller "
        f"prune ratio to ratio_grid."
    )
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
        report=empty,
        trials=ordered_trials,
    )


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
    """Build the AutoTuneResult after the search (best_idx=None means no feasible ratio)."""
    if best_idx is None:
        return _no_feasible_ratio(
            model,
            ordered_trials,
            tolerance=tolerance,
            baseline_quality=baseline_quality,
            params_before=params_before,
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


def _find_menu_domain(base_cfg: dict) -> Optional[str]:
    """Return the single domain whose method spec carries a ``menu`` key, or None."""
    methods = base_cfg.get("methods", {}) or {}
    menu_domains = []
    for domain, spec in methods.items():
        if isinstance(spec, Mapping) and "menu" in spec:
            menu_domains.append(domain)
    if len(menu_domains) > 1:
        raise ValueError(
            f"menu mode supports a single menu-carrying domain, got {menu_domains!r}."
        )
    return menu_domains[0] if menu_domains else None


def _menu_spec(
    base_cfg: dict, domain: str
) -> Tuple[List[Tuple[str, Dict[str, Any]]], float, Dict[str, Any]]:
    """Extract (menu, prune_ratio, common_kwargs) from the menu-carrying domain."""
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
        menu.append(
            (str(name), {"name": str(ms["name"]), "kwargs": dict(ms.get("kwargs", {}))})
        )
    # Normalise so a flat prune_ratio beside "name" is honoured, same as everywhere else.
    common = dict(
        _normalize_method_spec({k: v for k, v in spec.items() if k != "menu"}).kwargs
    )
    prune_ratio = float(common.pop("prune_ratio", 0.0))
    return menu, prune_ratio, common


def _menu_candidate_config(
    base_cfg: dict,
    domain: str,
    item_spec: Dict[str, Any],
    prune_ratio: float,
    common_kwargs: Dict[str, Any],
    copy_model: bool,
) -> PruneConfig:
    """Build a candidate config using ``item_spec`` for the menu domain; other domains neutralized to prune_ratio=0."""
    cfg = copy.deepcopy(base_cfg)
    methods = cfg.get("methods", {})
    # Every other domain is neutralised, including ones the config never listed: an unlisted
    # domain would otherwise fall back to its default ratio and prune structures the menu is
    # not comparing, confounding the candidates it is meant to rank.
    for other, neutral in NEUTRAL_METHODS.items():
        if other != domain:
            methods[other] = dict(neutral)
    kwargs = dict(common_kwargs)
    kwargs.update(item_spec.get("kwargs", {}))
    kwargs["prune_ratio"] = prune_ratio
    methods[domain] = {"name": item_spec["name"], "kwargs": kwargs}
    cfg["methods"] = methods
    cfg.pop("copy_model", None)
    return PruneConfig(copy_model=copy_model, **cfg)


def _menu_prune_with(
    model,
    base_cfg,
    domain,
    item_spec,
    prune_ratio,
    common_kwargs,
    *,
    data,
    batch_adapter,
    copy_model,
):
    cfg = _menu_candidate_config(
        base_cfg, domain, item_spec, prune_ratio, common_kwargs, copy_model
    )
    pruner = AutoPruner(cfg)
    pruned = pruner(model, data=data, batch_adapter=batch_adapter)
    return pruned, pruner.last_report


def _menu_trial(
    model,
    base_cfg,
    domain,
    menu,
    prune_ratio,
    common_kwargs,
    quality_fn,
    params_before,
    *,
    data,
    batch_adapter,
    finetune_fn,
    quant_fn,
) -> Tuple[Dict[str, float], Dict[str, int]]:
    """Prune one copy per menu item and score it. Returns (name -> quality, name -> params_after)."""
    quality: Dict[str, float] = {}
    pa_by_name: Dict[str, int] = {}
    for name, item_spec in menu:
        try:
            pruned, _ = _menu_prune_with(
                model,
                base_cfg,
                domain,
                item_spec,
                prune_ratio,
                common_kwargs,
                data=data,
                batch_adapter=batch_adapter,
                copy_model=True,
            )
            if finetune_fn is not None:
                finetune_fn(pruned)
            if quant_fn is not None:
                quant_fn(pruned)
            quality[name] = float(quality_fn(pruned))
            pa_by_name[name] = count_parameters(pruned)
            del pruned
        except (
            RuntimeError,
            ValueError,
            TypeError,
            AttributeError,
            KeyError,
            IndexError,
        ) as exc:
            LOGGER.logw(
                f"[menu-prune] variant {name!r} trial did not complete (not a normal rejection): "
                f"{type(exc).__name__}: {exc}",
                "amct_prune",
            )
            quality[name] = float("-inf")
            pa_by_name[name] = params_before
    return quality, pa_by_name


def _menu_select(
    quality: Dict[str, float], fallback_name: str
) -> Tuple[str, float, float]:
    """Pick the highest-quality item, switching off the fallback only on a strict improvement.
    Returns (chosen, fallback_quality, gain_over_fallback)."""
    if fallback_name not in quality:
        raise RuntimeError("fallback item missing from menu evaluation.")
    fb_q = quality[fallback_name]
    chosen = max(quality, key=lambda n: quality[n])
    if not (quality[chosen] > fb_q):
        chosen = fallback_name
    gain = 0.0 if chosen == fallback_name else quality[chosen] - fb_q
    return chosen, fb_q, gain


def _menu_apply(
    model,
    base_cfg,
    domain,
    menu,
    chosen,
    prune_ratio,
    common_kwargs,
    pa_by_name,
    params_before,
    apply,
    *,
    data,
    batch_adapter,
    finetune_fn,
) -> Tuple[Optional[PruneReport], int]:
    """Prune ``model`` in place with the chosen item when ``apply``, else reuse the trial count.
    Returns (report, params_after)."""
    if not apply:
        return None, pa_by_name.get(chosen, params_before)
    item_spec = dict(menu)[chosen]
    pruned, report = _menu_prune_with(
        model,
        base_cfg,
        domain,
        item_spec,
        prune_ratio,
        common_kwargs,
        data=data,
        batch_adapter=batch_adapter,
        copy_model=False,
    )
    if pruned is not model:
        model.__dict__.update(pruned.__dict__)
    if finetune_fn is not None:
        finetune_fn(model)
    return report, count_parameters(model)


def _build_menu_trials(
    menu, prune_ratio, quality, baseline_quality, pa_by_name, params_before, chosen
) -> List[TrialResult]:
    """One TrialResult per menu item, marking the chosen one as accepted."""
    return [
        TrialResult(
            prune_ratio=prune_ratio,
            quality=quality[n],
            quality_drop=baseline_quality - quality[n],
            params_after=pa_by_name.get(n, params_before),
            weight_reduction=(1.0 - pa_by_name.get(n, params_before) / params_before)
            if params_before
            else 0.0,
            accepted=(n == chosen),
        )
        for n, _ in menu
    ]


def _menu_result(
    menu,
    domain,
    chosen,
    fallback_name,
    prune_ratio,
    quality,
    pa_by_name,
    *,
    report,
    baseline_quality,
    params_before,
    params_after,
    apply,
) -> AutoTuneResult:
    """Note the winning variant in the report and package the menu search as an AutoTuneResult."""
    if report is not None:
        report.add(
            "menu-select",
            domain,
            domain,
            f"chosen variant {chosen!r} (fallback {fallback_name!r}) at "
            f"prune_ratio={prune_ratio:.4f}",
        )
    winner_quality = quality[chosen]
    trials = _build_menu_trials(
        menu, prune_ratio, quality, baseline_quality, pa_by_name, params_before, chosen
    )
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


def _menu_setup(base_cfg):
    """Unpack the menu domain and its shared settings; the first variant is the fallback."""
    domain = _find_menu_domain(base_cfg)
    menu, prune_ratio, common_kwargs = _menu_spec(base_cfg, domain)
    if not (0.0 <= prune_ratio < 1.0):
        raise ValueError(f"prune_ratio must be in [0.0, 1.0). Got {prune_ratio}.")
    return domain, menu, prune_ratio, common_kwargs, menu[0][0]


def _menu_measure(
    model,
    base_cfg,
    domain,
    menu,
    prune_ratio,
    common_kwargs,
    params_before,
    *,
    data,
    evaluator,
    eval_iterations,
    eval_batches,
    batch_adapter,
    finetune_fn,
    quant_fn,
):
    """Measure the baseline and every menu variant on a copy. Returns (baseline, per-variant quality, params)."""
    quality_fn, baseline_quality = _resolve_quality(
        model, evaluator, eval_iterations, eval_batches, batch_adapter
    )
    quality, pa_by_name = _menu_trial(
        model,
        base_cfg,
        domain,
        menu,
        prune_ratio,
        common_kwargs,
        quality_fn,
        params_before,
        data=data,
        batch_adapter=batch_adapter,
        finetune_fn=finetune_fn,
        quant_fn=quant_fn,
    )
    return baseline_quality, quality, pa_by_name


def _run_menu_mode(
    model,
    base_cfg,
    *,
    data,
    evaluator,
    eval_iterations,
    eval_batches,
    batch_adapter,
    apply,
    finetune_fn,
    quant_fn,
) -> AutoTuneResult:
    """Guarded best-of-menu search returning a normal AutoTuneResult."""
    domain, menu, prune_ratio, common_kwargs, fallback_name = _menu_setup(base_cfg)

    params_before = count_parameters(model)
    baseline_quality, quality, pa_by_name = _menu_measure(
        model,
        base_cfg,
        domain,
        menu,
        prune_ratio,
        common_kwargs,
        params_before,
        data=data,
        evaluator=evaluator,
        eval_iterations=eval_iterations,
        eval_batches=eval_batches,
        batch_adapter=batch_adapter,
        finetune_fn=finetune_fn,
        quant_fn=quant_fn,
    )
    chosen, _fb_q, _gain = _menu_select(quality, fallback_name)
    report, params_after = _menu_apply(
        model,
        base_cfg,
        domain,
        menu,
        chosen,
        prune_ratio,
        common_kwargs,
        pa_by_name,
        params_before,
        apply,
        data=data,
        batch_adapter=batch_adapter,
        finetune_fn=finetune_fn,
    )
    return _menu_result(
        menu,
        domain,
        chosen,
        fallback_name,
        prune_ratio,
        quality,
        pa_by_name,
        report=report,
        baseline_quality=baseline_quality,
        params_before=params_before,
        params_after=params_after,
        apply=apply,
    )


def _tolerance_search(
    model,
    config,
    tolerance,
    ratio_grid,
    safe_skip_attention,
    *,
    data,
    evaluator,
    eval_iterations,
    eval_data,
    batch_adapter,
    finetune_fn,
    quant_fn,
    apply,
) -> AutoTuneResult:
    """Grid search for the largest prune ratio whose quality drop stays within tolerance."""
    grid, base_cfg, params_before, eval_batches = _prepare_search_inputs(
        model, config, safe_skip_attention, ratio_grid, data, eval_data
    )
    quality_fn, baseline_quality = _resolve_quality(
        model, evaluator, eval_iterations, eval_batches, batch_adapter
    )
    trial = _make_trial_fn(
        model,
        base_cfg,
        params_before,
        data=data,
        batch_adapter=batch_adapter,
        finetune_fn=finetune_fn,
        quant_fn=quant_fn,
        quality_fn=quality_fn,
        baseline_quality=baseline_quality,
        accept_fn=lambda drop, pa: drop <= tolerance,
        warn_tag="auto-prune",
    )
    best_idx, ordered_trials, trials = _search_max_ratio(grid, trial, tolerance)
    return _finalize_autotune(
        model,
        base_cfg,
        grid,
        best_idx,
        trials,
        ordered_trials,
        tolerance=tolerance,
        baseline_quality=baseline_quality,
        params_before=params_before,
        data=data,
        batch_adapter=batch_adapter,
        finetune_fn=finetune_fn,
        apply=apply,
    )


def _accuracy_based_auto_prune(
    model: nn.Module,
    config=None,
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
    Function: search the grid for the largest prune ratio within tolerance and apply it in place
    Parameter: model: model to prune (mutated in place when apply=True)
               config: dict / PruneConfig / None; None uses the three-domain defaults
               data: calibration data; also the default fidelity eval set when eval_data is not given
               tolerance: upper bound on acceptable accuracy loss; default 0.02
               evaluator: Callable[[model], float] or any object exposing .evaluate(model) -> float;
                          None = default fidelity
               eval_iterations: iteration count passed to evaluator.evaluate
               eval_data: eval batches for the default fidelity metric; falls back to data
               ratio_grid: candidate prune ratios (ascending)
               batch_adapter: maps a data batch to (args, kwargs) for the model forward
               safe_skip_attention: merges attention submodules into skip_layers; default True
               finetune_fn: optional post-prune recovery callback run on each pruned copy before evaluation
               quant_fn: optional callback applied to the pruned copy before evaluation
               apply: True applies the chosen ratio in place; False searches only
    Return: AutoTuneResult carrying chosen_ratio / weight_reduction / quality_drop and summary()
    """
    if not (0.0 <= tolerance):
        raise ValueError("tolerance must be >= 0.")
    menu_cfg = _config_to_dict(config)
    if safe_skip_attention:
        menu_cfg = _merge_attention_skips(menu_cfg)
    if _find_menu_domain(menu_cfg) is not None:
        eval_batches = (
            list(eval_data)
            if eval_data is not None
            else (list(data) if data is not None else [])
        )
        return _run_menu_mode(
            model,
            menu_cfg,
            data=data,
            evaluator=evaluator,
            eval_iterations=eval_iterations,
            eval_batches=eval_batches,
            batch_adapter=batch_adapter,
            apply=apply,
            finetune_fn=finetune_fn,
            quant_fn=quant_fn,
        )
    return _tolerance_search(
        model,
        config,
        tolerance,
        ratio_grid,
        safe_skip_attention,
        data=data,
        evaluator=evaluator,
        eval_iterations=eval_iterations,
        eval_data=eval_data,
        batch_adapter=batch_adapter,
        finetune_fn=finetune_fn,
        quant_fn=quant_fn,
        apply=apply,
    )


_BudgetSetup = namedtuple(
    "_BudgetSetup",
    "grid base_cfg params_before eval_batches target_params unreachable prunable_fraction",
)


def _resolve_budget_config(
    model,
    config,
    safe_skip_attention,
    ratio_grid,
    data,
    eval_data,
    target_keep_ratio,
    target_params,
    budget_scope,
):
    """Resolve the size-budget entry (search setup, prunable-mass accounting, target conversion).
    Returns a _BudgetSetup."""
    grid, base_cfg, params_before, eval_batches = _prepare_search_inputs(
        model, config, safe_skip_attention, ratio_grid, data, eval_data
    )
    prunable_mass = _prunable_param_count(model, base_cfg)
    prunable_fraction = prunable_mass / params_before if params_before else 0.0
    target_params, unreachable = _resolve_size_target(
        params_before,
        prunable_mass,
        target_keep_ratio,
        target_params,
        budget_scope,
        grid[-1],
    )
    return _BudgetSetup(
        grid,
        base_cfg,
        params_before,
        eval_batches,
        target_params,
        unreachable,
        prunable_fraction,
    )


def _build_budget_report(
    model,
    base_cfg,
    grid,
    best_idx,
    trials,
    ordered_trials,
    *,
    baseline_quality,
    params_before,
    data,
    batch_adapter,
    finetune_fn,
    apply,
    unreachable,
    prunable_fraction,
):
    """Assemble the size-budget result after the search (tolerance=-1 plus budget fields)."""
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
    return _attach_budget_fields(result, unreachable, prunable_fraction)


def _size_budget_prune(
    model: nn.Module,
    config=None,
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
    Function: binary-search the smallest ratio whose params fit the target; report that point's (size, accuracy)
    Parameter: model: model to prune (mutated in place when apply=True)
               config: dict / PruneConfig / None; None uses the three-domain defaults
               data: calibration data; also the default fidelity eval set when eval_data is not given
               target_keep_ratio: fraction of params to keep; use either this or target_params
               target_params: direct upper bound on the target param count
               evaluator: Callable[[model], float] or any object exposing .evaluate(model) -> float;
                          None = default fidelity
               eval_iterations: iteration count passed to evaluator.evaluate
               eval_data: eval batches for the default fidelity metric; falls back to data
               ratio_grid: candidate prune ratios (ascending)
               batch_adapter: maps a data batch to (args, kwargs) for the model forward
               safe_skip_attention: merges attention submodules into skip_layers; default True
               finetune_fn: optional post-prune recovery callback run on each pruned copy before evaluation
               quant_fn: optional callback applied before evaluation
               apply: True applies the chosen ratio in place; False searches only
               budget_scope: 'total' (over all params) or 'prunable' (over prunable mass only); default 'total'
    Return: AutoTuneResult; tolerance is -1, chosen_ratio is None when no ratio meets the budget
    """
    setup = _resolve_budget_config(
        model,
        config,
        safe_skip_attention,
        ratio_grid,
        data,
        eval_data,
        target_keep_ratio,
        target_params,
        budget_scope,
    )
    quality_fn, baseline_quality = _resolve_quality(
        model, evaluator, eval_iterations, setup.eval_batches, batch_adapter
    )
    trial = _make_trial_fn(
        model,
        setup.base_cfg,
        setup.params_before,
        data=data,
        batch_adapter=batch_adapter,
        finetune_fn=finetune_fn,
        quant_fn=quant_fn,
        quality_fn=quality_fn,
        baseline_quality=baseline_quality,
        accept_fn=lambda drop, pa: pa <= setup.target_params,
        warn_tag="size-budget",
    )
    best_idx, ordered_trials, trials = _search_min_ratio(
        setup.grid, trial, setup.target_params
    )
    return _build_budget_report(
        model,
        setup.base_cfg,
        setup.grid,
        best_idx,
        trials,
        ordered_trials,
        baseline_quality=baseline_quality,
        params_before=setup.params_before,
        data=data,
        batch_adapter=batch_adapter,
        finetune_fn=finetune_fn,
        apply=apply,
        unreachable=setup.unreachable,
        prunable_fraction=setup.prunable_fraction,
    )
