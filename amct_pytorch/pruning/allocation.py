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
import math
import re
from dataclasses import fields
from typing import Any, List, Mapping, Optional, Sequence

from torch import nn

from ..common.utils.log import LOGGER
from .accuracy_based_auto_prune import (
    _make_fidelity_quality,
    _target_module_paths,
    _with_ratio,
)
from .calib import calib_nll
from .compat import detect_backend
from .config import PruneConfig
from .pruner import AutoPruner
from .report import PruneReport
from .utils import count_parameters

_LAYER_RE = re.compile(r"(.*\b(?:layers|blocks|h)\.\d+)(?:\.|$)")
_IDX_RE = re.compile(r"(?:layers|blocks|h)\.(\d+)")


def layer_prefixes(model: nn.Module) -> List[str]:
    """Layer prefixes of the form '...layers.{i}', ascending by layer index."""
    keys = set()
    for name, _ in model.named_modules():
        match = _LAYER_RE.search(name)
        if match:
            keys.add(match.group(1))
    return sorted(keys, key=lambda s: int(_IDX_RE.search(s).group(1)))


def _base_config_dict(config: Optional[PruneConfig]) -> dict:
    """Base config for per-layer application: copy fields, strip allocation, disable cloning."""
    cfg = config or PruneConfig()
    base = {f.name: copy.deepcopy(getattr(cfg, f.name)) for f in fields(cfg)}
    base["allocation"] = None
    base["copy_model"] = False
    return base


def _prunable_target_paths(model: nn.Module, base_cfg: dict) -> List[str]:
    """All module paths hit by the three-domain (cnn/dense/moe) find_targets."""
    from .domains.cnn import CNNPruningDomain
    from .domains.dense import DensePruningDomain
    from .domains.moe import MoEPruningDomain

    cfg = PruneConfig(**copy.deepcopy(base_cfg))
    paths: List[str] = []
    for domain in (CNNPruningDomain(), DensePruningDomain(), MoEPruningDomain()):
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
                f"[allocation] {domain.name} find_targets did not return targets; per-layer "
                f"isolation may be incomplete ({type(exc).__name__}: {exc}).",
                "amct_prune",
            )
            continue
        for target in targets:
            paths.extend(_target_module_paths(target))
    return sorted(set(paths))


def _isolation_skips(
    keys: Sequence[str], keep: str, extra: Sequence[str], target_paths: Sequence[str]
) -> List[str]:
    """Skip everything outside the keep prefix (other layer prefixes + out-of-layer targets), keeping user skips."""
    skips = list(extra) + [k + "." for k in keys if k != keep]
    prefix = keep + "."
    skips += [p for p in target_paths if p != keep and not p.startswith(prefix)]
    return skips


def _prune_single_layer(model, base_cfg, keys, key, ratio, data, batch_adapter):
    """Prune only the key layer at ratio (isolating other layers and out-of-layer targets); mutates model in place."""
    cfg_dict = _with_ratio(base_cfg, float(ratio))
    cfg_dict["skip_layers"] = _isolation_skips(
        keys,
        key,
        base_cfg.get("skip_layers") or [],
        _prunable_target_paths(model, base_cfg),
    )
    pruner = AutoPruner(PruneConfig(**cfg_dict))
    pruned = pruner(model, data=data, batch_adapter=batch_adapter)
    if pruned is not model:
        model.__dict__.update(pruned.__dict__)


def _apply_layer_ratios(model, base_cfg, keys, ratios, data, batch_adapter):
    """Prune layers sequentially per ratios (skipping ratio<=0); mutates and returns model."""
    for key, ratio in zip(keys, ratios):
        if ratio <= 0:
            continue
        _prune_single_layer(model, base_cfg, keys, key, ratio, data, batch_adapter)
    return model


def measure_layer_sensitivity(
    model: nn.Module,
    data: Sequence[Any],
    config: Optional[PruneConfig] = None,
    ref_ratio: float = 0.5,
    skip_layers: Optional[Sequence[str]] = None,
) -> "dict[str, float]":
    """Prune each layer alone at ref_ratio, measure top-1 fidelity drop -> {layer prefix: drop}."""
    batches = list(data)
    if not batches:
        raise ValueError("measure_layer_sensitivity needs calibration data batches.")
    keys = layer_prefixes(model)
    if not keys:
        raise ValueError(
            "no '(layers|blocks|h).<i>' prefixed layers found; "
            "sensitivity allocation needs a layered model."
        )
    base_cfg = _base_config_dict(config)
    if skip_layers:
        base_cfg["skip_layers"] = list(base_cfg.get("skip_layers") or []) + list(
            skip_layers
        )
    fid = _make_fidelity_quality(model, batches, None)
    sens: dict = {}
    for key in keys:
        cand = copy.deepcopy(model)
        try:
            _prune_single_layer(cand, base_cfg, keys, key, ref_ratio, batches, None)
            sens[key] = 1.0 - float(fid(cand))
        except (
            RuntimeError,
            ValueError,
            TypeError,
            AttributeError,
            KeyError,
            IndexError,
        ) as exc:
            LOGGER.logw(
                f"[allocation] layer '{key}' sensitivity probe did not complete "
                f"({type(exc).__name__}: {exc}); treating as maximally sensitive.",
                "amct_prune",
            )
            sens[key] = 1.0
    return sens


def _redistribute_residual(
    ratios: List[float], total: float, floor: float, ceil: float
):
    """Iteratively redistribute the budget residual to unclamped layers until it converges or all are clamped."""
    for _ in range(len(ratios) + 1):
        residual = total - sum(ratios)
        if abs(residual) <= 1e-9:
            return
        if residual > 0:
            free = [i for i, r in enumerate(ratios) if r < ceil - 1e-12]
        else:
            free = [i for i, r in enumerate(ratios) if r > floor + 1e-12]
        if not free:
            return
        share = residual / len(free)
        for i in free:
            ratios[i] = min(ceil, max(floor, ratios[i] + share))


def water_fill_ratios(
    sens, total_cut: float, floor: float = 0.05, ceil: float = 0.9
) -> List[float]:
    """water-fill allocation: total cut total_cut*N is fixed; low-sensitivity layers get a larger share."""
    vals = list(sens.values()) if isinstance(sens, Mapping) else list(sens)
    if not vals:
        raise ValueError("sens must be non-empty.")
    n = len(vals)
    total = total_cut * n
    if not (floor * n - 1e-6 <= total <= ceil * n + 1e-6):
        raise ValueError(
            f"total budget {total:.4f} infeasible under floor={floor}, ceil={ceil}, n={n}."
        )
    mean = sum(vals) / n if sum(vals) else 1.0
    ratios = [
        min(ceil, max(floor, total_cut * (1.0 + (mean - s) / (mean + 1e-9))))
        for s in vals
    ]
    _redistribute_residual(ratios, total, floor, ceil)
    return ratios


def _uniform_cut(config: PruneConfig) -> float:
    """Per-layer ratio under the uniform convention (allocation keeps the same total).

    Take it from whichever domains are actually active, not from ``dense`` alone: a config
    that only names cnn or moe leaves dense pinned at 0.0.
    """
    ratios = [
        float(spec.kwargs.get("prune_ratio", 0.0))
        for spec in config.resolved_methods().values()
    ]
    cut = max(ratios) if ratios else 0.0
    if cut <= 0.0:
        raise ValueError(
            "sensitivity allocation needs at least one domain with prune_ratio > 0; "
            "every domain in this config resolves to 0.0."
        )
    return cut


def _guard_pick(model, base_cfg, keys, ratios, cut, batches, batch_adapter):
    """Build sensitivity / uniform candidates on deepcopies, pick lower calib NLL; warn on revert."""
    cand_sens = _apply_layer_ratios(
        copy.deepcopy(model), base_cfg, keys, ratios, batches, batch_adapter
    )
    cand_uni = _apply_layer_ratios(
        copy.deepcopy(model), base_cfg, keys, [cut] * len(keys), batches, batch_adapter
    )
    try:
        nll_sens = calib_nll(cand_sens, batches)
        nll_uni = calib_nll(cand_uni, batches)
    except (ValueError, AttributeError, TypeError) as exc:
        LOGGER.logw(
            f"[allocation] calib NLL guard unavailable ({type(exc).__name__}); "
            f"reverting to uniform allocation.",
            "amct_prune",
        )
        return cand_uni, "uniform"
    if math.isnan(nll_sens) or math.isnan(nll_uni):
        LOGGER.logw(
            f"[allocation] NaN calib NLL detected (sens={nll_sens}, uniform={nll_uni}); "
            f"guard cannot trust sensitivity candidate.",
            "amct_prune",
        )
    if not nll_sens < nll_uni:
        LOGGER.logw(
            f"[allocation] sensitivity allocation does not beat uniform on calib NLL "
            f"(sens={nll_sens:.4f} vs uniform={nll_uni:.4f}); reverting to uniform.",
            "amct_prune",
        )
        return cand_uni, "uniform"
    return cand_sens, "sensitivity"


def _uniform_whole_model(model, base_cfg, keys, cut, batches, batch_adapter, report):
    """When too few layers (keys<2), actually run whole-model uniform pruning in place (never warn-then-no-op)."""
    LOGGER.logw(
        "[allocation] fewer than 2 layered blocks found; "
        "applying uniform pruning to the whole model.",
        "amct_prune",
    )
    pruner = AutoPruner(PruneConfig(**copy.deepcopy(base_cfg)))
    pruned = pruner(model, data=batches, batch_adapter=batch_adapter)
    if pruned is not model:
        model.__dict__.update(pruned.__dict__)
    for key in keys:
        report.add("allocation", "uniform", key, f"ratio={cut:.3f}")
    report.params_after = count_parameters(model)
    report.allocation_choice = "uniform"
    return report


def prune_with_allocation(
    model: nn.Module,
    config: PruneConfig,
    data: Any = None,
    batch_adapter=None,
) -> PruneReport:
    """Sensitivity-allocation pruning: measure sensitivity -> water-fill -> guard -> apply in place."""
    if data is None:
        raise ValueError(
            "allocation strategy 'sensitivity' requires calibration data; "
            "pass prune(..., data=...)."
        )
    alloc = dict(config.allocation or {})
    batches = list(data)
    backend = detect_backend(model)
    params_before = count_parameters(model)
    base_cfg = _base_config_dict(config)
    keys = layer_prefixes(model)
    cut = _uniform_cut(config)
    report = PruneReport(backend=backend.name, params_before=params_before)
    if len(keys) < 2:
        return _uniform_whole_model(
            model, base_cfg, keys, cut, batches, batch_adapter, report
        )
    sens = measure_layer_sensitivity(
        model,
        batches,
        config,
        float(alloc.get("ref_ratio", cut)),
        config.skip_layers,
    )
    ratios = water_fill_ratios(
        sens,
        cut,
        float(alloc.get("min_ratio", 0.05)),
        float(alloc.get("max_ratio", 0.9)),
    )
    if alloc.get("guard", "calib_nll") == "calib_nll":
        chosen, choice = _guard_pick(
            model, base_cfg, keys, ratios, cut, batches, batch_adapter
        )
    else:
        chosen = _apply_layer_ratios(
            copy.deepcopy(model), base_cfg, keys, ratios, batches, batch_adapter
        )
        choice = "sensitivity"
    model.__dict__.update(chosen.__dict__)
    for key, ratio in zip(
        keys, ratios if choice == "sensitivity" else [cut] * len(keys)
    ):
        report.add("allocation", choice, key, f"ratio={ratio:.3f}")
    report.params_after = count_parameters(model)
    report.allocation_choice = choice
    return report
