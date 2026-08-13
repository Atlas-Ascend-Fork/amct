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
import dataclasses
from typing import Any, Callable, Mapping, Optional, Union

from torch import nn

from amct_pytorch.common.utils.check_params import check_params

from .config import PruneConfig
from .context import BatchAdapter
from .pruner import AutoPruner
from .report import PruneReport

ConfigLike = Union[dict, PruneConfig, type(None)]


def _coerce_config(config: ConfigLike) -> PruneConfig:
    if config is None:
        return PruneConfig()
    if isinstance(config, PruneConfig):
        return dataclasses.replace(config)
    if isinstance(config, Mapping):
        return PruneConfig(**dict(config))
    raise TypeError(
        f"prune config must be dict / PruneConfig / None, got {type(config)!r}"
    )


def _run_tolerance_mode(
    model,
    config,
    *,
    data,
    tolerance,
    evaluator,
    eval_iterations,
    eval_data,
    ratio_grid,
    batch_adapter,
    finetune_fn,
    quant_fn,
):
    """Accuracy-driven search. Returns the report."""
    from .accuracy_based_auto_prune import (
        DEFAULT_RATIO_GRID,
        _accuracy_based_auto_prune,
    )

    result = _accuracy_based_auto_prune(
        model,
        config,
        data=data,
        tolerance=tolerance,
        evaluator=evaluator,
        eval_iterations=eval_iterations,
        eval_data=eval_data,
        ratio_grid=ratio_grid if ratio_grid is not None else DEFAULT_RATIO_GRID,
        batch_adapter=batch_adapter,
        finetune_fn=finetune_fn,
        quant_fn=quant_fn,
        apply=True,
    )
    return result.report


def _run_size_budget_mode(
    model,
    config,
    *,
    data,
    size_budget,
    evaluator,
    eval_iterations,
    eval_data,
    ratio_grid,
    batch_adapter,
    finetune_fn,
    quant_fn,
):
    """Prune to a target keep-ratio. Returns the report."""
    from .accuracy_based_auto_prune import DEFAULT_RATIO_GRID, _size_budget_prune

    result = _size_budget_prune(
        model,
        config,
        data=data,
        target_keep_ratio=size_budget,
        evaluator=evaluator,
        eval_iterations=eval_iterations,
        eval_data=eval_data,
        ratio_grid=ratio_grid if ratio_grid is not None else DEFAULT_RATIO_GRID,
        batch_adapter=batch_adapter,
        finetune_fn=finetune_fn,
        quant_fn=quant_fn,
        apply=True,
    )
    return result.report


def _run_menu_mode(
    model,
    config,
    *,
    data,
    evaluator,
    eval_iterations,
    eval_data,
    batch_adapter,
    finetune_fn,
    quant_fn,
):
    """Guarded menu search: try each variant, keep the one that measures best. Returns the report."""
    from .accuracy_based_auto_prune import _run_menu_mode as _menu_mode

    eval_batches = (
        list(eval_data)
        if eval_data is not None
        else (list(data) if data is not None else [])
    )
    result = _menu_mode(
        model,
        config,
        data=data,
        evaluator=evaluator,
        eval_iterations=eval_iterations,
        eval_batches=eval_batches,
        batch_adapter=batch_adapter,
        apply=True,
        finetune_fn=finetune_fn,
        quant_fn=quant_fn,
    )
    return result.report


def _reject_search_args(**kwargs) -> None:
    """Fixed-ratio mode has nothing to search, so these arguments must not be silently dropped."""
    unused = sorted(name for name, value in kwargs.items() if value is not None)
    if unused:
        raise ValueError(
            f"{', '.join(unused)} only apply to a search mode; pass tolerance=, "
            f"size_budget=, or a config carrying a 'menu'."
        )


def _menu_config(config):
    """Normalised config dict when it carries a method ``menu``, else None."""
    from .accuracy_based_auto_prune import (
        _config_to_dict,
        _find_menu_domain,
        _merge_attention_skips,
    )

    cfg = _merge_attention_skips(_config_to_dict(config))
    return cfg if _find_menu_domain(cfg) is not None else None


def _run_fixed_mode(model, config, *, data, batch_adapter):
    """Fixed prune ratio from config (default mode). Returns the report."""
    cfg = _coerce_config(config)
    cfg.validate()
    if cfg.allocation and cfg.allocation.get("strategy", "uniform") == "sensitivity":
        from .allocation import prune_with_allocation

        return prune_with_allocation(model, cfg, data=data, batch_adapter=batch_adapter)
    cfg.copy_model = False
    pruner = AutoPruner(cfg)
    pruned = pruner(model, data=data, batch_adapter=batch_adapter)
    if pruned is not model:
        model.__dict__.update(pruned.__dict__)
    return pruner.last_report


@check_params(model=nn.Module, config=(dict, PruneConfig, type(None)))
def prune(
    model: nn.Module,
    config: ConfigLike = None,
    data: Any = None,
    batch_adapter: Optional[BatchAdapter] = None,
    tolerance: Optional[float] = None,
    evaluator: Optional[Union[Callable[[nn.Module], float], Any]] = None,
    eval_iterations: int = 1,
    eval_data: Any = None,
    ratio_grid: Optional[Any] = None,
    size_budget: Optional[float] = None,
    finetune_fn: Optional[Callable[[nn.Module], Any]] = None,
    quant_fn: Optional[Callable[[nn.Module], Any]] = None,
    report: Optional[PruneReport] = None,
) -> None:
    """
    Structured pruning of a torch model in-place. Modes: fixed ratio (default), accuracy-driven
    (tolerance=), or size-budget (size_budget=); tolerance and size_budget are mutually exclusive.
    A config carrying a method ``menu`` runs the guarded menu search instead of the fixed ratio.
    """
    if tolerance is not None and size_budget is not None:
        raise ValueError(
            "Pass only one of tolerance= or size_budget= (mutually exclusive search modes)."
        )
    menu_cfg = _menu_config(config)
    if menu_cfg is not None:
        _reject_search_args(
            tolerance=tolerance, size_budget=size_budget, ratio_grid=ratio_grid
        )
    search_args = {
        "data": data,
        "evaluator": evaluator,
        "eval_iterations": eval_iterations,
        "eval_data": eval_data,
        "batch_adapter": batch_adapter,
        "finetune_fn": finetune_fn,
        "quant_fn": quant_fn,
    }
    if tolerance is not None:
        out = _run_tolerance_mode(
            model, config, tolerance=tolerance, ratio_grid=ratio_grid, **search_args
        )
    elif size_budget is not None:
        out = _run_size_budget_mode(
            model, config, size_budget=size_budget, ratio_grid=ratio_grid, **search_args
        )
    elif menu_cfg is not None:
        out = _run_menu_mode(model, menu_cfg, **search_args)
    else:
        _reject_search_args(
            evaluator=evaluator,
            eval_data=eval_data,
            ratio_grid=ratio_grid,
            finetune_fn=finetune_fn,
            quant_fn=quant_fn,
        )
        out = _run_fixed_mode(model, config, data=data, batch_adapter=batch_adapter)
    if report is not None and out is not None:
        report.copy_from(out)
