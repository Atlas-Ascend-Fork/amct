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
"""AMCT-style façade: ``amct.prune(model, cfg, ...)`` and a ModelOptimizer-compatible
``PrunePass``. The signature mirrors ``amct.quantize(model, config)`` and the same
config can be added as a Pass in a prune -> quantize -> convert pipeline. The library
never downloads models; the caller passes an instantiated torch.nn.Module.
"""

from typing import Any, Callable, Mapping, Optional, Union

from torch import nn

from amct_pytorch.common.utils.check_params import check_params

from .config import PruneConfig
from .context import BatchAdapter
from .pruner import AutoPruner
from .report import PruneReport

ConfigLike = Union[dict, PruneConfig, type(None)]

# Like amct.quantize/convert, the façade returns None (in-place). The prune report
# is retrieved via get_last_report().
_LAST_REPORT: Optional[PruneReport] = None
# Tolerance-mode (accuracy-driven auto-prune) result, via get_last_auto_prune().
_LAST_AUTO_PRUNE = None


def _coerce_config(config: ConfigLike) -> PruneConfig:
    if config is None:
        return PruneConfig()
    if isinstance(config, PruneConfig):
        return config
    if isinstance(config, Mapping):
        return PruneConfig(**dict(config))
    raise TypeError(
        f"prune config must be dict / PruneConfig / None, got {type(config)!r}"
    )


@check_params(model=nn.Module, config=(dict, PruneConfig, type(None)))
def prune(
    model: nn.Module,
    config: ConfigLike = None,
    *,
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
) -> None:
    """
    Function: Structured pruning of an instantiated torch model in-place (like amct.quantize)
    Parameter: model: torch.nn.Module to prune (caller prepares it, e.g. via from_pretrained)
               config: dict / PruneConfig / None; None uses the default PruneConfig
               data: calibration data (DataLoader / tensor / dict); required by variance methods
               batch_adapter: callable unpacking a custom batch into args, kwargs, or None
               tolerance: if given, enables accuracy-driven auto-prune (max acceptable accuracy loss)
               evaluator: tolerance-mode accuracy source -- a Callable[[model], float] or an
                   AutoCalibrationEvaluatorBase object; None uses default top-1 fidelity
                   (shared with quantization)
               eval_iterations: number of evaluation iterations in tolerance mode
               eval_data: evaluation data passed through to accuracy_based_auto_prune (tolerance mode)
               ratio_grid: candidate prune ratios passed through in tolerance/size_budget mode
               size_budget: if given (keep-ratio in (0, 1]), prune to that size budget
                   (subsumes size_budget_prune); mutually exclusive with tolerance
               finetune_fn: optional callable to finetune the model between ratio steps
               quant_fn: optional callable to quantize the model during the search
    Return: None; stats via get_last_report, search-mode result via get_last_auto_prune

    Three modes: fixed prune ratio (default) uses prune_ratio in config; accuracy-driven
    (tolerance given) searches the largest ratio within the tolerance (subsumes
    accuracy_based_auto_prune); size-budget (size_budget given) prunes to a target keep-ratio
    (subsumes size_budget_prune).
    """
    global _LAST_REPORT, _LAST_AUTO_PRUNE
    if tolerance is not None and size_budget is not None:
        raise ValueError(
            "Pass only one of tolerance= or size_budget= (mutually exclusive search modes)."
        )
    if tolerance is not None:
        # Lazy import to avoid a circular dependency.
        from .accuracy_based_auto_prune import (
            DEFAULT_RATIO_GRID,
            accuracy_based_auto_prune,
        )

        result = accuracy_based_auto_prune(
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
        _LAST_AUTO_PRUNE = result
        _LAST_REPORT = result.report
        return

    if size_budget is not None:
        from .accuracy_based_auto_prune import (
            DEFAULT_RATIO_GRID,
            size_budget_prune,
        )

        result = size_budget_prune(
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
        _LAST_AUTO_PRUNE = result
        _LAST_REPORT = result.report
        return

    cfg = _coerce_config(config)
    cfg.validate()
    # Per-layer sensitivity allocation: when strategy='sensitivity', measure
    # per-layer sensitivity then water-fill the prune ratio (calib_nll guard by
    # default); uniform/None keeps the current path unchanged.
    if cfg.allocation and cfg.allocation.get("strategy", "uniform") == "sensitivity":
        from .allocation import prune_with_allocation  # lazy import, avoid cycle

        _LAST_REPORT = prune_with_allocation(
            model, cfg, data=data, batch_adapter=batch_adapter
        )
        _LAST_AUTO_PRUNE = None
        return
    # Pruning defaults to copy_model=True (clones the model); force it off here to
    # match amct.quantize's in-place semantics.
    cfg.copy_model = False
    pruner = AutoPruner(cfg)
    pruned = pruner(model, data=data, batch_adapter=batch_adapter)
    # Even with copy_model=False, pruning may return a new object (compat path);
    # copy state back so the caller's reference stays valid.
    if pruned is not model:
        model.__dict__.update(pruned.__dict__)
    _LAST_REPORT = pruner.last_report
    _LAST_AUTO_PRUNE = None


def get_last_report() -> Optional[PruneReport]:
    """
    Function: Return the PruneReport from the last prune call
    Parameter: None
    Return: PruneReport or None if no prune has run
    """
    return _LAST_REPORT


def get_last_auto_prune():
    """
    Function: Return the last tolerance-mode auto-prune result
    Parameter: None
    Return: AutoTuneResult or None if not applicable
    """
    return _LAST_AUTO_PRUNE


class PrunePass:
    """
    Function: ModelOptimizer-style prune Pass; chains with other optimizer passes
    Parameter: config: dict / PruneConfig / None; None uses the default PruneConfig
               data: calibration data (DataLoader / tensor / dict), or None
               batch_adapter: callable unpacking a custom batch into args, kwargs, or None
    Return: PrunePass instance; call run(model) to prune in-place, then read last_report

    Add it to a ModelOptimizer via add_pass(PrunePass(prune_cfg, data=calib)) to run in a
    prune -> quantize pipeline.
    """

    def __init__(
        self,
        config: ConfigLike = None,
        *,
        data: Any = None,
        batch_adapter: Optional[BatchAdapter] = None,
    ) -> None:
        cfg = _coerce_config(config)
        cfg.copy_model = False
        self._pruner = AutoPruner(cfg)
        self._data = data
        self._batch_adapter = batch_adapter
        self.last_report: Optional[PruneReport] = None

    def run(self, model: nn.Module) -> None:
        if not isinstance(model, nn.Module):
            raise TypeError(
                f"PrunePass.run expects torch.nn.Module, got {type(model)!r}"
            )
        pruned = self._pruner(model, data=self._data, batch_adapter=self._batch_adapter)
        if pruned is not model:
            model.__dict__.update(pruned.__dict__)
        self.last_report = self._pruner.last_report
