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
    quality_fn: Optional[Callable[[nn.Module], float]] = None,
    eval_iterations: int = 1,
    eval_data: Any = None,
    ratio_grid: Optional[Any] = None,
    finetune_fn: Optional[Callable[[nn.Module], Any]] = None,
    quant_fn: Optional[Callable[[nn.Module], Any]] = None,
) -> None:
    """Structured pruning of an instantiated torch model (in-place, like amct.quantize).

    Two modes:

    1. **Fixed prune ratio** (``tolerance=None``, default): prune directly using the
       prune_ratio in ``config``.
    2. **Accuracy-driven auto-prune** (``tolerance`` given): the user only states the
       maximum acceptable accuracy loss; the largest prune ratio meeting the tolerance
       is searched and applied, auto-skipping attention submodules to avoid breaking
       LLMs. Result via :func:`get_last_auto_prune`::

           amct.prune(model, data=calib, tolerance=0.02)   # at most 2% drop
           print(amct.get_last_auto_prune().summary())

       This is the **unified entry point**: ``prune(..., tolerance=...)`` subsumes
       :func:`accuracy_based_auto_prune` (it drives the same tolerance search under the
       hood), so most users can stay on ``prune`` and never call
       ``accuracy_based_auto_prune`` directly.

    Parameters
    ----------
    model : torch.nn.Module
        Loaded model to prune (caller prepares it, e.g. via from_pretrained).
    config : dict | PruneConfig | None
        Prune config; None uses the default PruneConfig.
    data : Any
        Calibration data (DataLoader / tensor / dict); required by variance methods.
    batch_adapter : Callable | None
        Adapter unpacking a custom batch into ``args, kwargs``.
    tolerance : float | None
        If given, enables accuracy-driven auto-prune (see
        :func:`accuracy_based_auto_prune`).
    quality_fn : Callable[[nn.Module], float] | None
        Accuracy function (larger is better) used in tolerance mode. None uses the default
        top-1 fidelity metric. If you hold an evaluator object, pass its score through
        ``quality_fn``, e.g. ``prune(..., tolerance=t, quality_fn=lambda m: evaluator.evaluate(m))``.
    eval_data, ratio_grid :
        Passed through to :func:`accuracy_based_auto_prune` in tolerance mode only.

    Notes
    -----
    Returns nothing (like ``amct.quantize``); stats/warnings via
    :func:`get_last_report`, tolerance-mode search result via
    :func:`get_last_auto_prune`.
    """
    global _LAST_REPORT, _LAST_AUTO_PRUNE
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
            quality_fn=quality_fn,
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
    """Return the :class:`PruneReport` from the last :func:`prune` call, or None."""
    return _LAST_REPORT


def get_last_auto_prune():
    """Return the last tolerance-mode ``AutoTuneResult``, or None if not applicable."""
    return _LAST_AUTO_PRUNE


class PrunePass:
    """ModelOptimizer-style Pass; chains with InsertQuantizeModulePass and others.

    Usage::

        from amct_pytorch.classic.optimizer.model_optimizer import ModelOptimizer
        from amct_pytorch.pruning import PrunePass

        opt = ModelOptimizer()
        opt.add_pass(PrunePass(prune_cfg, data=calib))
        opt.add_pass(InsertQuantizeModulePass(layer_cfg))
        opt.do_optimizer(model)
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
