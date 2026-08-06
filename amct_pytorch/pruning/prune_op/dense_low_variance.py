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

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from ...common.utils.log import LOGGER

from ..config import MethodSpec, PruneConfig
from ..context import PruneContext
from ..domains.base import BasePruningDomain
from ..domains.dense import DensePruningDomain, GatedDenseTarget, TwoLayerDenseTarget
from ..report import PruneReport
from ..utils import (
    is_linear_like,
    record_prune_size,
    record_prune_width,
    run_calibration_epoch,
    RunningVariance,
    get_submodule,
    topk_keep_indices,
)
from .base import BasePruningMethod


def _make_hook(rv: RunningVariance):
    def hook(
        module: nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor
    ) -> None:
        _ = module
        _ = inputs
        if isinstance(output, tuple):
            output_tensor = output[0]
        else:
            output_tensor = output
        if torch.is_tensor(output_tensor):
            rv.update(output_tensor)

    return hook


def _register_two_layer_hook(model, target, stats, hooks) -> None:
    producer = get_submodule(model, target.producer_path)
    # Match the domain, which also emits targets for Conv1D-style producers (GPT-2 et al.);
    # hooking only nn.Linear silently skipped them and left the model unchanged.
    if not is_linear_like(producer):
        return
    rv = RunningVariance()
    stats[target.producer_path] = rv
    hooks.append(producer.register_forward_hook(_make_hook(rv)))


def _register_fused_hook(model, target, stats, hooks) -> None:
    gate_up = get_submodule(model, target.gate_up_path)
    if not isinstance(gate_up, nn.Linear):
        return
    rv = RunningVariance()
    stats[target.down_path] = rv
    hooks.append(gate_up.register_forward_hook(_make_hook(rv)))


def _register_gated_hooks(model, target, stats, hooks) -> None:
    gate = get_submodule(model, target.gate_path)
    up = get_submodule(model, target.up_path)
    if not isinstance(gate, nn.Linear) or not isinstance(up, nn.Linear):
        return
    gate_rv = RunningVariance()
    up_rv = RunningVariance()
    stats[target.down_path] = (gate_rv, up_rv)
    hooks.append(gate.register_forward_hook(_make_hook(gate_rv)))
    hooks.append(up.register_forward_hook(_make_hook(up_rv)))


def _register_hooks(
    model: nn.Module,
    targets: List[TwoLayerDenseTarget | GatedDenseTarget],
) -> Tuple[
    Dict[str, RunningVariance | Tuple[RunningVariance, RunningVariance]],
    List[torch.utils.hooks.RemovableHandle],
]:
    stats: Dict[str, RunningVariance | Tuple[RunningVariance, RunningVariance]] = {}
    hooks: List[torch.utils.hooks.RemovableHandle] = []
    for target in targets:
        if isinstance(target, TwoLayerDenseTarget):
            _register_two_layer_hook(model, target, stats, hooks)
        elif hasattr(target, "gate_up_path"):
            _register_fused_hook(model, target, stats, hooks)
        else:
            _register_gated_hooks(model, target, stats, hooks)
    return stats, hooks


def _score_for_target(target, stats) -> Optional[Tuple[torch.Tensor, str]]:
    if isinstance(target, TwoLayerDenseTarget):
        rv = stats.get(target.producer_path)
        if not isinstance(rv, RunningVariance) or rv.count == 0:
            return None
        return rv.variance(), target.producer_path
    gated_stats = stats.get(target.down_path)
    if isinstance(gated_stats, RunningVariance):
        if gated_stats.count == 0:
            return None
        v = gated_stats.variance()
        inter = v.shape[0] // 2
        report_module = target.down_path.rpartition(".")[0] or target.down_path
        return v[:inter] + v[inter:], report_module
    if not isinstance(gated_stats, tuple):
        return None
    gate_rv, up_rv = gated_stats
    if gate_rv.count == 0 or up_rv.count == 0:
        return None
    report_module = target.down_path.rpartition(".")[0] or target.down_path
    return gate_rv.variance() + up_rv.variance(), report_module


def _prune_targets(
    method_name: str,
    model: nn.Module,
    domain: DensePruningDomain,
    targets: List[TwoLayerDenseTarget | GatedDenseTarget],
    stats,
    config: PruneConfig,
    report: PruneReport,
    prune_ratio: float,
) -> Tuple[Optional[int], bool]:
    last_hidden: Optional[int] = None
    any_collected = False
    for target in targets:
        hidden = domain.hidden_size(model, target)
        if hidden <= config.min_neurons:
            continue
        scored = _score_for_target(target, stats)
        if scored is None:
            continue
        score, report_module = scored
        any_collected = True
        keep_idx = topk_keep_indices(score, prune_ratio, config.min_neurons)
        if len(keep_idx) == hidden:
            continue
        domain.apply_keep_indices(model, target, keep_idx)
        report.add(
            domain.name,
            method_name,
            report_module,
            f"Activation variance prune {hidden} -> {len(keep_idx)}",
        )
        report.record_layer_sparsity(report_module, hidden, len(keep_idx))
        last_hidden = len(keep_idx)
        record_prune_width(model, "_dense_widths", len(keep_idx))
    return last_hidden, any_collected


def _warn_skip(report: PruneReport, message: str) -> None:
    LOGGER.logw(message, "amct_prune")
    report.add_warning(message)


class LowVarianceDensePruningMethod(BasePruningMethod):
    domain = "dense"
    name = "low_variance"
    accepted_kwargs = frozenset({"prune_ratio"})
    requires_data = True

    def apply(
        self,
        model: nn.Module,
        domain: BasePruningDomain,
        targets: List[TwoLayerDenseTarget | GatedDenseTarget],
        context: PruneContext,
        report: PruneReport,
        config: PruneConfig,
        spec: MethodSpec,
    ) -> None:
        if context.data is None:
            raise ValueError(
                "Dense low-variance pruning requires 1-epoch input data via prune(..., data=...)."
            )
        if not isinstance(domain, DensePruningDomain):
            raise TypeError(
                "LowVarianceDensePruningMethod requires DensePruningDomain."
            )

        prune_ratio = float(spec.kwargs.get("prune_ratio", 0.50))
        if prune_ratio <= 0:
            return

        stats, hooks = _register_hooks(model, targets)
        try:
            num_batches = run_calibration_epoch(model, context)
        finally:
            for handle in hooks:
                handle.remove()

        if num_batches == 0:
            _warn_skip(
                report,
                "Dense low-variance pruning received an empty pruning dataset. "
                "The dense stage was skipped without changing the model.",
            )
            return

        last_hidden, any_collected = _prune_targets(
            self.name, model, domain, targets, stats, config, report, prune_ratio
        )

        if not any_collected:
            _warn_skip(
                report,
                "Dense low-variance pruning could not collect activation statistics from the provided pruning data. "
                "The dense stage was skipped without changing the model.",
            )
            return

        if last_hidden is not None:
            record_prune_size(model, "dense_hidden_size", last_hidden)
