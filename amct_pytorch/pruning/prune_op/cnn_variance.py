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

from typing import Dict, List

import torch
import torch.nn as nn

from ...common.utils.log import LOGGER

from ..config import MethodSpec, PruneConfig
from ..context import PruneContext
from ..domains.base import BasePruningDomain
from ..domains.cnn import CNNChannelTarget, CNNPruningDomain
from ..report import PruneReport
from ..utils import (
    run_calibration_epoch,
    RunningVariance,
    get_submodule,
    topk_keep_indices,
)
from .base import BasePruningMethod


def _make_hook(rv: RunningVariance):
    def hook(module: nn.Module, _inp, output: torch.Tensor) -> None:
        _ = module
        if not torch.is_tensor(output) or output.ndim != 4:
            return
        with torch.no_grad():
            b, c, h, w = output.shape
            rv.update(output.detach().permute(0, 2, 3, 1).reshape(b * h * w, c))

    return hook


def _register_variance_hooks(model: nn.Module, targets: List[CNNChannelTarget]):
    stats: Dict[str, RunningVariance] = {}
    hooks: List[torch.utils.hooks.RemovableHandle] = []
    for target in targets:
        producer = get_submodule(model, target.producer_path)
        if not isinstance(producer, nn.Conv2d):
            continue
        rv = RunningVariance()
        stats[target.producer_path] = rv
        hooks.append(producer.register_forward_hook(_make_hook(rv)))
    return stats, hooks


def _prune_targets_from_stats(
    method_name: str,
    model: nn.Module,
    domain: CNNPruningDomain,
    targets: List[CNNChannelTarget],
    stats: Dict[str, RunningVariance],
    config: PruneConfig,
    report: PruneReport,
    prune_ratio: float,
) -> bool:
    any_collected = False
    for target in targets:
        rv = stats.get(target.producer_path)
        if rv is None or rv.count == 0:
            continue
        n_out = domain.channel_count(model, target)
        if n_out <= config.min_channels:
            continue
        scores = rv.variance()
        keep_idx = topk_keep_indices(scores, prune_ratio, config.min_channels)
        if len(keep_idx) >= n_out:
            continue
        domain.apply_keep_indices(model, target, keep_idx)
        report.add(
            domain.name,
            method_name,
            target.producer_path,
            f"Variance channel prune {n_out} -> {len(keep_idx)}",
        )
        report.record_layer_sparsity(target.producer_path, n_out, len(keep_idx))
        any_collected = True
    return any_collected


class VarianceChannelPruningMethod(BasePruningMethod):
    """Channel pruning via activation variance, applied through CNNPruningDomain.

    kwargs: prune_ratio (float): fraction of channels to remove per target. Default 0.30.
    """

    domain = "cnn"
    name = "variance_channel"
    accepted_kwargs = frozenset({"prune_ratio"})
    supports_masked_trial = True
    requires_data = True

    def apply(
        self,
        model: nn.Module,
        domain: BasePruningDomain,
        targets: List[CNNChannelTarget],
        context: PruneContext,
        report: PruneReport,
        config: PruneConfig,
        spec: MethodSpec,
    ) -> None:
        if not isinstance(domain, CNNPruningDomain):
            raise TypeError("VarianceChannelPruningMethod requires CNNPruningDomain.")
        if context.data is None:
            raise ValueError(
                "variance_channel requires calibration data via prune(..., data=...)."
            )

        prune_ratio = float(spec.kwargs.get("prune_ratio", 0.30))
        if prune_ratio <= 0:
            return

        stats, hooks = _register_variance_hooks(model, targets)
        try:
            num_batches = run_calibration_epoch(model, context)
        finally:
            for handle in hooks:
                handle.remove()

        if num_batches == 0:
            message = "variance_channel: empty calibration data -- CNN stage skipped."
            LOGGER.logw(message, "amct_prune")
            report.add_warning(message)
            return

        any_collected = _prune_targets_from_stats(
            self.name, model, domain, targets, stats, config, report, prune_ratio
        )
        if not any_collected:
            message = (
                "variance_channel: no channels pruned (all importance above threshold) -- "
                "CNN model left unchanged."
            )
            LOGGER.logw(message, "amct_prune")
            report.add(domain.name, self.name, "<none>", message)
            report.add_warning(message)
