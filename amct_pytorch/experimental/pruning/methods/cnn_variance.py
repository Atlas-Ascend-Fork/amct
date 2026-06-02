"""
CNN channel pruning: Activation-Variance Importance
====================================================
Channels with the highest output-activation variance are kept; the rest are
removed. For a conv output y in R^(B x C x H x W) we flatten to (B*H*W, C) and
track E[y] and E[y^2] per channel (Welford-style). Importance = Var[y_c].

Coupling across Conv -> BN -> Conv and Conv -> flatten -> Linear is handled by
CNNPruningDomain.apply_keep_indices, so no external graph library is required.
"""
from __future__ import annotations

import warnings
from typing import Dict, List

import torch
import torch.nn as nn

from ..config import MethodSpec, PruneConfig
from ..context import PruneContext
from ..domains.base import BasePruningDomain
from ..domains.cnn import CNNChannelTarget, CNNPruningDomain
from ..report import PruneReport
from ..utils import (
    RunningVariance,
    get_submodule,
    infer_model_device,
    move_batch_to_device,
    topk_keep_indices,
)
from .base import BasePruningMethod


class VarianceChannelPruningMethod(BasePruningMethod):
    """Channel pruning via activation variance, applied through CNNPruningDomain.

    Required kwargs
    ---------------
    prune_ratio : float
        Fraction of channels to remove per target (default 0.30).
    """

    domain = "cnn"
    name = "variance_channel"
    requires_data = True

    @staticmethod
    def _make_hook(rv: RunningVariance):
        def hook(module: nn.Module, _inp, output: torch.Tensor) -> None:
            _ = module
            if not torch.is_tensor(output) or output.ndim != 4:
                return
            with torch.no_grad():
                b, c, h, w = output.shape
                rv.update(output.detach().permute(0, 2, 3, 1).reshape(b * h * w, c))

        return hook

    @staticmethod
    def _run_one_epoch(model: nn.Module, context: PruneContext) -> int:
        device = infer_model_device(model)
        was_training = model.training
        model.eval()
        num_batches = 0
        try:
            with torch.no_grad():
                for args, kwargs in context.iter_model_inputs():
                    moved_args, moved_kwargs = move_batch_to_device(args, kwargs, device)
                    model(*moved_args, **moved_kwargs)
                    num_batches += 1
        finally:
            model.train(was_training)
        return num_batches

    def _register_variance_hooks(self, model: nn.Module, targets: List[CNNChannelTarget]):
        stats: Dict[str, RunningVariance] = {}
        hooks: List[torch.utils.hooks.RemovableHandle] = []
        for target in targets:
            producer = get_submodule(model, target.producer_path)
            if not isinstance(producer, nn.Conv2d):
                continue
            rv = RunningVariance()
            stats[target.producer_path] = rv
            hooks.append(producer.register_forward_hook(self._make_hook(rv)))
        return stats, hooks

    def _prune_targets_from_stats(
        self,
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
                self.name,
                target.producer_path,
                f"Variance channel prune {n_out} -> {len(keep_idx)}",
            )
            report.record_layer_sparsity(target.producer_path, n_out, len(keep_idx))
            any_collected = True
        return any_collected

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
            raise ValueError("variance_channel requires calibration data via prune(..., data=...).")

        prune_ratio = float(spec.kwargs.get("prune_ratio", 0.30))
        if prune_ratio <= 0:
            return

        stats, hooks = self._register_variance_hooks(model, targets)
        try:
            num_batches = self._run_one_epoch(model, context)
        finally:
            for handle in hooks:
                handle.remove()

        if num_batches == 0:
            message = "variance_channel: empty calibration data — CNN stage skipped."
            warnings.warn(message, RuntimeWarning, stacklevel=2)
            report.add_warning(message)
            return

        any_collected = self._prune_targets_from_stats(
            model, domain, targets, stats, config, report, prune_ratio
        )
        if not any_collected:
            report.add(
                domain.name,
                self.name,
                "<none>",
                "No channels pruned (all importance above threshold)",
            )
