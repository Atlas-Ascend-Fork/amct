from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from ..config import MethodSpec, PruneConfig
from ..context import PruneContext
from ..domains.base import BasePruningDomain
from ..domains.dense import DensePruningDomain, GatedDenseTarget, TwoLayerDenseTarget
from ..report import PruneReport
from ..utils import RunningVariance, get_submodule, infer_model_device, move_batch_to_device, topk_keep_indices
from .base import BasePruningMethod


class LowVarianceDensePruningMethod(BasePruningMethod):
    domain = "dense"
    name = "low_variance"
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
            raise ValueError("Dense low-variance pruning requires 1-epoch input data via prune(..., data=...).")
        if not isinstance(domain, DensePruningDomain):
            raise TypeError("LowVarianceDensePruningMethod requires DensePruningDomain.")

        prune_ratio = float(spec.kwargs.get("prune_ratio", 0.50))
        if prune_ratio <= 0:
            return

        stats, hooks = self._register_hooks(model, targets)
        try:
            num_batches = self._run_one_epoch(model, context)
        finally:
            for handle in hooks:
                handle.remove()

        if num_batches == 0:
            self._warn_skip(
                report,
                "Dense low-variance pruning received an empty pruning dataset. "
                "The dense stage was skipped without changing the model.",
            )
            return

        last_hidden, any_collected = self._prune_targets(
            model, domain, targets, stats, config, report, prune_ratio
        )

        if not any_collected:
            self._warn_skip(
                report,
                "Dense low-variance pruning could not collect activation statistics from the provided pruning data. "
                "The dense stage was skipped without changing the model.",
            )
            return

        if last_hidden is not None:
            meta = getattr(model, "_amct_prune_meta", {})
            meta["dense_hidden_size"] = int(last_hidden)
            setattr(model, "_amct_prune_meta", meta)

    def _register_hooks(
        self,
        model: nn.Module,
        targets: List[TwoLayerDenseTarget | GatedDenseTarget],
    ) -> Tuple[Dict[str, RunningVariance | Tuple[RunningVariance, RunningVariance]],
               List[torch.utils.hooks.RemovableHandle]]:
        stats: Dict[str, RunningVariance | Tuple[RunningVariance, RunningVariance]] = {}
        hooks: List[torch.utils.hooks.RemovableHandle] = []
        for target in targets:
            if isinstance(target, TwoLayerDenseTarget):
                self._register_two_layer_hook(model, target, stats, hooks)
            else:
                self._register_gated_hooks(model, target, stats, hooks)
        return stats, hooks

    def _register_two_layer_hook(self, model, target, stats, hooks) -> None:
        producer = get_submodule(model, target.producer_path)
        if not isinstance(producer, nn.Linear):
            return
        rv = RunningVariance()
        stats[target.producer_path] = rv
        hooks.append(producer.register_forward_hook(self._make_hook(rv)))

    def _register_gated_hooks(self, model, target, stats, hooks) -> None:
        gate = get_submodule(model, target.gate_path)
        up = get_submodule(model, target.up_path)
        if not isinstance(gate, nn.Linear) or not isinstance(up, nn.Linear):
            return
        gate_rv = RunningVariance()
        up_rv = RunningVariance()
        stats[target.down_path] = (gate_rv, up_rv)
        hooks.append(gate.register_forward_hook(self._make_hook(gate_rv)))
        hooks.append(up.register_forward_hook(self._make_hook(up_rv)))

    def _make_hook(self, rv: RunningVariance):
        def hook(module: nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
            _ = module
            _ = inputs
            if isinstance(output, tuple):
                output_tensor = output[0]
            else:
                output_tensor = output
            if torch.is_tensor(output_tensor):
                rv.update(output_tensor)

        return hook

    def _run_one_epoch(self, model: nn.Module, context: PruneContext) -> int:
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

    def _prune_targets(
        self,
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
            scored = self._score_for_target(target, stats)
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
                self.name,
                report_module,
                f"Activation variance prune {hidden} -> {len(keep_idx)}",
            )
            report.record_layer_sparsity(report_module, hidden, len(keep_idx))
            last_hidden = len(keep_idx)
        return last_hidden, any_collected

    @staticmethod
    def _score_for_target(target, stats) -> Optional[Tuple[torch.Tensor, str]]:
        if isinstance(target, TwoLayerDenseTarget):
            rv = stats.get(target.producer_path)
            if not isinstance(rv, RunningVariance) or rv.count == 0:
                return None
            return rv.variance(), target.producer_path
        gated_stats = stats.get(target.down_path)
        if not isinstance(gated_stats, tuple):
            return None
        gate_rv, up_rv = gated_stats
        if gate_rv.count == 0 or up_rv.count == 0:
            return None
        report_module = target.down_path.rpartition(".")[0] or target.down_path
        return gate_rv.variance() + up_rv.variance(), report_module

    @staticmethod
    def _warn_skip(report: PruneReport, message: str) -> None:
        warnings.warn(message, RuntimeWarning, stacklevel=2)
        report.add_warning(message)
