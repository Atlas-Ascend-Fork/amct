from __future__ import annotations

import warnings
from typing import Dict, List, Tuple

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

        stats: Dict[str, RunningVariance | Tuple[RunningVariance, RunningVariance]] = {}
        hooks: List[torch.utils.hooks.RemovableHandle] = []
        for target in targets:
            if isinstance(target, TwoLayerDenseTarget):
                producer = get_submodule(model, target.producer_path)
                if not isinstance(producer, nn.Linear):
                    continue
                rv = RunningVariance()
                stats[target.producer_path] = rv
                hooks.append(producer.register_forward_hook(self._make_hook(rv)))
            else:
                gate = get_submodule(model, target.gate_path)
                up = get_submodule(model, target.up_path)
                if not isinstance(gate, nn.Linear) or not isinstance(up, nn.Linear):
                    continue
                gate_rv = RunningVariance()
                up_rv = RunningVariance()
                stats[target.down_path] = (gate_rv, up_rv)
                hooks.append(gate.register_forward_hook(self._make_hook(gate_rv)))
                hooks.append(up.register_forward_hook(self._make_hook(up_rv)))

        try:
            num_batches = self._run_one_epoch(model, context)
        finally:
            for handle in hooks:
                handle.remove()

        if num_batches == 0:
            message = (
                "Dense low-variance pruning received an empty pruning dataset. "
                "The dense stage was skipped without changing the model."
            )
            warnings.warn(message, RuntimeWarning, stacklevel=2)
            report.add_warning(message)
            return

        last_hidden: int | None = None
        any_collected = False
        for target in targets:
            hidden = domain.hidden_size(model, target)
            if hidden <= config.min_neurons:
                continue
            if isinstance(target, TwoLayerDenseTarget):
                rv = stats.get(target.producer_path)
                if not isinstance(rv, RunningVariance) or rv.count == 0:
                    continue
                score = rv.variance()
                report_module = target.producer_path
                any_collected = True
            else:
                gated_stats = stats.get(target.down_path)
                if not isinstance(gated_stats, tuple):
                    continue
                gate_rv, up_rv = gated_stats
                if gate_rv.count == 0 or up_rv.count == 0:
                    continue
                score = gate_rv.variance() + up_rv.variance()
                report_module = target.down_path.rpartition(".")[0] or target.down_path
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
            last_hidden = len(keep_idx)

        if not any_collected:
            message = (
                "Dense low-variance pruning could not collect activation statistics from the provided pruning data. "
                "The dense stage was skipped without changing the model."
            )
            warnings.warn(message, RuntimeWarning, stacklevel=2)
            report.add_warning(message)
            return

        if last_hidden is not None:
            meta = getattr(model, "_ascend_prune_meta", {})
            meta["dense_hidden_size"] = int(last_hidden)
            setattr(model, "_ascend_prune_meta", meta)

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
