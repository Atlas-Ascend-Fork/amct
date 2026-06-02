"""
MoE Mass + Variance Expert Pruning.

Per-layer adaptive expert selection:
- Shallow layers (idx <= boundary): mass-based selection (activation frequency)
- Deep layers (idx > boundary): variance-based selection (per-expert routing weight variance)

Variance metric: for each expert i, collect all routing weights assigned to
expert i across all tokens where i is in the top-k. Compute Var(w) = E[w^2] - E[w]^2.
High variance = the expert receives very different weights depending on context
= specialized/important. Low variance = always gets similar weight = interchangeable.

Experts never selected during calibration are assigned the global mean variance
so they are not automatically pruned (they may be rare but critical).

Default boundary=10 (layers 0-10 use mass, layers 11+ use variance).
Validated on LLaDA2.1-mini: boundary=10 gives best ACC.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from ..config import MethodSpec, PruneConfig
from ..context import PruneContext
from ..domains.base import BasePruningDomain
from ..domains.moe import MoEPruningDomain, MoETarget
from ..report import PruneReport
from ..utils import (
    get_submodule,
    infer_model_device,
    move_batch_to_device,
    topk_keep_indices,
)
from .base import BasePruningMethod


@dataclass(slots=True)
class LayerRoutingStats:
    """Per-layer routing statistics collected during one epoch."""
    mass: torch.Tensor
    expert_var_sum: torch.Tensor
    expert_select_count: torch.Tensor
    expert_var: torch.Tensor
    num_experts: int = 0
    top_k: int = 8


class MassVarianceMoEPruningMethod(BasePruningMethod):
    """
    Adaptive mass + variance expert pruning.

    Shallow layers use mass (frequency), deep layers use variance
    (per-expert routing weight variance after softmax).

    kwargs:
        prune_ratio (float): Fraction of experts to remove. Default 0.50.
        top_k (int): Number of experts per token. Default 8.
        boundary (int): Layer index boundary. Layers <= boundary use mass,
                        layers > boundary use variance. Default 10.
    """

    domain = "moe"
    name = "mass_variance"
    requires_data = True

    def apply(
        self,
        model: nn.Module,
        domain: BasePruningDomain,
        targets: List[MoETarget],
        context: PruneContext,
        report: PruneReport,
        config: PruneConfig,
        spec: MethodSpec,
    ) -> None:
        if context.data is None:
            raise ValueError("MoE mass+variance pruning requires input data.")
        if not isinstance(domain, MoEPruningDomain):
            raise TypeError("Requires MoEPruningDomain.")

        prune_ratio = float(spec.kwargs.get("prune_ratio", 0.50))
        top_k = int(spec.kwargs.get("top_k", 8))
        boundary = int(spec.kwargs.get("boundary", 10))
        if prune_ratio <= 0:
            return

        layer_stats, hooks = self._register_router_hooks(model, domain, targets, top_k)
        try:
            num_batches = self._run_one_epoch(model, context)
        finally:
            for h in hooks:
                h.remove()

        if num_batches == 0:
            msg = "Empty dataset. Skipped."
            warnings.warn(msg, RuntimeWarning, stacklevel=2)
            report.add_warning(msg)
            return

        self._finalize_variance(layer_stats)
        last_num_experts = self._prune_layers(
            model, domain, targets, layer_stats, config, report, prune_ratio, top_k, boundary
        )

        if last_num_experts is not None:
            meta = getattr(model, "_amct_prune_meta", {})
            meta["moe_num_experts"] = int(last_num_experts)
            setattr(model, "_amct_prune_meta", meta)

    @staticmethod
    def _make_router_hook(stats: "LayerRoutingStats", k: int):
        def hook(module, inputs, output):
            _ = module
            _ = inputs
            if isinstance(output, tuple) and len(output) >= 3:
                topk_idx = output[0].detach().cpu()
                topk_wt = output[1].detach().float().cpu()
            elif torch.is_tensor(output):
                logits = output.detach().float().cpu()
                flat = logits.reshape(-1, logits.shape[-1])
                probs = torch.softmax(flat, dim=-1)
                topk_wt, topk_idx = torch.topk(probs, min(k, flat.shape[-1]), dim=-1)
            else:
                return

            topk_idx = topk_idx.reshape(-1, topk_idx.shape[-1])
            topk_wt = topk_wt.reshape(-1, topk_wt.shape[-1])
            n_tokens = topk_idx.shape[0]

            for t in range(n_tokens):
                for j in range(topk_idx.shape[1]):
                    eidx = int(topk_idx[t, j].item())
                    if 0 <= eidx < stats.num_experts:
                        w = float(topk_wt[t, j].item())
                        stats.mass[eidx] += w
                        stats.expert_var_sum[eidx] += w * w
                        stats.expert_select_count[eidx] += 1

        return hook

    def _register_router_hooks(
        self,
        model: nn.Module,
        domain: MoEPruningDomain,
        targets: List[MoETarget],
        top_k: int,
    ):
        layer_stats: Dict[str, LayerRoutingStats] = {}
        hooks: List[torch.utils.hooks.RemovableHandle] = []
        for target in targets:
            if target.router_path is None:
                continue
            router = get_submodule(model, target.router_path)
            num_experts = domain.num_experts(model, target)
            stats = LayerRoutingStats(
                mass=torch.zeros(num_experts, dtype=torch.float64),
                expert_var_sum=torch.zeros(num_experts, dtype=torch.float64),
                expert_select_count=torch.zeros(num_experts, dtype=torch.long),
                expert_var=torch.zeros(num_experts, dtype=torch.float32),
                num_experts=num_experts,
                top_k=top_k,
            )
            layer_stats[target.router_path] = stats
            hooks.append(router.register_forward_hook(self._make_router_hook(stats, top_k)))
        return layer_stats, hooks

    @staticmethod
    def _finalize_variance(layer_stats: Dict[str, "LayerRoutingStats"]) -> None:
        for st in layer_stats.values():
            observed = st.expert_select_count > 0
            count = st.expert_select_count.double()
            mean_w = torch.where(observed, st.mass / count, torch.zeros_like(st.mass))
            mean_w_sq = torch.where(observed, st.expert_var_sum / count, torch.zeros_like(st.expert_var_sum))
            var = (mean_w_sq - mean_w * mean_w).clamp(min=0).float()
            if observed.any() and (~observed).any():
                var[~observed] = var[observed].mean()
            st.expert_var = var

    def _prune_layers(
        self,
        model: nn.Module,
        domain: MoEPruningDomain,
        targets: List[MoETarget],
        layer_stats: Dict[str, "LayerRoutingStats"],
        config: PruneConfig,
        report: PruneReport,
        prune_ratio: float,
        top_k: int,
        boundary: int,
    ) -> Optional[int]:
        last_num_experts: Optional[int] = None
        for idx, target in enumerate(targets):
            keep_idx, method_desc, num_experts = self._select_keep_indices(
                idx, target, layer_stats, config, prune_ratio, boundary
            )
            if keep_idx is None:
                continue
            self._prune_one_layer(model, domain, target, keep_idx, top_k)
            report.add(
                domain.name, self.name,
                target.module_path or target.experts_path,
                f"{method_desc}-based prune {num_experts} -> {len(keep_idx)}",
            )
            report.record_layer_sparsity(target.module_path or target.experts_path, num_experts, len(keep_idx))
            last_num_experts = len(keep_idx)
        return last_num_experts

    @staticmethod
    def _select_keep_indices(
        idx: int,
        target: MoETarget,
        layer_stats: Dict[str, "LayerRoutingStats"],
        config: PruneConfig,
        prune_ratio: float,
        boundary: int,
    ):
        if target.router_path is None or target.router_path not in layer_stats:
            return None, None, 0
        st = layer_stats[target.router_path]
        num_experts = st.num_experts
        if num_experts <= config.min_experts:
            return None, None, num_experts
        if idx <= boundary:
            score = st.mass.float()
            method_desc = "mass"
        else:
            score = st.expert_var
            method_desc = "variance"
        keep_idx = topk_keep_indices(score, prune_ratio, config.min_experts)
        if len(keep_idx) == num_experts:
            return None, None, num_experts
        return keep_idx, method_desc, num_experts

    def _prune_one_layer(
        self,
        model: nn.Module,
        domain: MoEPruningDomain,
        target: MoETarget,
        keep_idx: List[int],
        top_k: int,
    ) -> None:
        experts = domain.get_experts(model, target)
        new_experts = nn.ModuleList([experts[i] for i in keep_idx])
        domain.replace_experts(model, target, new_experts)
        domain.prune_router(model, target, keep_idx)
        domain.update_moe_attributes(model, target, len(keep_idx))
        self._patch_router_module(model, target, keep_idx, top_k)

    @staticmethod
    def _patch_router_module(
        model: nn.Module,
        target: MoETarget,
        keep_idx: List[int],
        top_k: int,
    ) -> None:
        moe_module = get_submodule(model, target.module_path) if target.module_path else model
        router = getattr(moe_module, "gate", None) or getattr(moe_module, "router", None)
        if router is None:
            return
        if hasattr(router, "group_limited_topk"):
            def _simple_topk(scores, _router=router, _k=top_k):
                k = min(getattr(_router, "top_k", _k), scores.shape[-1])
                return torch.topk(scores, k=k, dim=-1)
            router.group_limited_topk = _simple_topk
        if hasattr(router, "expert_bias"):
            device = infer_model_device(model)
            router.expert_bias = torch.zeros(
                len(keep_idx), device=device, dtype=router.expert_bias.dtype
            )

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
