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
"""Non-destructive stand-in for a pruning trial.

The accuracy search tries each candidate ratio on a full copy of the model, so peak memory
is about twice the model. A trial only needs the *quality* of the pruned model, not the
pruned model itself, and that quality can be measured without rebuilding anything:

    consumer_out = sum_i W[:, i] * x_i

so forcing ``x_i = 0`` at the consumer's input removes channel ``i``'s contribution
exactly as deleting it would. Masking at the consumer input (rather than at the
producer output) keeps the equivalence exact no matter what sits in between: a
BatchNorm's beta or an activation with ``f(0) != 0`` would break producer-side masking.

For MoE the equivalent statement is about routing: a removed expert must never be
selected, so its router logit is forced to ``-inf`` and the remaining experts compete
for the freed slots exactly as they would after a real removal.

``reconstruct`` also rewrites its consumer's weights, which no mask can express. There
the session journals just that one tensor (on host memory) and writes the least-squares
result into the kept columns -- a delta, not a copy of the model.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .utils import get_submodule

_ACTIVE: Optional["SimulationSession"] = None

# Distinguishes "the model had no prune metadata" from "it had None".
_MISSING = object()


def active() -> Optional["SimulationSession"]:
    """The session intercepting pruning right now, or None when pruning for real."""
    return _ACTIVE


def _channel_mask(total: int, keep: List[int], device, dtype) -> torch.Tensor:
    mask = torch.zeros(total, device=device, dtype=dtype)
    mask[torch.as_tensor(sorted(keep), device=device, dtype=torch.long)] = 1
    return mask


def _mask_last_dim_hook(mask: torch.Tensor):
    """Zero the removed channels of a Linear-like consumer's input."""

    def hook(module: nn.Module, args: tuple):
        _ = module
        if not args or not torch.is_tensor(args[0]):
            return None
        x = args[0]
        m = mask.to(device=x.device, dtype=x.dtype)
        if x.shape[-1] != m.shape[0]:
            # A Linear consuming a flattened conv map: one channel spans a whole block.
            block = x.shape[-1] // m.shape[0]
            m = m.repeat_interleave(block)
        return (x * m,) + args[1:]

    return hook


def _mask_channel_dim_hook(mask: torch.Tensor):
    """Zero the removed channels of a Conv2d consumer's input (N, C, H, W)."""

    def hook(module: nn.Module, args: tuple):
        _ = module
        if not args or not torch.is_tensor(args[0]):
            return None
        x = args[0]
        m = mask.to(device=x.device, dtype=x.dtype).view(1, -1, 1, 1)
        return (x * m,) + args[1:]

    return hook


def _looks_like_probabilities(tensor: torch.Tensor) -> bool:
    """A router that softmaxes before top-k hands out probabilities, not raw logits."""
    if not tensor.is_floating_point():
        return False
    row_sums = tensor.detach().float().sum(dim=-1)
    return bool((tensor.detach() >= 0).all()) and bool(
        torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-3)
    )


def _reselect(scores: torch.Tensor, k: int, normalise: bool):
    """Redo the router's top-k on masked scores, the way it would after a real removal."""
    k = max(1, min(int(k), scores.shape[-1]))
    values, indices = torch.topk(scores, k, dim=-1)
    if normalise:
        values = values / values.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return values, indices


def _router_mask_hook(removed: List[int], total: int, top_k: Optional[int]):
    """Route the removed experts away, rebuilding whatever the router hands downstream.

    Zeroing the logits is not enough for a router that returns its own top-k: the block
    dispatches on those indices and never looks at the logits again. So the selection is
    recomputed from the masked scores. For a router that softmaxes before selecting, the
    normalising denominators cancel, and masking then renormalising the kept top-k is
    exactly what the router would have produced with those experts gone.
    """

    def hook(module: nn.Module, args: tuple, output: Any):
        _ = args
        items = list(output) if isinstance(output, tuple) else [output]
        logits_at = next(
            (
                i
                for i, x in enumerate(items)
                if torch.is_tensor(x) and x.dim() >= 1 and x.shape[-1] == total
            ),
            None,
        )
        if logits_at is None:
            raise ValueError(
                "a simulated MoE trial cannot mask this router: none of its outputs is "
                "indexed by expert"
            )
        logits = items[logits_at]
        idx = torch.as_tensor(removed, device=logits.device, dtype=torch.long)
        probabilities = _looks_like_probabilities(logits)
        masked = logits.clone()
        masked[..., idx] = 0.0 if probabilities else float("-inf")
        items[logits_at] = masked
        if not isinstance(output, tuple):
            return masked

        k = top_k if top_k is not None else getattr(module, "top_k", None)
        if k is not None:
            values, indices = _reselect(
                masked, min(int(k), total - len(removed)), probabilities
            )
            for i, item in enumerate(items):
                if i == logits_at or not torch.is_tensor(item):
                    continue
                if (
                    item.dtype in (torch.int32, torch.int64)
                    and item.shape[-1] == values.shape[-1]
                ):
                    items[i] = indices
                elif item.is_floating_point() and item.shape[-1] == values.shape[-1]:
                    items[i] = values.to(item.dtype)
        return tuple(items)

    return hook


def _numel(module: Optional[nn.Module]) -> int:
    if module is None:
        return 0
    return sum(p.numel() for p in module.parameters())


class SimulationSession:
    """Measure a pruning trial without rebuilding the model.

    Enter the session, run the ordinary pruning pipeline, and read ``removed_params``:
    the model behaves as if pruned while every parameter tensor keeps its original
    shape. Leaving the session restores the model bit-for-bit.
    """

    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self._cuts: Dict[int, Dict[str, Tuple[int, int]]] = {}
        self._cut_modules: Dict[int, nn.Module] = {}
        self._dropped_modules: List[nn.Module] = []
        self._hooks: List[Any] = []
        self._meta: Any = _MISSING
        self._prev: Optional["SimulationSession"] = None

    # -- lifecycle ---------------------------------------------------------
    def __enter__(self) -> "SimulationSession":
        global _ACTIVE
        self._prev = _ACTIVE
        # Trials share one model, so the prune metadata each one records would pile up:
        # two ratios leave two widths behind and the final config sync then reads them as
        # a non-uniform cut and refuses to write the count. Snapshot it and put it back.
        meta = getattr(self.model, "_amct_prune_meta", _MISSING)
        self._meta = _MISSING if meta is _MISSING else copy.deepcopy(meta)
        _ACTIVE = self
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        global _ACTIVE
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()
        if self._meta is _MISSING:
            if hasattr(self.model, "_amct_prune_meta"):
                delattr(self.model, "_amct_prune_meta")
        else:
            setattr(self.model, "_amct_prune_meta", self._meta)
        _ACTIVE = self._prev

    # -- weight journal ----------------------------------------------------
    def record_dense(self, model, target, hidden: int, keep: List[int]) -> None:
        """Dense FFN cut: the producer loses rows, the consumer loses the matching columns."""
        from .domains.dense import FusedGatedDenseTarget, TwoLayerDenseTarget

        if isinstance(target, TwoLayerDenseTarget):
            producers = [get_submodule(model, target.producer_path)]
            consumer_path = target.consumer_path
        else:
            consumer_path = target.down_path
            if isinstance(target, FusedGatedDenseTarget):
                # gate and up are stacked on dim 0, so the fused producer keeps twice as many rows.
                gate_up = get_submodule(model, target.gate_up_path)
                self.cut(gate_up, "out", 2 * len(keep), 2 * hidden)
                producers = []
            else:
                producers = [
                    get_submodule(model, target.gate_path),
                    get_submodule(model, target.up_path),
                ]
        self._record_channel_cut(model, consumer_path, hidden, keep, producers)

    def record_cnn(self, model, target, keep: List[int]) -> None:
        """CNN channel cut: the conv (and its BatchNorm) lose channels, the consumer its inputs."""
        producer = get_submodule(model, target.producer_path)
        consumer = get_submodule(model, target.consumer_path)
        total = int(producer.out_channels)
        producers = [producer]
        if target.bn_path is not None:
            bn_module = get_submodule(model, target.bn_path)
            # running_mean / running_var are buffers, so only the affine pair counts.
            if getattr(bn_module, "affine", False):
                producers.append(bn_module)
        in_total = (
            consumer.in_features
            if isinstance(consumer, nn.Linear)
            else consumer.in_channels
        )
        self._record_channel_cut(
            model, target.consumer_path, total, keep, producers, in_total
        )

    def record_moe(self, model, target, total: int, keep: List[int]) -> None:
        """Expert cut: the dropped experts' params go, and the router must stop selecting them."""
        experts = get_submodule(model, target.experts_path)
        keep_set = set(keep)
        if target.fused:
            # One stacked tensor per projection: experts live on dim 0.
            for module in experts.modules():
                if any(
                    q.dim() >= 2 and q.shape[0] == total
                    for q in module.parameters(recurse=False)
                ):
                    self.cut(module, "out", len(keep), total)
        elif isinstance(experts, nn.ModuleList):
            self.drop_modules([experts[i] for i in range(total) if i not in keep_set])
        self._mask_router(model, target.router_path, total, keep)

    def _record_channel_cut(
        self,
        model,
        consumer_path: str,
        total: int,
        keep: List[int],
        producers,
        consumer_axis_total=None,
    ) -> None:
        for producer in producers:
            self.cut(producer, "out", len(keep), total)
        consumer = get_submodule(model, consumer_path)
        in_total = consumer_axis_total if consumer_axis_total is not None else total
        in_kept = len(keep) * (in_total // total) if total else 0
        self.cut(consumer, "in", in_kept, in_total)
        self._mask_consumer_input(model, consumer_path, total, keep)

    def _mask_router(
        self, model, router_path: Optional[str], total: int, keep: List[int]
    ) -> None:
        removed = [i for i in range(total) if i not in set(keep)]
        if not router_path:
            raise ValueError(
                "a simulated MoE trial needs a router to mask: without one the dropped "
                "experts would still be selected"
            )
        router = get_submodule(model, router_path)
        self._hooks.append(
            router.register_forward_hook(
                _router_mask_hook(removed, total, getattr(router, "top_k", None))
            )
        )
        # The router's own expert-indexed parameters go too. It is not always an
        # nn.Linear: Qwen3.5-MoE's top-k router is a bare (num_experts, hidden) weight,
        # and some models wrap an inner Linear.
        self.cut(router, "out", len(keep), total)
        for child in router.modules():
            if child is router:
                continue
            if any(
                q.dim() >= 1 and q.shape[0] == total
                for q in child.parameters(recurse=False)
            ):
                self.cut(child, "out", len(keep), total)

    def _mask_consumer_input(
        self, model, consumer_path: str, total: int, keep: List[int]
    ) -> None:
        consumer = get_submodule(model, consumer_path)
        weight = getattr(consumer, "weight", None)
        device = weight.device if weight is not None else torch.device("cpu")
        dtype = weight.dtype if weight is not None else torch.float32
        mask = _channel_mask(total, keep, device, dtype)
        hook = (
            _mask_channel_dim_hook(mask)
            if isinstance(consumer, nn.Conv2d)
            else _mask_last_dim_hook(mask)
        )
        self._hooks.append(consumer.register_forward_pre_hook(hook))

    def cut(self, module: nn.Module, axis: str, kept: int, total: int) -> None:
        """Record that ``module`` loses an axis, so the param count can compose the cuts.

        Counting per target and summing does not work: a chained cut (conv2's output feeds
        conv3, whose input is cut in turn) shrinks a tensor along two axes, and a real prune
        applies the second cut to an already-smaller tensor.
        """
        key = id(module)
        self._cut_modules[key] = module
        self._cuts.setdefault(key, {})[axis] = (kept, total)

    def drop_modules(self, modules) -> None:
        """Whole submodules a real prune would delete (MoE experts)."""
        self._dropped_modules.extend(modules)

    @property
    def removed_params(self) -> int:
        """Params a real prune would have removed, composing every recorded cut."""
        removed = sum(
            sum(p.numel() for p in m.parameters()) for m in self._dropped_modules
        )
        for key, axes in self._cuts.items():
            module = self._cut_modules[key]
            for param in module.parameters(recurse=False):
                kept_numel = param.numel()
                for axis, dim in (("out", 0), ("in", 1)):
                    if axis not in axes or param.dim() <= dim:
                        continue
                    kept, total = axes[axis]
                    if param.shape[dim] != total:
                        continue
                    kept_numel = kept_numel * kept // total
                removed += param.numel() - kept_numel
        return removed
