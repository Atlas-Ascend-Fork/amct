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

from .utils import get_submodule, is_conv1d_like

_ACTIVE: Optional["SimulationSession"] = None


class MaskUnsupported(RuntimeError):
    """This model routes in a shape the mask cannot faithfully rebuild.

    Raised instead of letting the mask silently do nothing: a no-op mask measures every
    candidate at zero quality drop and the search would pick the most aggressive ratio.
    The search catches this and falls back to copy-based trials.
    """


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


def _reselect(
    logits: torch.Tensor, k: int, already_probabilities: bool, renormalise: bool
):
    """Redo the router's selection on masked logits, the way it would after a real removal.

    The weights a block multiplies expert outputs by come from a softmax, never from the
    raw logit values, so the softmax is reapplied unless the router already did it before
    we saw the tensor. Whether the top-k is then renormalised is the router's own choice
    (``norm_topk_prob``); doing it when the router would not leaves a visible gap.
    """
    k = max(1, min(int(k), logits.shape[-1]))
    if already_probabilities:
        # The distribution was computed before the removal, so the dropped experts still
        # sit in its denominator. A real removal softmaxes over the survivors only:
        # renormalise the surviving mass first, then select.
        scores = logits / logits.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    else:
        scores = torch.softmax(logits.float(), dim=-1)
    values, indices = torch.topk(scores, k, dim=-1)
    if renormalise:
        values = values / values.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return values.to(logits.dtype), indices


def _logit_mask_hook(removed: List[int]):
    """Send the removed experts' logits to -inf at the module that produces them."""

    def hook(module: nn.Module, args: tuple, output: Any):
        _ = module
        _ = args
        if not torch.is_tensor(output):
            return None
        idx = torch.as_tensor(removed, device=output.device, dtype=torch.long)
        masked = output.clone()
        masked[..., idx] = float("-inf")
        return masked

    return hook


def _mask_expert_scores(items, removed: List[int], total: int):
    """Mask the expert-indexed tensor in ``items``; returns (index, masked, probabilities)."""
    logits_at = next(
        (
            i
            for i, x in enumerate(items)
            if torch.is_tensor(x) and x.dim() >= 1 and x.shape[-1] == total
        ),
        None,
    )
    if logits_at is None:
        raise MaskUnsupported(
            "a simulated MoE trial cannot mask this router: none of its outputs is "
            "indexed by expert"
        )
    logits = items[logits_at]
    idx = torch.as_tensor(removed, device=logits.device, dtype=torch.long)
    probabilities = _looks_like_probabilities(logits)
    masked = logits.clone()
    masked[..., idx] = 0.0 if probabilities else float("-inf")
    items[logits_at] = masked
    return logits_at, masked, probabilities


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
        logits_at, masked, probabilities = _mask_expert_scores(items, removed, total)
        if not isinstance(output, tuple):
            return masked

        k = top_k if top_k is not None else getattr(module, "top_k", None)
        if k is None:
            raise MaskUnsupported(
                "a simulated MoE trial cannot rebuild this router's selection: it returns "
                "its own top-k but exposes no top_k to redo it with"
            )
        values, indices = _reselect(
            masked,
            min(int(k), total - len(removed)),
            probabilities,
            # A router that softmaxes before top-k owns this flag; one that hands out
            # probabilities (Qwen3.5-style) always renormalises its top-k.
            renormalise=bool(getattr(module, "norm_topk_prob", True)),
        )
        replaced = 0
        for i, item in enumerate(items):
            if i == logits_at or not torch.is_tensor(item):
                continue
            if (
                item.dtype in (torch.int32, torch.int64)
                and item.shape[-1] == values.shape[-1]
            ):
                items[i] = indices
                replaced += 1
            elif item.is_floating_point() and item.shape[-1] == values.shape[-1]:
                items[i] = values.to(item.dtype)
                replaced += 1
        if not replaced:
            # GraniteMoe-5.5-style gating: the selection leaves as flat sorted dispatch
            # tensors (plus a python list), nothing (tokens, k)-shaped to swap. Masking
            # only the logits would change nothing downstream.
            raise MaskUnsupported(
                "a simulated MoE trial cannot rebuild this router's dispatch tensors"
            )
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
            # Mirror the real prune's walk (domains.moe._prune_fused): the experts module,
            # the block itself, and the block's direct children carrying an expert-stacked
            # tensor -- GraniteMoe keeps output_linear as a sibling of the experts module,
            # and counting only the experts misses it entirely.
            for module in self._fused_expert_modules(model, target, experts, total):
                self.cut(module, "out", len(keep), total)
        elif isinstance(experts, nn.ModuleList):
            self.drop_modules([experts[i] for i in range(total) if i not in keep_set])
        self._mask_router(model, target.router_path, total, keep)

    def _fused_expert_modules(self, model, target, experts, total: int):
        """The modules a real fused prune slices on the expert axis, in the same order."""
        from .domains.moe import MoEPruningDomain
        from .utils import get_submodule as _get

        def carries_expert_dim(module):
            return any(
                q.dim() >= 1 and q.shape[0] == total
                for q in module.parameters(recurse=False)
            ) or any(
                b is not None and b.dim() >= 1 and b.shape[0] == total
                for _, b in module.named_buffers(recurse=False)
            )

        seen = []
        if carries_expert_dim(experts):
            seen.append(experts)
        block = _get(model, target.module_path) if target.module_path else model
        if block is not experts and carries_expert_dim(block):
            seen.append(block)
        domain = MoEPruningDomain()
        for name, child in block.named_children():
            if (
                child is experts
                or name in MoEPruningDomain.ROUTER_NAMES
                or domain._is_shared_expert_name(name)
            ):
                continue
            if any(
                q.dim() >= 3 and q.shape[0] == total
                for q in child.parameters(recurse=False)
            ):
                seen.append(child)
        return seen

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
        producer = next(
            (
                child
                for child in router.modules()
                if child is not router
                and isinstance(child, nn.Linear)
                and child.out_features == total
            ),
            None,
        )
        if producer is not None:
            # Mask the logits where they are born and let the router's own forward
            # recompute its top-k, sort orders and per-expert counts from them. This
            # covers gatings whose dispatch tensors no output rewrite could rebuild
            # (GraniteMoe sorts and flattens its selection before returning it).
            self._hooks.append(
                producer.register_forward_hook(_logit_mask_hook(removed))
            )
        else:
            self._hooks.append(
                router.register_forward_hook(
                    _router_mask_hook(removed, total, getattr(router, "top_k", None))
                )
            )
        # The router's own expert-indexed parameters go too. It is not always an
        # nn.Linear: Qwen3.5-MoE's top-k router is a bare (num_experts, hidden) weight,
        # some models wrap an inner Linear, and the expert axis is not always dim 0 --
        # Ernie4.5 keeps a (1, num_experts) correction bias. Mirror the real prune,
        # which slices whichever axis matches the expert count.
        for module in router.modules():
            for param in module.parameters(recurse=False):
                if param.dim() >= 1 and param.shape[0] == total:
                    self.cut(module, "out", len(keep), total)
                elif param.dim() >= 2 and param.shape[1] == total:
                    self.cut(module, "in", len(keep), total)

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
            # GPT-2-style Conv1D stores weight as [in, out]; everything else in scope
            # keeps [out, in]. Getting this wrong undercounts a dense cut by the whole
            # weight matrix, so resolve the layout per module, not globally.
            transposed = is_conv1d_like(module)
            for name, param in module.named_parameters(recurse=False):
                kept_numel = param.numel()
                for axis, dim in (("out", 0), ("in", 1)):
                    if axis not in axes:
                        continue
                    d = dim
                    if transposed and name == "weight" and param.dim() >= 2:
                        d = 1 - dim
                    if param.dim() <= d:
                        continue
                    kept, total = axes[axis]
                    if param.shape[d] != total:
                        continue
                    kept_numel = kept_numel * kept // total
                removed += param.numel() - kept_numel
        return removed
