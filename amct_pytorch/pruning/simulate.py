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

from ..common.utils.log import LOGGER

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


def _router_mask_hook(removed: List[int]):
    """Force removed experts out of the top-k by sending their logits to -inf."""

    def hook(module: nn.Module, args: tuple, output: Any):
        _ = module
        _ = args
        logits = output[0] if isinstance(output, tuple) else output
        if not torch.is_tensor(logits):
            return None
        idx = torch.as_tensor(removed, device=logits.device, dtype=torch.long)
        masked = logits.clone()
        masked[..., idx] = float("-inf")
        if isinstance(output, tuple):
            return (masked,) + tuple(output[1:])
        return masked

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
        self.plan: List[Tuple[str, str, int, int]] = []
        self._cuts: Dict[int, Dict[str, Tuple[int, int]]] = {}
        self._cut_modules: Dict[int, nn.Module] = {}
        self._dropped_modules: List[nn.Module] = []
        self._hooks: List[Any] = []
        self._journal: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        self._masked_inputs: Dict[str, torch.Tensor] = {}
        self._created: List[Tuple[nn.Module, str]] = []
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
        for tensor, original in self._journal.values():
            tensor.data.copy_(original.to(tensor.device))
        self._journal.clear()
        for module, name in self._created:
            setattr(module, name, None)
        self._created.clear()
        if self._meta is _MISSING:
            if hasattr(self.model, "_amct_prune_meta"):
                delattr(self.model, "_amct_prune_meta")
        else:
            setattr(self.model, "_amct_prune_meta", self._meta)
        _ACTIVE = self._prev

    # -- weight journal ----------------------------------------------------
    def journal(self, tensor: torch.Tensor) -> None:
        """Stash a tensor's current value on host memory so __exit__ can restore it."""
        key = id(tensor)
        if key not in self._journal:
            self._journal[key] = (tensor, tensor.detach().to("cpu", copy=True))

    def note_bias_creation(self, module: nn.Module) -> None:
        """A recovery pass may add a bias where there was none; drop it again on exit."""
        if getattr(module, "bias", None) is None:
            self._created.append((module, "bias"))

    def write_kept_columns(
        self, consumer: nn.Module, keep: List[int], new_weight: torch.Tensor, dim: int
    ) -> bool:
        """Apply a reconstruction result to the kept slots of a full-size consumer.

        The removed slots stay whatever they were -- the input mask zeroes them anyway --
        so only the consumer's own weight is journaled, never the model.
        """
        weight = getattr(consumer, "weight", None)
        if weight is None:
            return False
        index = torch.as_tensor(sorted(keep), device=weight.device, dtype=torch.long)
        if new_weight.shape[dim] != index.shape[0]:
            return False
        self.journal(weight)
        weight.data.index_copy_(dim, index, new_weight.to(weight.dtype))
        return True

    # -- recording ---------------------------------------------------------
    def record_channel_cut(
        self,
        model,
        domain: str,
        consumer_path: str,
        total: int,
        keep: List[int],
        producers,
        consumer_axis_total=None,
    ) -> None:
        """A producer/consumer channel cut: mask the consumer's input, record both axes."""
        for producer in producers:
            self.cut(producer, "out", len(keep), total)
        consumer = get_submodule(model, consumer_path)
        in_total = consumer_axis_total if consumer_axis_total is not None else total
        in_kept = len(keep) * (in_total // total) if total else 0
        self.cut(consumer, "in", in_kept, in_total)
        self._mask_consumer_input(model, consumer_path, total, keep)
        self._note(consumer_path, domain, total, len(keep))

    def record_moe(
        self,
        model,
        router_path: Optional[str],
        experts_path: str,
        total: int,
        keep: List[int],
    ) -> None:
        removed = [i for i in range(total) if i not in set(keep)]
        if router_path:
            router = get_submodule(model, router_path)
            self._hooks.append(router.register_forward_hook(_router_mask_hook(removed)))
        else:
            LOGGER.logw(
                "[amct.prune] simulated MoE trial without a router module: the removed "
                "experts cannot be routed away, so this trial is skipped.",
                "amct_prune",
            )
            raise ValueError("MoE simulation needs a router to mask")
        if isinstance(router, nn.Linear):
            self.cut(router, "out", len(keep), total)
        self._note(experts_path, "moe", total, len(keep))

    def _mask_consumer_input(
        self, model, consumer_path: str, total: int, keep: List[int]
    ) -> None:
        consumer = get_submodule(model, consumer_path)
        weight = getattr(consumer, "weight", None)
        device = weight.device if weight is not None else torch.device("cpu")
        dtype = weight.dtype if weight is not None else torch.float32
        mask = _channel_mask(total, keep, device, dtype)
        self._masked_inputs[consumer_path] = mask
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

    def _note(self, path: str, domain: str, total: int, kept: int) -> None:
        self.plan.append((domain, path, total, kept))
