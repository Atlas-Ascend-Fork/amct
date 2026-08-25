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


_NEG = (
    -1e9
)  # finite "never select this" logit: -inf breeds NaNs in softmax denominators


def _neg_for(dtype: torch.dtype) -> float:
    """The most negative finite poison this dtype can hold: -1e9 overflows fp16."""
    if dtype.is_floating_point:
        return max(_NEG, float(torch.finfo(dtype).min) / 2)
    return _NEG


class MaskUnsupported(RuntimeError):
    """This model routes in a shape the mask cannot faithfully rebuild.

    Raised instead of letting the mask silently do nothing: a no-op mask measures every
    candidate at zero quality drop and the search would pick the most aggressive ratio.
    The search catches this and falls back to copy-based trials.
    """


# Distinguishes "the model had no prune metadata" from "it had None".
_MISSING = object()


def _swallow(fn, *args) -> None:
    """Run one restore step; a failure must not abort the steps after it."""
    try:
        fn(*args)
    except Exception:  # noqa: BLE001 -- rollback is best-effort by design
        pass


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
    """A router that softmaxes before top-k hands out probabilities, not raw logits.

    Known limitation: the test is purely shape-of-values (non-negative, rows sum to
    one), so a one-hot or otherwise L1-normalised tensor false-positives as a softmax
    output. For scatter-pattern dispatch tensors (mostly-zero rows) the router mask
    hook rebuilds the tensor from the reselected top-k, which makes the false positive
    benign there; a dense non-softmax tensor that happens to be L1-normalised is still
    misclassified.
    """
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
        # renormalise the surviving mass first, then select. Renormalising in float32
        # (like the softmax branch) keeps bf16/fp16 survivors from underflowing to
        # exact-zero ties; the cast back happens at the return below.
        scores = logits.float()
        scores = scores / scores.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    else:
        scores = torch.softmax(logits.float(), dim=-1)
    values, indices = torch.topk(scores, k, dim=-1)
    if renormalise:
        values = values / values.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return values.to(logits.dtype), indices


def _is_scatter_dispatch(tensor: torch.Tensor, k: int) -> bool:
    """True when a full-width tensor is a top-k scatter, not a dense distribution.

    A router that scatters its top-k weights back into a (tokens, num_experts) canvas
    leaves every non-selected entry exactly zero, so per-row support is at most k. A
    dense softmax output has (essentially) full support. The distinction decides
    whether the mask can rebuild the tensor faithfully from a reselected top-k: for a
    scatter tensor it must, because its survivors keep unrenormalised values and its
    zeros tie with masked-removed entries.
    """
    if tensor.shape[-1] <= k:
        return False
    support = (tensor.detach() != 0).sum(dim=-1)
    return bool((support <= k).all())


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


def _bias_tripwire_hook(removed: List[int], total: int):
    """Verify, once, that a poisoned router bias actually reached the logits.

    A custom forward may compute F.linear(x, self.weight) and never read self.bias,
    leaving the poison inert and every candidate measuring unpruned. The poison is
    static, so one look at the first forward's expert-wide output is enough.
    """
    checked = [False]

    def hook(module: nn.Module, args: tuple, output: Any) -> None:
        _ = module
        _ = args
        if checked[0]:
            return
        items = list(output) if isinstance(output, tuple) else [output]
        for item in items:
            if torch.is_tensor(item) and item.dim() >= 1 and item.shape[-1] == total:
                checked[0] = True
                idx = torch.as_tensor(removed, device=item.device, dtype=torch.long)
                # The poison was clamped to the bias dtype's range (fp16 cannot hold
                # -1e9), so the tripwire threshold must be clamped the same way.
                threshold = _neg_for(item.dtype) / 2
                if not bool((item[..., idx].float() <= threshold).all()):
                    raise MaskUnsupported(
                        "this router's forward ignores its bias: the poisoned "
                        "logits never reach the selection, so the removed "
                        "experts would still be routable"
                    )
                break

    return hook


def _selector_reroute_hook(router: nn.Module, selector, removed: List[int]):
    """Recompute the block's expert selection from masked logits, using its own selector.

    The pre-hook fires on the experts module, whose args are (hidden, indices, weights)
    in some order; the recomputed pair replaces whichever args match their shapes. If no
    index arg and no shape-matched weights arg are found, the mask would change nothing
    downstream, so the hook refuses rather than measure an unpruned model.
    """

    def hook(module: nn.Module, args: tuple):
        _ = module
        if not args or not torch.is_tensor(args[0]):
            return None
        hidden = args[0]
        weight = router.weight
        bias = getattr(router, "bias", None)
        # The recompute runs in float32 while the real router may run bf16, so
        # near-tie selections can differ. That is accepted noise: the poisoned
        # experts sit at -inf and can never win in either precision.
        logits = nn.functional.linear(
            hidden.float(),
            weight.float(),
            bias.float() if bias is not None else None,
        )
        idx = torch.as_tensor(removed, device=logits.device, dtype=torch.long)
        logits[..., idx] = float("-inf")
        new_index, new_weights = selector(logits.unsqueeze(0))
        new_index = new_index.view(hidden.shape[0], -1)
        new_weights = new_weights.view(hidden.shape[0], -1).to(hidden.dtype)
        out = list(args)
        replaced_index = False
        replaced_weights = False
        for i, item in enumerate(out[1:], start=1):
            if not torch.is_tensor(item):
                continue
            if item.dtype in (torch.int32, torch.int64):
                out[i] = new_index.to(item.dtype)
                replaced_index = True
            elif item.is_floating_point() and item.shape == new_weights.shape:
                # Shape-gated on purpose: the experts module may take unrelated
                # float args (residuals, scales) that must not be clobbered.
                out[i] = new_weights.to(item.dtype)
                replaced_weights = True
        if not (replaced_index and replaced_weights):
            raise MaskUnsupported(
                "a simulated MoE trial cannot rebuild this block's dispatch: the "
                "experts module's arguments carry no (index, weights) pair matching "
                "the recomputed selection, so the removed experts would still run"
            )
        return tuple(out)

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
        if probabilities and _is_scatter_dispatch(output[logits_at], int(k)):
            # A scatter-pattern dispatch tensor: zeroing the removed entries leaves the
            # survivors with their unrenormalised values (the removed mass is lost, not
            # redistributed) and freed slots at exactly 0.0, tied with every other zero.
            # A block that consumes this full-width tensor would mismeasure the trial,
            # so rebuild it: scatter the renormalised reselected top-k into a fresh
            # zero canvas. Dense probability rows (where the scatter pattern cannot be
            # established) keep the masked tensor as before.
            items[logits_at] = torch.zeros_like(masked).scatter_(-1, indices, values)
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
        self._attr_journal: List[Tuple[nn.Module, str, Any]] = []
        self._tensor_journal: List[Tuple[torch.Tensor, torch.Tensor]] = []
        self._created: List[Tuple[nn.Module, str]] = []

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
        # Restoration must never leave _ACTIVE pointing at a dead session: everything
        # after this session -- including real prunes -- would silently route through
        # the simulation branches. Hence the finally, and hence restoring even when a
        # step above it raised.
        global _ACTIVE
        # Every stage restores independently: one failing setattr must not abandon the
        # tensor restores behind it, or a trial's -1e9 poison outlives the session.
        try:
            for handle in self._hooks:
                _swallow(handle.remove)
            self._hooks.clear()
            if self._meta is _MISSING:
                if hasattr(self.model, "_amct_prune_meta"):
                    _swallow(delattr, self.model, "_amct_prune_meta")
            else:
                _swallow(setattr, self.model, "_amct_prune_meta", self._meta)
            for module, name in self._created:
                _swallow(setattr, module, name, None)
            self._created.clear()
            for module, name, value in reversed(self._attr_journal):
                _swallow(setattr, module, name, value)
            self._attr_journal.clear()
            for tensor, original in reversed(self._tensor_journal):
                _swallow(lambda t=tensor, o=original: t.data.copy_(o.to(t.device)))
            self._tensor_journal.clear()
        finally:
            _ACTIVE = self._prev

    def note_bias_creation(self, module: nn.Module) -> None:
        """A trial may add a bias where there was none; drop it again on exit."""
        self._created.append((module, "bias"))

    def set_attr_for_trial(self, module: nn.Module, name: str, value) -> None:
        """Set an attribute for the duration of the session and restore it on exit.

        Journal only after the write succeeds: recording first would guarantee the
        restore fails on the same read-only attribute and abort the rest of the
        rollback. A refusing attribute is skipped, exactly as the real prune's
        `_collapse_expert_groups` skips it.
        """
        previous = getattr(module, name)
        try:
            setattr(module, name, value)
        except (AttributeError, TypeError):
            return
        self._attr_journal.append((module, name, previous))

    def poison_tensor_for_trial(
        self, tensor: torch.Tensor, removed: List[int], value: float
    ) -> None:
        """Write ``value`` into ``tensor[removed]`` for the session, restoring on exit.

        The value is clamped to the tensor's own dtype range: -1e9 saturates to -inf in
        fp16, which is exactly the NaN-in-softmax hazard the finite poison exists to avoid.
        """
        self._tensor_journal.append((tensor, tensor.detach().to("cpu", copy=True)))
        idx = torch.as_tensor(removed, device=tensor.device, dtype=torch.long)
        if tensor.is_floating_point():
            value = max(value, float(torch.finfo(tensor.dtype).min) / 2)
        tensor.data[idx] = value

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
        self._mask_router(model, target, total, keep)

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

    def _try_selector_rung(self, block, router, removed: List[int], total: int) -> bool:
        """DeepSeek-v2 pattern: the block selects experts itself; reroute at the experts' door."""
        selector = getattr(block, "route_tokens_to_experts", None)
        experts = getattr(block, "experts", None)
        weight = getattr(router, "weight", None)
        if (
            callable(selector)
            and isinstance(router, nn.Linear)
            and isinstance(experts, nn.Module)
            and weight is not None
        ):
            if isinstance(experts, (nn.ModuleList, nn.ModuleDict)):
                # An nn.ModuleList has no forward, so a pre-hook on it never fires --
                # and this block computes its logits functionally (never calling the
                # gate module), so no other rung can reach them either. Falling through
                # to bias injection would be a silent no-op measuring every candidate
                # at zero drop; refuse instead, and the search copies.
                raise MaskUnsupported(
                    "this block selects experts itself and dispatches over a ModuleList: "
                    "no hookable point exists for a masked trial"
                )
            # DeepSeek-v2 pattern: the block computes logits with F.linear(x, gate.weight)
            # and never calls the gate module, so no hook on the router and no bias can
            # reach those logits. Recompute the selection at the experts' door instead,
            # from masked logits, with the block's own selector doing the algorithm.
            hook = _selector_reroute_hook(router, selector, removed)
            self._hooks.append(experts.register_forward_pre_hook(hook))
            return True
        return False

    def _reroute(
        self, block, router: nn.Module, removed: List[int], total: int
    ) -> None:
        """Make the removed experts unroutable, whatever the selection algorithm.

        Strategy order: poison the logits where they are born, so the router's own code
        -- softmax, sparsemixer, sigmoid-with-correction, group top-k, sorted dispatch --
        excludes the removed experts by itself. Only bare-Parameter tuple routers, where
        no bias term exists to poison, fall back to rebuilding the returned selection.
        """
        if self._try_selector_rung(block, router, removed, total):
            self._poison_selection_state(block, router, removed, total)
            return
        candidates = [
            child
            for child in router.modules()
            if child is not router
            and isinstance(child, nn.Linear)
            and child.out_features == total
        ]
        if len(candidates) > 1:
            # Noisy-gating routers keep decoy Linears of the same width (a noise
            # projection next to the logit projection). Masking the wrong one is a
            # silent no-op, so refuse rather than guess.
            raise MaskUnsupported(
                "a simulated MoE trial cannot pick this router's logit producer: "
                f"{len(candidates)} inner Linears share out_features == {total}"
            )
        producer = candidates[0] if candidates else None
        if producer is not None:
            # A wrapped Linear born the logits: mask its output.
            self._hooks.append(
                producer.register_forward_hook(_logit_mask_hook(removed))
            )
        elif type(router) is nn.Linear and router.out_features == total:
            # A plain Linear returns its logits as the single output: mask them there.
            # Injecting a bias would be exact in real arithmetic but swaps the matmul
            # kernel (mm -> addmm), and the ulp-level change flips near-tie top-k picks
            # in fp16 -- observed as a large masked-vs-real gap on half models.
            self._hooks.append(router.register_forward_hook(_logit_mask_hook(removed)))
        elif isinstance(router, nn.Linear) and router.out_features == total:
            # The router IS a Linear (possibly with a custom forward on top, like
            # Phimoe's sparsemixer). Poisoning its bias corrupts the logits inside
            # super().forward() itself, before any selection code runs.
            if router.bias is None:
                device = router.weight.device
                router.bias = nn.Parameter(
                    torch.zeros(total, device=device, dtype=router.weight.dtype),
                    requires_grad=False,
                )
                self.note_bias_creation(router)
                router.bias.data[torch.as_tensor(removed, device=device)] = _neg_for(
                    router.bias.dtype
                )
            else:
                self.poison_tensor_for_trial(router.bias, removed, _NEG)
            # Tripwire: a custom forward may never read self.bias, in which case
            # the poison is inert and every candidate would measure unpruned.
            tripwire = _bias_tripwire_hook(removed, total)
            self._hooks.append(router.register_forward_hook(tripwire))
        else:
            self._hooks.append(
                router.register_forward_hook(
                    _router_mask_hook(removed, total, getattr(router, "top_k", None))
                )
            )
        self._poison_selection_state(block, router, removed, total)

    def _poison_selection_state(
        self, block, router, removed: List[int], total: int
    ) -> None:
        """Neutralise selection machinery that lives outside the logits.

        DeepSeek-style blocks add a learned per-expert correction bias to the selection
        scores and pick expert *groups* before experts. A real prune slices that bias and
        collapses the group count to 1 (`_collapse_expert_groups`); a trial must do the
        same for its selection to match, restoring both on exit.
        """
        from .domains.moe import _EXPERT_GROUP_ATTRS

        seen = set()
        for module in block.modules():
            for name, buf in list(module.named_buffers(recurse=False)) + list(
                module.named_parameters(recurse=False)
            ):
                if "correction" not in name and "score" not in name:
                    continue
                if buf.dim() >= 1 and buf.shape[-1] == total and id(buf) not in seen:
                    seen.add(id(buf))
                    flat = buf if buf.dim() == 1 else buf.view(-1)
                    self.poison_tensor_for_trial(flat, removed, _NEG)
        # Mirror the real prune's scoping exactly: update_moe_attributes collapses the
        # group attrs on the block object only, and _prune_fused_router collapses them
        # on the router. Neither ever walks nested descendants, so the trial must not
        # either -- and skipping the router would leave a fused-style router doing
        # grouped top-k among poisoned logits, making kept experts in decimated groups
        # unroutable in the trial only.
        for module in (block,) if router is block else (block, router):
            for attr in _EXPERT_GROUP_ATTRS:
                value = getattr(module, attr, None)
                if isinstance(value, int) and value > 1:
                    self.set_attr_for_trial(module, attr, 1)

    def _mask_router(self, model, target, total: int, keep: List[int]) -> None:
        removed = [i for i in range(total) if i not in set(keep)]
        if not target.router_path:
            raise MaskUnsupported(
                "a simulated MoE trial needs a router to mask: without one the dropped "
                "experts would still be selected"
            )
        router = get_submodule(model, target.router_path)
        block = (
            get_submodule(model, target.module_path) if target.module_path else model
        )
        self._reroute(block, router, removed, total)
        # The router's own expert-indexed parameters go too. It is not always an
        # nn.Linear: Qwen3.5-MoE's top-k router is a bare (num_experts, hidden) weight,
        # some models wrap an inner Linear, and the expert axis is not always dim 0 --
        # Ernie4.5 keeps a (1, num_experts) correction bias. Mirror the real prune,
        # which slices whichever axis matches the expert count.
        for module in router.modules():
            for param in module.parameters(recurse=False):
                if (
                    param.dim() >= 2
                    and param.shape[0] == total
                    and param.shape[1] == total
                ):
                    # A square (total, total) tensor is ambiguous: the real prune
                    # (_expert_slice_axis) refuses to pick an expert axis and leaves
                    # it unsliced, so the trial counts no cut for it either.
                    continue
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
        # A dense stage matches the FFNs inside experts that a MoE stage then drops
        # wholesale; the dropped module's full numel already covers everything inside
        # it, so cuts recorded on the dropped module or any of its descendants must
        # not be counted on top of it.
        dropped_ids = {id(sub) for m in self._dropped_modules for sub in m.modules()}
        for key, axes in self._cuts.items():
            if key in dropped_ids:
                continue
            module = self._cut_modules[key]
            # GPT-2-style Conv1D stores weight as [in, out]; everything else in scope
            # keeps [out, in]. Getting this wrong undercounts a dense cut by the whole
            # weight matrix, so resolve the layout per module, not globally.
            transposed = is_conv1d_like(module)
            created = {n for m, n in self._created if m is module}
            for name, param in module.named_parameters(recurse=False):
                if name in created:
                    # A bias this session invented (to poison a router's logits) is not
                    # part of the model; counting its cut would shift params_after.
                    continue
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
