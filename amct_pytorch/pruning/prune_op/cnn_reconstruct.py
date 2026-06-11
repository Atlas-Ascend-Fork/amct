# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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
"""Output-preserving (reconstruct) CNN channel pruning -- self-contained, no third-party deps.

Reuses the sequential (producer conv -> bn -> consumer) channel targets found by the ``cnn``
domain. For each target, low-saliency producer output channels are removed and the consumer
weights are reconstructed by least squares (ThiNet/CP style) to keep the consumer output:

    Y = im2col(X) @ W_flat^T               (X: consumer input, Y: original output)
    W_kept = argmin || Y - im2col(X[:, keep]) @ W^T || + ridge

Reconstruction absorbs the removed channels' contribution into the kept ones, so accuracy
loss is smaller than naive slicing (``variance_channel``). Only sequential couplings the
``cnn`` domain can safely identify are handled (residual/branch boundaries are left alone).
kwargs: prune_ratio (default 0.3), ridge (default 1e-2).
"""

from __future__ import annotations

from typing import Any, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...common.utils.log import LOGGER

from ..config import MethodSpec, PruneConfig
from ..context import PruneContext
from ..domains.base import BasePruningDomain
from ..domains.cnn import CNNChannelTarget, CNNPruningDomain
from ..report import PruneReport
from ..utils import (
    run_calibration_epoch,
    get_submodule,
    infer_model_device,
    topk_keep_indices,
)
from .base import BasePruningMethod


def _channel_saliency(acts: torch.Tensor, n_channels: int) -> torch.Tensor:
    """Per-channel saliency = rms of that channel's activation (robust to 4D/2D consumer input)."""
    a = acts.float()
    if a.dim() == 4:  # conv consumer input [N, C, H, W]
        return a.pow(2).mean(dim=(0, 2, 3)).sqrt()  # [C]
    # linear consumer input [N, C*span] (flattened): reshape per channel
    a = a.reshape(a.shape[0], n_channels, -1)
    return a.pow(2).mean(dim=(0, 2)).sqrt()  # [C]


@torch.no_grad()
def _reconstruct_conv_consumer(
    consumer: nn.Conv2d, x_keep: torch.Tensor, y: torch.Tensor, ridge: float
):
    """Reconstruct conv consumer weights on the kept input channels via im2col least squares."""
    k = consumer.kernel_size
    cols = F.unfold(
        x_keep.float(),
        kernel_size=k,
        padding=consumer.padding,
        stride=consumer.stride,
        dilation=consumer.dilation,
    )  # [N, keep*kh*kw, L]
    p = cols.permute(0, 2, 1).reshape(-1, cols.shape[1])  # [rows, keep*kh*kw]
    target = y.float().permute(0, 2, 3, 1).reshape(-1, y.shape[1])  # [rows, out]
    if consumer.bias is not None:
        target = target - consumer.bias.detach().float()
    if p.shape[0] < p.shape[1]:
        return None  # underdetermined: skip reconstruction, keep naive slice
    gram = p.t() @ p
    gram.diagonal().add_(ridge * gram.diagonal().mean())
    w = torch.linalg.solve(gram, p.t() @ target)  # [keep*khkw, out]
    return w.t().reshape(consumer.out_channels, len(x_keep[0]), k[0], k[1])


@torch.no_grad()
def _conv_bias_fold_delta(consumer, x_full, keep, total):
    """Mean-fold the removed input channels' average output contribution into the consumer bias.

    Zero the kept channels (leaving only the pruned ones) and run one consumer conv (no bias),
    taking the spatial/batch mean. Conv is linear in its input, so this is a robust O(1 forward)
    compensation with no least squares. Returns None if consumer is not Conv2d.
    """
    if not isinstance(consumer, nn.Conv2d):
        return None
    keep_mask = torch.zeros(total, dtype=torch.bool)
    keep_mask[torch.as_tensor(keep)] = True
    xp = x_full.float().clone()
    xp[:, keep_mask] = 0.0  # keep only the removed channels
    y_pruned = nn.functional.conv2d(
        xp,
        consumer.weight.detach().float(),
        None,
        stride=consumer.stride,
        padding=consumer.padding,
        dilation=consumer.dilation,
    )
    return y_pruned.mean(dim=(0, 2, 3))  # [out]


def _finalize_report(done, recovery, target, domain, method_name, total, keep, report):
    """Finalize after reconstruction: warn loudly when compensation was skipped, then record."""
    # Not silent: when the requested compensation could not run (underdetermined / Linear
    # consumer), explicitly warn that we fell back to a naive slice.
    if not done:
        msg = (
            f"[cnn.reconstruct] target '{target.producer_path}': recovery={recovery} not applied"
            f" (underdetermined: calibration rows < keep*k*k, or consumer is not Conv2d), fall back "
            f"to naive slicing. Increase calibration to enable."
        )
        LOGGER.logw(msg, "amct_prune")
        report.add_warning(msg)
    report.add(
        domain.name,
        method_name,
        target.producer_path,
        f"{recovery} prune {total} -> {len(keep)}",
    )
    report.record_layer_sparsity(target.producer_path, total, len(keep))


@torch.no_grad()
def _prune_and_reconstruct(
    model,
    domain,
    target,
    x_full,
    y,
    config,
    prune_ratio,
    ridge,
    recovery,
    report,
    method_name,
) -> bool:
    """One CNN target: pick kept channels -> compensate per ``recovery`` (ls/bias/none) -> prune."""
    producer = get_submodule(model, target.producer_path)
    consumer = get_submodule(model, target.consumer_path)
    total = int(producer.out_channels)
    if total <= config.min_neurons:
        return False
    saliency = _channel_saliency(x_full, total)
    keep_idx = topk_keep_indices(saliency, prune_ratio, config.min_neurons)
    if len(keep_idx) >= total:
        return False
    keep = sorted(int(i) for i in keep_idx)
    # recovery prep (before slicing; needs the full-width consumer / activations)
    delta_bias = (
        _conv_bias_fold_delta(consumer, x_full, keep, total)
        if recovery == "bias"
        else None
    )

    domain.apply_keep_indices(model, target, keep)
    new_consumer = get_submodule(model, target.consumer_path)
    done = recovery == "none"
    if recovery == "ls" and isinstance(new_consumer, nn.Conv2d):
        x_keep = x_full[:, torch.as_tensor(keep)]
        new_w = _reconstruct_conv_consumer(new_consumer, x_keep, y, ridge)
        if new_w is not None and tuple(new_w.shape) == tuple(new_consumer.weight.shape):
            new_consumer.weight.data = new_w.to(new_consumer.weight.dtype)
            done = True
    elif recovery == "bias" and delta_bias is not None:
        if getattr(new_consumer, "bias", None) is None:
            new_consumer.bias = nn.Parameter(
                torch.zeros(
                    delta_bias.shape[0],
                    dtype=new_consumer.weight.dtype,
                    device=new_consumer.weight.device,
                )
            )
        new_consumer.bias.data = new_consumer.bias.data + delta_bias.to(
            new_consumer.bias.dtype
        )
        done = True
    _finalize_report(done, recovery, target, domain, method_name, total, keep, report)
    return True


def _register_consumer_hooks(model, conv_targets):
    """Register a forward hook per target consumer, accumulating inputs x / outputs y over all
    calibration batches. Returns (caches, handles); caches[id(t)] = {"x": [...], "y": [...]}.
    """
    caches: dict = {}
    handles = []
    for t in conv_targets:
        mod = get_submodule(model, t.consumer_path)
        caches[id(t)] = {"x": [], "y": []}

        def hook(mod, inp, out, key=id(t)):
            # accumulate over all calibration batches (not just the last) so the
            # reconstruction sample count grows with the data
            caches[key]["x"].append(inp[0].detach())
            caches[key]["y"].append(out.detach())

        handles.append(mod.register_forward_hook(hook))
    return caches, handles


class ReconstructChannelPruningMethod(BasePruningMethod):
    """Output-preserving CNN channel pruning: cnn-domain sequential targets + least-squares
    reconstruction (low-damage, no third-party deps).

    kwargs: prune_ratio (default 0.3), ridge (default 1e-2),
        recovery -- "ls" (default, least-squares reconstruct consumer weights) / "bias"
        (mean-fold into consumer bias, no least squares, steadier on small calibration) /
        "none" (naive slice).
    """

    domain = "cnn"
    name = "reconstruct"
    requires_data = True
    RECOVERIES = ("ls", "bias", "none")

    def apply(
        self,
        model: nn.Module,
        domain: BasePruningDomain,
        targets: List[Any],
        context: PruneContext,
        report: PruneReport,
        config: PruneConfig,
        spec: MethodSpec,
    ) -> None:
        if context.data is None:
            raise ValueError("Reconstruct CNN pruning requires calibration data.")
        if not isinstance(domain, CNNPruningDomain):
            raise TypeError(
                "ReconstructChannelPruningMethod requires CNNPruningDomain."
            )
        prune_ratio = float(spec.kwargs.get("prune_ratio", 0.3))
        if prune_ratio <= 0:
            return
        ridge = float(spec.kwargs.get("ridge", 1e-2))
        recovery = str(spec.kwargs.get("recovery", "ls"))
        if recovery not in self.RECOVERIES:
            raise ValueError(
                f"recovery must be one of {self.RECOVERIES}, got {recovery!r}."
            )
        device = infer_model_device(model)

        conv_targets = [t for t in targets if isinstance(t, CNNChannelTarget)]
        caches, handles = _register_consumer_hooks(model, conv_targets)

        try:
            n_batches = run_calibration_epoch(model, context, device)
        finally:
            for h in handles:  # remove hooks even if calibration forward raised, avoiding leaks
                h.remove()
        if n_batches == 0:
            msg = "Reconstruct CNN pruning got an empty dataset; cnn stage skipped."
            LOGGER.logw(msg, "amct_prune")
            report.add_warning(msg)
            return

        for t in conv_targets:
            cache = caches.get(id(t)) or {}
            if not cache.get("x"):
                continue
            _prune_and_reconstruct(
                model,
                domain,
                t,
                torch.cat(cache["x"], 0),
                torch.cat(cache["y"], 0),
                config,
                prune_ratio,
                ridge,
                recovery,
                report,
                self.name,
            )
