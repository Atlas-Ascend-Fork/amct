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

from typing import List, Optional

import torch
import torch.nn as nn

from ...common.utils.log import LOGGER

from ..config import MethodSpec, PruneConfig
from ..context import PruneContext
from ..domains.base import BasePruningDomain
from ..domains.dense import DensePruningDomain, GatedDenseTarget, TwoLayerDenseTarget
from ..report import PruneReport
from ..utils import (
    record_prune_size,
    record_prune_width,
    ridge_solve,
    get_submodule,
    infer_model_device,
    linear_like_weight_oi,
    move_batch_to_device,
    topk_keep_indices,
)
from .base import BasePruningMethod


def _consumer_path(target) -> Optional[str]:
    """Path of the Linear that takes the intermediate dim as *input* (gated down / two-layer consumer)."""
    return getattr(target, "down_path", None) or getattr(target, "consumer_path", None)


def _fake_quant_per_outchannel(w: torch.Tensor, bits: int) -> torch.Tensor:
    """Symmetric per-output-channel (per-row) fake quant for quant-aware saliency. w: [out, M]."""
    qmax = 2 ** (bits - 1) - 1
    scale = w.abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / qmax
    return (w / scale).round().clamp(-qmax - 1, qmax) * scale


def _fake_quant_amct(w: torch.Tensor, quant_cfg: dict) -> torch.Tensor:
    """Fake-quant saliency using AMCT's own min-max scale/offset; falls back to per-channel int8."""
    try:
        from amct_pytorch.classic.quantize_op.utils import (
            calculate_scale_offset,
            get_weight_min_max_by_granularity,
        )

        wmin, wmax = get_weight_min_max_by_granularity(w, quant_cfg)
        wc = quant_cfg.get("weights_cfg", {})
        bits = 4 if "4" in str(wc.get("dtype", "int8")) else 8
        scale, offset = calculate_scale_offset(
            wmax, wmin, wc.get("symmetric", True), f"int{bits}"
        )
        if offset is None:
            offset = torch.zeros_like(scale)
        qmax = 2 ** (bits - 1) - 1
        q = (w / scale + offset).round().clamp(-qmax - 1, qmax)
        return (q - offset) * scale
    except (
        RuntimeError,
        ValueError,
        TypeError,
        AttributeError,
        KeyError,
        IndexError,
    ) as exc:
        LOGGER.logw(
            f"quant_cfg saliency unavailable ({exc}); fallback per-channel int8",
            "amct_prune",
        )
        return _fake_quant_per_outchannel(w, 8)


def _consumer_bias(module: nn.Module) -> Optional[nn.Parameter]:
    return getattr(module, "bias", None)


def _prepare_recovery(
    acts_h: torch.Tensor,
    w_cons: torch.Tensor,
    keep: torch.Tensor,
    hidden: int,
    recovery: str,
    ridge: float,
    cp: str,
    report: PruneReport,
):
    """Pre-slice recovery prep per ``recovery`` (ls falls back to none if underdetermined).
    Returns (recovery, w_new, delta_bias)."""
    w_new = None
    delta_bias = None
    if recovery == "ls":
        h_keep = acts_h[:, keep]
        if h_keep.shape[0] < h_keep.shape[1]:
            msg = (
                f"[dense.reconstruct] target '{cp}': recovery=ls not executed"
                f" (underdetermined: calibration rows {h_keep.shape[0]} < kept neurons {h_keep.shape[1]}),"
                f" fall back to naive slicing. Increase calibration to enable."
            )
            LOGGER.logw(msg, "amct_prune")
            report.add_warning(msg)
            recovery = "none"
        else:
            y_full = acts_h @ w_cons.t()
            gram = h_keep.t() @ h_keep
            w_new = ridge_solve(gram, h_keep.t() @ y_full, ridge)
    elif recovery == "bias":
        mask = torch.ones(hidden, dtype=torch.bool, device=acts_h.device)
        mask[keep] = False
        delta_bias = w_cons[:, mask] @ acts_h[:, mask].mean(0)
    return recovery, w_new, delta_bias


@torch.no_grad()
def _reconstruct_one_target(
    model,
    domain,
    target,
    cp: str,
    chunks: list,
    config: PruneConfig,
    prune_ratio: float,
    ridge: float,
    recovery: str,
    quant_aware: bool,
    quant_aware_bits: int,
    quant_cfg: Optional[dict],
    report: PruneReport,
    method_name: str,
) -> None:
    """Pick kept neurons -> compensate per ``recovery`` (ls / bias mean-fold / none) -> prune."""
    acts_h = torch.cat(chunks, 0)
    hidden = acts_h.shape[1]
    if hidden <= config.min_neurons:
        return
    cons = get_submodule(model, cp)
    w_cons = linear_like_weight_oi(cons).data.float()
    if quant_cfg is not None:
        w_sal = _fake_quant_amct(w_cons, quant_cfg)
    elif quant_aware:
        w_sal = _fake_quant_per_outchannel(w_cons, quant_aware_bits)
    else:
        w_sal = w_cons
    saliency = w_sal.norm(dim=0) * acts_h.pow(2).mean(0).sqrt()
    keep_idx = topk_keep_indices(saliency, prune_ratio, config.min_neurons)
    if len(keep_idx) >= hidden:
        return
    keep = torch.as_tensor(sorted(int(i) for i in keep_idx), device=acts_h.device)

    recovery, w_new, delta_bias = _prepare_recovery(
        acts_h, w_cons, keep, hidden, recovery, ridge, cp, report
    )

    domain.apply_keep_indices(model, target, keep.tolist())
    new_cons = get_submodule(model, cp)

    if recovery == "ls":
        if isinstance(new_cons, nn.Linear):
            new_cons.weight.data = w_new.t().to(new_cons.weight.dtype)
        else:
            new_cons.weight.data = w_new.to(new_cons.weight.dtype)
    elif recovery == "bias":
        if _consumer_bias(new_cons) is None:
            new_cons.bias = nn.Parameter(
                torch.zeros(
                    delta_bias.shape[0],
                    dtype=new_cons.weight.dtype,
                    device=new_cons.weight.device,
                )
            )
        new_cons.bias.data = new_cons.bias.data + delta_bias.to(new_cons.bias.dtype)

    report.add(
        domain.name,
        method_name,
        cp,
        f"output-preserving ({recovery}) prune {hidden} -> {len(keep)}",
    )
    report.record_layer_sparsity(cp, hidden, len(keep))
    record_prune_size(model, "dense_hidden_size", len(keep))
    record_prune_width(model, "_dense_widths", len(keep))


def _collect_consumer_acts(model, targets, context, device):
    """Collect each target's consumer input activations H via forward hooks. Returns (acts, paths, n_batches)."""
    acts: dict = {}
    paths: dict = {}
    handles = []
    for t in targets:
        cp = _consumer_path(t)
        if cp is None:
            continue
        paths[id(t)] = cp
        acts[id(t)] = []
        mod = get_submodule(model, cp)

        def hook(mod, inp, out, key=id(t)):
            acts[key].append(inp[0].detach().reshape(-1, inp[0].shape[-1]).float())

        handles.append(mod.register_forward_hook(hook))

    n_batches = 0
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for args, kwargs in context.iter_model_inputs():
                args, kwargs = move_batch_to_device(args, kwargs, device)
                model(*args, **kwargs)
                n_batches += 1
    finally:
        for h in handles:
            h.remove()
        model.train(was_training)
    return acts, paths, n_batches


class ReconstructDensePruningMethod(BasePruningMethod):
    """Output-preserving FFN pruning: drop low-saliency intermediate neurons + reconstruct consumer weights.

    kwargs:
        prune_ratio (float): fraction of intermediate neurons to prune. Default 0.5.
        ridge (float): ridge coefficient for the reconstruction least squares. Default 1e-2.
        recovery (str): post-prune compensation -- "ls" (default), "bias", or "none".
    """

    domain = "dense"
    name = "reconstruct"
    accepted_kwargs = frozenset(
        {
            "prune_ratio",
            "ridge",
            "recovery",
            "quant_aware",
            "quant_aware_bits",
            "quant_cfg",
        }
    )
    requires_data = True
    RECOVERIES = ("ls", "bias", "none")

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
            raise ValueError(
                "Reconstruct dense pruning requires calibration data via prune(..., data=...)."
            )
        if not isinstance(domain, DensePruningDomain):
            raise TypeError(
                "ReconstructDensePruningMethod requires DensePruningDomain."
            )
        prune_ratio = float(spec.kwargs.get("prune_ratio", 0.5))
        if prune_ratio <= 0:
            return
        ridge = float(spec.kwargs.get("ridge", 1e-2))
        recovery = str(spec.kwargs.get("recovery", "ls"))
        if recovery not in self.RECOVERIES:
            raise ValueError(
                f"recovery must be one of {self.RECOVERIES}, got {recovery!r}."
            )
        quant_aware = bool(spec.kwargs.get("quant_aware", False))
        quant_aware_bits = int(spec.kwargs.get("quant_aware_bits", 8))
        quant_cfg = spec.kwargs.get("quant_cfg")
        device = infer_model_device(model)

        acts, paths, n_batches = _collect_consumer_acts(model, targets, context, device)

        if n_batches == 0:
            msg = "Reconstruct dense pruning got an empty dataset; dense stage skipped."
            LOGGER.logw(msg, "amct_prune")
            report.add_warning(msg)
            return

        for t in targets:
            if id(t) not in paths:
                continue
            chunks = acts.get(id(t)) or []
            if not chunks:
                continue
            _reconstruct_one_target(
                model,
                domain,
                t,
                paths[id(t)],
                chunks,
                config,
                prune_ratio,
                ridge,
                recovery,
                quant_aware,
                quant_aware_bits,
                quant_cfg,
                report,
                self.name,
            )
