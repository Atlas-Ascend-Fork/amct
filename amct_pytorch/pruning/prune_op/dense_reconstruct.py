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
"""Output-preserving (reconstruct) dense FFN pruning -- low loss, no fine-tuning.

After removing the lowest-saliency intermediate neurons, the consumer (down) weights are
reconstructed by least squares so the FFN block output is preserved:

    y = H @ W_down^T          (H: down input activations, [N, M])
    keep subset of {0..M-1}   (neurons kept by saliency)
    W_kept = argmin || y - H[:, keep] @ W^T || + ridge        (reconstruct on kept columns)

The removed neurons' contribution is absorbed into the kept neurons, so the block output is
nearly unchanged -- unlike one-shot pruning that drops accuracy and relies on fine-tuning to
recover. The reconstruction idea is in the ThiNet/Channel-Pruning (ICCV'17) family; this is a
self-contained implementation for eager FFNs.

Important: the calibration set must be representative and diverse, with far more tokens than
the intermediate dim M. Otherwise the activations are rank-deficient, least squares overfits,
and held-out diverges. kwargs: prune_ratio (default 0.5), ridge (default 1e-2).
"""

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
    """Fake-quant saliency using AMCT's own min-max scale/offset (identical to calibrate).

    quant_cfg mirrors the amct.quantize weight config: {"weights_cfg": {"strategy":
    "channel"|"tensor", "symmetric": bool, "dtype": "INT8"|"INT4", ...}}; falls back to
    symmetric per-channel int8 when unavailable.
    """
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
    except Exception as exc:  # explicit warn+fallback when AMCT quant helpers unavailable (no silent)
        LOGGER.logw(
            f"quant_cfg saliency unavailable ({exc}); fallback per-channel int8", "amct_prune")
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
    """Recovery prep before slicing: compute reconstruct weights w_new / mean-fold delta_bias
    per ``recovery``. Needs full-width weights/activations, so it must run before slicing. When
    ls is underdetermined (calib rows < kept neurons), warn loudly and fall back to a naive
    slice, returning the demoted recovery alongside. Returns (recovery, w_new, delta_bias).
    """
    w_new = None
    delta_bias = None
    if recovery == "ls":
        # reconstruct: min || (H @ Wd^T) - H[:, keep] @ W^T ||  (ridge)
        h_keep = acts_h[:, keep]
        if h_keep.shape[0] < h_keep.shape[1]:
            # underdetermined: calib rows < kept neurons -> ls overfits (same as cnn path).
            # Not silent: warn loudly + fall back to naive slice.
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
            gram.diagonal().add_(ridge * gram.diagonal().mean())
            w_new = torch.linalg.solve(gram, h_keep.t() @ y_full)  # [keep, out]
    elif recovery == "bias":
        # mean-fold: add the removed neurons' average contribution sum_pruned mean(H_c)*W[:,c]
        # into the consumer bias. O(M), no least squares, does not diverge when calib rows < M
        # (robust compensation for small calibration sets).
        mask = torch.ones(hidden, dtype=torch.bool, device=acts_h.device)
        mask[keep] = False
        delta_bias = w_cons[:, mask] @ acts_h[:, mask].mean(0)  # [out]
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
    acts_h = torch.cat(chunks, 0)  # [N, M]
    hidden = acts_h.shape[1]
    if hidden <= config.min_neurons:
        return
    cons = get_submodule(model, cp)
    w_cons = linear_like_weight_oi(cons).data.float()  # [out, M] (Conv1D transpose normalized)
    # saliency weights: when quant-aware, score on fake-quantized weights (GETA), else fp weights
    if quant_cfg is not None:
        w_sal = _fake_quant_amct(w_cons, quant_cfg)
    elif quant_aware:
        w_sal = _fake_quant_per_outchannel(w_cons, quant_aware_bits)
    else:
        w_sal = w_cons
    saliency = w_sal.norm(dim=0) * acts_h.pow(2).mean(0).sqrt()  # [M]
    keep_idx = topk_keep_indices(saliency, prune_ratio, config.min_neurons)
    if len(keep_idx) >= hidden:
        return
    keep = torch.as_tensor(sorted(int(i) for i in keep_idx), device=acts_h.device)

    # recovery prep (before slicing; needs full-width weights / activations)
    recovery, w_new, delta_bias = _prepare_recovery(
        acts_h, w_cons, keep, hidden, recovery, ridge, cp, report
    )

    domain.apply_keep_indices(model, target, keep.tolist())
    new_cons = get_submodule(model, cp)

    if recovery == "ls":
        # w_new is [keep, out]: Linear weight layout [out, in] needs transpose; Conv1D layout
        # [in, out] is used directly.
        if isinstance(new_cons, nn.Linear):
            new_cons.weight.data = w_new.t().to(new_cons.weight.dtype)
        else:  # Conv1D: weight [in=keep, out]
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
    meta = getattr(model, "_amct_prune_meta", {})
    meta["dense_hidden_size"] = len(keep)  # for patch_common_config to update intermediate_size
    meta.setdefault("_dense_widths", set()).add(len(keep))  # per-layer widths: compat refuses scalar when widths differ
    setattr(model, "_amct_prune_meta", meta)


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
    """Output-preserving FFN pruning: remove low-saliency intermediate neurons + reconstruct
    consumer weights (low loss, no fine-tuning).

    kwargs:
        prune_ratio (float): fraction of intermediate neurons to prune. Default 0.5.
        ridge (float): ridge coefficient for the reconstruction least squares (relative to the
            mean of A's diagonal). Default 1e-2.
        recovery (str): post-prune compensation -- "ls" (default, least-squares reconstruct
            consumer weights; best with ample calibration) / "bias" (mean-fold into consumer
            bias, O(M) no least squares, robust even when calib rows < intermediate dim M) /
            "none" (naive slice, no compensation). With tiny calibration "ls" overfits when
            underdetermined; "bias" is steadier.
    """

    domain = "dense"
    name = "reconstruct"
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
        # quant-aware saliency (GETA, practical core of CVPR'25 Eq.9): score neurons on the
        # quantized weights, aligning "which to prune" with the deployment quant scheme, not
        # just search. Small effect at int8; significant only at int4/low-bit.
        quant_aware = bool(spec.kwargs.get("quant_aware", False))
        quant_aware_bits = int(spec.kwargs.get("quant_aware_bits", 8))
        # pass AMCT quant config through: saliency fake-quant matches amct.quantize calibrate
        # exactly scale,offset,granularity
        quant_cfg = spec.kwargs.get("quant_cfg")
        device = infer_model_device(model)

        # 1) collect each target's consumer input activations H (intermediate-neuron activations)
        acts, paths, n_batches = _collect_consumer_acts(model, targets, context, device)

        if n_batches == 0:
            msg = "Reconstruct dense pruning got an empty dataset; dense stage skipped."
            LOGGER.logw(msg, "amct_prune")
            report.add_warning(msg)
            return

        # 2) per target: reconstruct (extracted into a module-level helper to bound function size)
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
