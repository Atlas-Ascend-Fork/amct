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
"""MoE output-space expert merge pruning (output_merge).

The merge direction is in the MC-SMoE (ICLR'24)/ZipIt family; the output-space LS pairing, drop
fallback, and router logit-LS fit are this implementation's design.

Algorithm (validated in prototype v6):
1. One calibration forward per MoE block, collecting block input tokens (capped at 4096 to
   bound memory).
2. Greedy pairing by output similarity within the block: reconstruct the absorbing expert's last
   linear layer via usage-frequency-weighted least squares to approximate the merged expert's
   traffic; fall back to dropping when a plain drop (traffic redistributed by the remaining
   experts' softmax) costs less.
3. The router's absorbing row is fit by least squares in logit space on the calibration inputs
   (target = the logit equivalent of the summed group probability), including bias; elementwise
   logaddexp is not equivalent to p_a+p_b and was abandoned.
4. selector='calib_nll' (default): run this and the activation_count baseline on deep copies,
   land whichever has the lower calibration NLL, and record the choice in the report (never silent).

Only legacy ModuleList experts + Linear router are supported; modern fused expert-tensor
structures emit a loud warning and are skipped (no silent behavior change).
"""

from __future__ import annotations

import copy
import warnings
from typing import Dict, List, Optional, Tuple

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
    replace_submodule,
)
from .base import BasePruningMethod
from .moe_mass_pruning import MassMoEPruningMethod

# cap on calibration tokens collected per block (memory control)
_TOKEN_CAP = 4096
# common names of the last linear layer inside an expert (LLaMA / GPT / Mixtral style)
_LAST_LINEAR_NAMES = ("down_proj", "fc2", "w2", "out_proj")


def _make_input_hook(feats: Dict[int, List[torch.Tensor]], key: int):
    """forward_pre_hook that accumulates block input tokens, capped at ``_TOKEN_CAP``."""

    def hook(module, args):
        _ = module
        have = sum(t.shape[0] for t in feats[key])
        if have >= _TOKEN_CAP:
            return
        x = args[0].detach().reshape(-1, args[0].shape[-1]).float()
        feats[key].append(x[: _TOKEN_CAP - have])

    return hook


def _collect_block_inputs(
    model: nn.Module, blocks: List[nn.Module], context: PruneContext
) -> Dict[int, Optional[torch.Tensor]]:
    """One calibration forward, collecting each MoE block's input (keyed by block id; None if empty)."""
    feats: Dict[int, List[torch.Tensor]] = {id(b): [] for b in blocks}
    hooks = [b.register_forward_pre_hook(_make_input_hook(feats, id(b))) for b in blocks]
    device = infer_model_device(model)
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for args, kwargs in context.iter_model_inputs():
                moved_args, moved_kwargs = move_batch_to_device(args, kwargs, device)
                model(*moved_args, **moved_kwargs)
    finally:
        model.train(was_training)
        for h in hooks:
            h.remove()
    return {k: (torch.cat(v) if v else None) for k, v in feats.items()}


def _last_linear(expert: nn.Module) -> Optional[nn.Linear]:
    """Locate the expert's last linear layer (reconstruction target): by common name,
    else the last direct child Linear.
    """
    for name in _LAST_LINEAR_NAMES:
        lin = getattr(expert, name, None)
        if isinstance(lin, nn.Linear):
            return lin
    linears = [m for m in expert.children() if isinstance(m, nn.Linear)]
    return linears[-1] if linears else None


def _expert_hidden_outs(experts: nn.ModuleList, x: torch.Tensor):
    """Forward each expert, hooking the last linear's input hidden and the expert output. Returns None on mismatch."""
    hidden_list, out_list, last_list = [], [], []
    for expert in experts:
        lin = _last_linear(expert)
        if lin is None:
            return None
        cache: List[torch.Tensor] = []
        handle = lin.register_forward_pre_hook(lambda m, a, c=cache: c.append(a[0]))
        try:
            out = expert(x)
        finally:
            handle.remove()  # remove the pre-hook even if the expert forward raised, avoiding leaks
        hidden_list.append(cache[-1].reshape(-1, cache[-1].shape[-1]).float())
        out_list.append(out.float())
        last_list.append(lin)
    return torch.stack(out_list), hidden_list, last_list


def _solve_least_squares(gram: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    """Solve on the current device; fall back to CPU and back if the device lacks a solver
    (e.g. some NPUs have no LAPACK).
    """
    try:
        return torch.linalg.solve(gram, rhs)
    except RuntimeError:
        return torch.linalg.solve(gram.cpu(), rhs.cpu()).to(gram.device)


def _ls_fit(
    hidden_a: torch.Tensor,
    bias_a: Optional[torch.Tensor],
    probs: torch.Tensor,
    outs: torch.Tensor,
    grp: List[int],
) -> Tuple[torch.Tensor, float]:
    """Approximate the group traffic by weighted LS on the absorbing expert's hidden
    -> (new last-layer weight^T, weighted residual).
    """
    pg = probs[:, grp]
    w = pg.sum(1).clamp_min(1e-6)
    tgt = (pg[:, :, None] * outs[grp].permute(1, 0, 2)).sum(1) / w[:, None]
    if bias_a is not None:
        tgt = tgt - bias_a
    sw = w.sqrt()[:, None]
    hw = hidden_a * sw
    gram = hw.t() @ hw
    gram.diagonal().add_(1e-2 * gram.diagonal().mean())
    w_new = _solve_least_squares(gram, hw.t() @ (tgt * sw))
    resid = float((hw @ w_new - tgt * sw).pow(2).sum())
    return w_new, resid


def _drop_costs(probs: torch.Tensor, outs: torch.Tensor) -> List[float]:
    """Drop cost: weighted error when the expert's traffic is redistributed to the
    remaining experts' softmax-mean output.
    """
    e_n = outs.shape[0]
    costs = []
    for e in range(e_n):
        rest = [o for o in range(e_n) if o != e]
        mean_rest = (probs[:, rest, None] * outs[rest].permute(1, 0, 2)).sum(1)
        mean_rest = mean_rest / probs[:, rest].sum(1, keepdim=True).clamp_min(1e-6)
        costs.append(float((probs[:, e, None] * (outs[e] - mean_rest)).pow(2).sum()))
    return costs


def _best_action(groups, usage, drop_cost, probs, outs, hidden_list, biases):
    """Enumerate (absorber a, merged b): cost = min(LS residual, drop cost), return the best action."""
    best = None
    for b in list(groups):
        cost_drop = sum(drop_cost[e] for e in groups[b])
        for a in groups:
            if a == b or usage[a] < usage[b]:
                continue
            _, resid = _ls_fit(
                hidden_list[a], biases[a], probs, outs, groups[a] + groups[b]
            )
            cost = min(resid, cost_drop)
            if best is None or cost < best[0]:
                best = (cost, a, b, resid >= cost_drop)
    return best


def _greedy_grouping(probs, outs, hidden_list, biases, usage, n_merge, drop_cost):
    """LS-residual greedy pairing + drop fallback: run n_merge steps, return (groups, dropped)."""
    groups = {e: [e] for e in range(outs.shape[0])}
    dropped: List[int] = []
    for _ in range(n_merge):
        best = _best_action(groups, usage, drop_cost, probs, outs, hidden_list, biases)
        if best is None:
            break
        _, a, b, do_drop = best
        if do_drop:
            dropped.extend(groups.pop(b))
        else:
            groups[a] += groups.pop(b)
    return groups, dropped


def _fit_gate_row(gate: nn.Linear, x: torch.Tensor, grp: List[int], probs: torch.Tensor):
    """Calibration-LS fit of the absorbing row: in logit space solve x.w(+b) ~ log(sum group probs)+LSE(orig logits).

    The softmax over the remaining experts equals p_a+p_b (others unchanged) iff the absorbing
    row's logit takes that target value, so least squares on it fits the target probability.
    Returns (weight row, bias or None).
    """
    logits = gate(x).float()
    tgt = torch.log(probs[:, grp].sum(1).clamp_min(1e-12)) + torch.logsumexp(logits, -1)
    feat = x.float()
    if gate.bias is not None:
        feat = torch.cat([feat, torch.ones_like(feat[:, :1])], 1)
    gram = feat.t() @ feat
    gram.diagonal().add_(1e-6 * gram.diagonal().mean().clamp_min(1e-12))
    sol = _solve_least_squares(gram, feat.t() @ tgt[:, None]).squeeze(1)
    if gate.bias is not None:
        return sol[:-1], sol[-1]
    return sol, None


def _bake_merged_weights(gate, groups, hidden_list, biases, probs, outs, last_list, x):
    """Bake merged groups: absorber last-layer weight = LS solution; router absorbing row =
    calibration logit-space LS fit.
    """
    for a, grp in groups.items():
        if len(grp) < 2:
            continue
        w_new, _ = _ls_fit(hidden_list[a], biases[a], probs, outs, grp)
        last_list[a].weight.data = w_new.t().to(last_list[a].weight.dtype).contiguous()
        row, bias_val = _fit_gate_row(gate, x, grp, probs)
        gate.weight.data[a] = row.to(gate.weight.dtype)
        if bias_val is not None:
            gate.bias.data[a] = bias_val.to(gate.bias.dtype)


@torch.no_grad()
def _merge_target(model, domain, target, x_tokens, keep_ratio, top_k, config):
    """One MoE block: greedily merge/drop down to the target expert count, then prune via the
    domain. Returns (before, after, num_dropped).
    """
    gate = domain.get_router(model, target)
    experts = domain.get_experts(model, target)
    e_n = len(experts)
    n_keep = min(e_n, max(2, config.min_experts, int(round(e_n * keep_ratio))))
    if n_keep >= e_n or not isinstance(gate, nn.Linear):
        return None
    x = x_tokens.to(infer_model_device(model))
    fwd = _expert_hidden_outs(experts, x)
    if fwd is None:
        return None
    outs, hidden_list, last_list = fwd
    probs = torch.softmax(gate(x).float(), -1)
    biases = [None if layer.bias is None else layer.bias.detach().float() for layer in last_list]
    groups, dropped = _greedy_grouping(
        probs, outs, hidden_list, biases, probs.sum(0), e_n - n_keep, _drop_costs(probs, outs)
    )
    _bake_merged_weights(gate, groups, hidden_list, biases, probs, outs, last_list, x)
    keep_idx = sorted(groups.keys())
    domain.prune_experts(model, target, keep_idx, top_k)
    return e_n, len(keep_idx), len(dropped)


def _split_fused_targets(targets: List[MoETarget], report: PruneReport):
    """Modern fused expert structures are not yet supported for output merge: warn loudly and skip, never silent."""
    kept = []
    for target in targets:
        if target.fused:
            msg = (
                f"output_merge supports ModuleList experts only; skipping fused "
                f"experts '{target.experts_path}'."
            )
            warnings.warn(msg, RuntimeWarning, stacklevel=3)
            report.add_warning(msg)
        else:
            kept.append(target)
    return kept


def _apply_merge_all(
    method_name, model, domain, targets, context, report, config, keep_ratio, top_k
):
    """Run output merge on all target blocks; record per-block events and sparsity, warn loudly
    when a block is unprocessable.
    """
    blocks = [
        get_submodule(model, t.module_path) if t.module_path else model for t in targets
    ]
    feats = _collect_block_inputs(model, blocks, context)
    merged_any = False
    for target, blk in zip(targets, blocks):
        path = target.module_path or target.experts_path
        x = feats[id(blk)]
        result = None
        if x is not None and x.shape[0] > 0:
            result = _merge_target(model, domain, target, x, keep_ratio, top_k, config)
        if result is None:
            msg = f"output_merge skipped '{path}' (no calib tokens or unsupported expert structure)."
            warnings.warn(msg, RuntimeWarning, stacklevel=3)
            report.add_warning(msg)
            continue
        e_n, kept_n, n_drop = result
        report.add(
            domain.name,
            method_name,
            path,
            f"Output-merge experts {e_n} -> {kept_n} (dropped {n_drop})",
        )
        report.record_layer_sparsity(path, e_n, kept_n)
        merged_any = True
    return merged_any


def _is_token_tensor(value) -> bool:
    """True for a 2D integer token tensor."""
    return torch.is_tensor(value) and not value.is_floating_point() and value.dim() == 2


def _batch_token_ids(args, kwargs) -> Optional[torch.Tensor]:
    """Get token labels: prefer kwargs['input_ids'], else args[0]; never attention_mask."""
    ids = kwargs.get("input_ids")
    if _is_token_tensor(ids):
        return ids
    if args and _is_token_tensor(args[0]):
        return args[0]
    return None


@torch.no_grad()
def _calib_nll(model: nn.Module, context: PruneContext) -> Optional[float]:
    """Next-token NLL on the calibration set (no labels, never touches the test set); None if no token labels."""
    device = infer_model_device(model)
    total, count = 0.0, 0
    for args, kwargs in context.iter_model_inputs():
        moved_args, moved_kwargs = move_batch_to_device(args, kwargs, device)
        ids = _batch_token_ids(moved_args, moved_kwargs)
        if ids is None or ids.shape[1] < 2:
            return None
        logits = model(*moved_args, **moved_kwargs)
        if not torch.is_tensor(logits):
            logits = logits[0] if isinstance(logits, (tuple, list)) else logits.logits
        nll = nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, logits.shape[-1]), ids[:, 1:].reshape(-1)
        )
        total, count = total + float(nll), count + 1
    return total / count if count else None


def _transplant_targets(model, winner, targets, domain) -> int:
    """Transplant the winning copy's MoE blocks (experts + router) back into the original model,
    refreshing expert-count attrs.
    """
    num = 0
    for target in targets:
        replace_submodule(
            model, target.experts_path, get_submodule(winner, target.experts_path)
        )
        if target.router_path is not None:
            replace_submodule(
                model, target.router_path, get_submodule(winner, target.router_path)
            )
        num = domain.num_experts(model, target)
        domain.update_moe_attributes(model, target, num)
    return num


def _select_by_calib_nll(
    method_name, model, domain, targets, context, report, config, keep_ratio, top_k
) -> None:
    """A/B selector: run output_merge and activation_count on deep copies, land whichever has lower calibration NLL."""
    cand_merge = copy.deepcopy(model)
    cand_act = copy.deepcopy(model)
    merge_report = PruneReport(backend="selector", params_before=0)
    act_report = PruneReport(backend="selector", params_before=0)
    _apply_merge_all(
        method_name, cand_merge, domain, targets, context, merge_report, config, keep_ratio, top_k
    )
    act_spec = MethodSpec(
        name="activation_count", kwargs={"prune_ratio": 1.0 - keep_ratio, "top_k": top_k}
    )
    MassMoEPruningMethod().apply(cand_act, domain, targets, context, act_report, config, act_spec)
    nll_merge, nll_act = _calib_nll(cand_merge, context), _calib_nll(cand_act, context)
    if nll_merge is None or nll_act is None:
        msg = "output_merge selector: calib NLL unavailable (no token labels); keep merge."
        warnings.warn(msg, RuntimeWarning, stacklevel=3)
        report.add_warning(msg)
        winner, win_report, sel = cand_merge, merge_report, "output_merge"
    elif nll_merge <= nll_act:
        winner, win_report, sel = cand_merge, merge_report, "output_merge"
    else:
        winner, win_report, sel = cand_act, act_report, "activation_count"
    report.events.extend(win_report.events)
    report.warnings.extend(win_report.warnings)
    report.per_layer_sparsity.update(win_report.per_layer_sparsity)
    num = _transplant_targets(model, winner, targets, domain)
    report.add(
        domain.name,
        method_name,
        "<model>",
        f"calib_nll selector: merge={nll_merge} act_count={nll_act} -> selected {sel}",
    )
    meta = getattr(model, "_amct_prune_meta", {})
    meta["moe_num_experts"] = int(num)
    setattr(model, "_amct_prune_meta", meta)


class OutputMergeMoEPruningMethod(BasePruningMethod):
    """
    Merge MoE experts in output space (LS-residual greedy pairing, drop fallback).

    Pairs experts by output similarity, reconstructs the absorbing expert's last
    linear layer via traffic-weighted least squares, refits the absorbing router
    row (weight and bias) by calibration least squares in logit space so that
    its softmax matches the summed probability of the merged group.
    Default selector 'calib_nll' falls back to activation_count when
    the latter has lower calibration NLL (selection is always reported).

    kwargs:
        keep_ratio (float): Fraction of experts to keep. Default 0.50.
        prune_ratio (float): Optional alias (= 1 - keep_ratio).
        top_k (int): Experts per token (router patching). Default 8.
        selector (str): 'calib_nll' (default) or 'none'.
    """

    domain = "moe"
    name = "output_merge"
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
            raise ValueError(
                "MoE output_merge requires input data via prune(..., data=...)."
            )
        if not isinstance(domain, MoEPruningDomain):
            raise TypeError("OutputMergeMoEPruningMethod requires MoEPruningDomain.")
        keep_ratio = float(spec.kwargs.get("keep_ratio", 0.50))
        if "prune_ratio" in spec.kwargs:
            keep_ratio = 1.0 - float(spec.kwargs["prune_ratio"])
        top_k = int(spec.kwargs.get("top_k", 8))
        selector = str(spec.kwargs.get("selector", "calib_nll"))
        if selector not in ("calib_nll", "none"):
            raise ValueError(
                f"output_merge selector must be 'calib_nll' or 'none', got '{selector}'."
            )
        merge_targets = _split_fused_targets(targets, report)
        if not merge_targets:
            msg = "output_merge found no ModuleList MoE targets; model unchanged."
            warnings.warn(msg, RuntimeWarning, stacklevel=2)
            report.add_warning(msg)
            return
        if selector == "calib_nll":
            _select_by_calib_nll(
                self.name, model, domain, merge_targets, context, report, config, keep_ratio, top_k
            )
            return
        _apply_merge_all(
            self.name, model, domain, merge_targets, context, report, config, keep_ratio, top_k
        )
