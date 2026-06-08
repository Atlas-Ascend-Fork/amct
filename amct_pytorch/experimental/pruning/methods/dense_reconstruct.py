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
"""输出保留（重构）的 dense FFN 剪枝 —— 低损失、无需微调。

移除显著度最低的中间神经元后，用最小二乘**重构 consumer(down) 权重**，使 FFN 块的输出尽量保持：

    y = H @ W_down^T          (H: down 的输入激活, [N, M])
    keep ⊂ {0..M-1}           (按显著度保留的神经元)
    W_kept = argmin || y - H[:, keep] @ W^T || + ridge        (在保留列上重构)

被移除神经元的贡献被**吸收进保留的神经元**，所以块输出近似不变 —— 这与“砍掉就掉精度、再靠微调救回”
的一次性剪枝不同。与既有 graph 子系统的 TaylorLossSensitivity/重训练同源，但面向 eager 模型、即插即用。

要求（重要）：校准集必须**有代表性且多样**，且 token 数远多于中间维 M。否则激活秩亏、最小二乘过拟合，
held-out 上会发散。kwargs: prune_ratio(默认 0.5), ridge(默认 1e-2)。
"""

from __future__ import annotations

import warnings
from typing import List, Optional

import torch
import torch.nn as nn

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
    """返回中间维作为 *输入* 的那个 Linear 的路径（gated 的 down / 两层的 consumer）。"""
    return getattr(target, "down_path", None) or getattr(target, "consumer_path", None)


def _fake_quant_per_outchannel(w: torch.Tensor, bits: int) -> torch.Tensor:
    """对称 per-output-channel (逐行) 假量化，用于量化感知显著度。w: [out, M]。"""
    qmax = 2 ** (bits - 1) - 1
    scale = w.abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / qmax
    return (w / scale).round().clamp(-qmax - 1, qmax) * scale


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
    quant_aware: bool,
    quant_aware_bits: int,
    report: PruneReport,
    method_name: str,
) -> None:
    """选保留神经元 → 最小二乘重构 consumer 权重 → 结构化剪枝 + 覆盖权重（单个 target）。"""
    acts_h = torch.cat(chunks, 0)  # [N, M]
    hidden = acts_h.shape[1]
    if hidden <= config.min_neurons:
        return
    cons = get_submodule(model, cp)
    w_cons = linear_like_weight_oi(cons).data.float()  # [out, M] (Conv1D 转置已归一)
    # 显著度权重：量化感知时用假量化后的权重打分（GETA），否则原 fp 权重
    w_sal = (
        _fake_quant_per_outchannel(w_cons, quant_aware_bits) if quant_aware else w_cons
    )
    saliency = w_sal.norm(dim=0) * acts_h.pow(2).mean(0).sqrt()  # [M]
    keep_idx = topk_keep_indices(saliency, prune_ratio, config.min_neurons)
    if len(keep_idx) >= hidden:
        return
    keep = torch.as_tensor(sorted(int(i) for i in keep_idx), device=acts_h.device)
    # 重构：min || (H @ Wd^T) - H[:, keep] @ W^T ||  (ridge)
    y_full = acts_h @ w_cons.t()
    h_keep = acts_h[:, keep]
    gram = h_keep.t() @ h_keep
    gram.diagonal().add_(ridge * gram.diagonal().mean())
    w_new = torch.linalg.solve(gram, h_keep.t() @ y_full)  # [keep, out]
    domain.apply_keep_indices(model, target, keep.tolist())
    # w_new 是 [keep, out]：Linear 权重布局 [out, in] 需转置；Conv1D 布局 [in, out] 直接用。
    new_cons = get_submodule(model, cp)
    if isinstance(new_cons, nn.Linear):
        new_cons.weight.data = w_new.t().to(new_cons.weight.dtype)
    else:  # Conv1D: weight [in=keep, out]
        new_cons.weight.data = w_new.to(new_cons.weight.dtype)
    report.add(
        domain.name,
        method_name,
        cp,
        f"output-preserving reconstruct prune {hidden} -> {len(keep)}",
    )
    report.record_layer_sparsity(cp, hidden, len(keep))


class ReconstructDensePruningMethod(BasePruningMethod):
    """输出保留的 FFN 剪枝：移除低显著度中间神经元 + 重构 consumer 权重（低损失，无需微调）。

    kwargs:
        prune_ratio (float): 中间神经元剪枝比例。默认 0.5。
        ridge (float): 重构最小二乘的岭正则系数（相对 A 对角均值）。默认 1e-2。
    """

    domain = "dense"
    name = "reconstruct"
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
        # 量化感知显著度 (GETA, CVPR'25 Eq.9 的实用核心)：用**量化后**的权重给神经元打分，
        # 使"剪谁"也对齐部署量化方案，而不仅是搜索阶段。int8 影响小、int4/低位才显著。
        quant_aware = bool(spec.kwargs.get("quant_aware", False))
        quant_aware_bits = int(spec.kwargs.get("quant_aware_bits", 8))
        device = infer_model_device(model)

        # 1) 用 hook 收集每个 target 的 consumer 输入激活（中间神经元的激活 H）
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

        if n_batches == 0:
            msg = "Reconstruct dense pruning got an empty dataset; dense stage skipped."
            warnings.warn(msg, RuntimeWarning, stacklevel=2)
            report.add_warning(msg)
            return

        # 2) 逐 target：重构（提取为模块级 helper，控制函数体规模）
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
                quant_aware,
                quant_aware_bits,
                report,
                self.name,
            )
