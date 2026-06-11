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
的一次性剪枝不同。重构思想属 ThiNet/Channel Pruning (ICCV'17) 一族；本实现自研、面向 eager FFN。
与既有 graph 子系统的 TaylorLossSensitivity/重训练同源，但面向 eager 模型、即插即用。

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


def _fake_quant_amct(w: torch.Tensor, quant_cfg: dict) -> torch.Tensor:
    """用 AMCT 自身的 min-max scale/offset（与 calibrate 完全同式）做假量化显著度。

    quant_cfg 与 amct.quantize 权重配置同构：{"weights_cfg": {"strategy": "channel"|"tensor",
    "symmetric": bool, "dtype": "INT8"|"INT4", ...}}；不可用时回退对称 per-channel int8。
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
    except Exception as exc:  # AMCT 量化 helper 不可用时显式告警回退（不可静默）
        warnings.warn(
            f"quant_cfg saliency unavailable ({exc}); fallback per-channel int8"
        )
        return _fake_quant_per_outchannel(w, 8)


def _consumer_bias(module: nn.Module) -> Optional[nn.Parameter]:
    """返回 consumer 的 bias 参数（Linear.bias / Conv1D.bias），无则 None。"""
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
    """切片前的 recovery 准备：按 ``recovery`` 算出重构权重 w_new / 均值折叠 delta_bias。

    纯抽取自 :func:`_reconstruct_one_target`（需要全宽权重/激活，故必须在切片前）。
    ls 欠定（校准行数 < 保留神经元）时明确告警 + 退回朴素切片，并把退回后的 recovery 一并返回。
    返回 (recovery, w_new, delta_bias)。"""
    w_new = None
    delta_bias = None
    if recovery == "ls":
        # 重构：min || (H @ Wd^T) - H[:, keep] @ W^T ||  (ridge)
        h_keep = acts_h[:, keep]
        if h_keep.shape[0] < h_keep.shape[1]:
            # 欠定：校准行数 < 保留神经元数 → ls 过拟合（与 cnn 路径一致）。不静默：明确告警 + 退回朴素切片。
            msg = (
                f"[dense.reconstruct] target '{cp}': recovery=ls 未执行"
                f"（欠定: 校准行数 {h_keep.shape[0]} < 保留神经元 {h_keep.shape[1]}），退回朴素切片。增大校准可启用。"
            )
            warnings.warn(msg, RuntimeWarning, stacklevel=2)
            report.add_warning(msg)
            recovery = "none"
        else:
            y_full = acts_h @ w_cons.t()
            gram = h_keep.t() @ h_keep
            gram.diagonal().add_(ridge * gram.diagonal().mean())
            w_new = torch.linalg.solve(gram, h_keep.t() @ y_full)  # [keep, out]
    elif recovery == "bias":
        # 均值折叠：把被移除神经元的**平均贡献** sum_pruned mean(H_c)·W[:,c] 加进 consumer bias。
        # O(M)、无需最小二乘，校准行数 < M 也不发散（小校准集的稳健补偿）。
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
    """选保留神经元 → 按 ``recovery`` 补偿（ls 最小二乘 / bias 均值折叠 / none 朴素切片）→ 结构化剪枝。"""
    acts_h = torch.cat(chunks, 0)  # [N, M]
    hidden = acts_h.shape[1]
    if hidden <= config.min_neurons:
        return
    cons = get_submodule(model, cp)
    w_cons = linear_like_weight_oi(cons).data.float()  # [out, M] (Conv1D 转置已归一)
    # 显著度权重：量化感知时用假量化后的权重打分（GETA），否则原 fp 权重
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

    # recovery 准备（在切片前，需要全宽权重 / 激活）
    recovery, w_new, delta_bias = _prepare_recovery(
        acts_h, w_cons, keep, hidden, recovery, ridge, cp, report
    )

    domain.apply_keep_indices(model, target, keep.tolist())
    new_cons = get_submodule(model, cp)

    if recovery == "ls":
        # w_new 是 [keep, out]：Linear 权重布局 [out, in] 需转置；Conv1D 布局 [in, out] 直接用。
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
    meta["dense_hidden_size"] = len(keep)  # 供 patch_common_config 更新 intermediate_size
    meta.setdefault("_dense_widths", set()).add(len(keep))  # 逐层宽度集合：非同宽时 compat 拒写标量
    setattr(model, "_amct_prune_meta", meta)


def _collect_consumer_acts(model, targets, context, device):
    """用 forward hook 收集每个 target 的 consumer 输入激活 H。返回 (acts, paths, n_batches)。"""
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
    """输出保留的 FFN 剪枝：移除低显著度中间神经元 + 重构 consumer 权重（低损失，无需微调）。

    kwargs:
        prune_ratio (float): 中间神经元剪枝比例。默认 0.5。
        ridge (float): 重构最小二乘的岭正则系数（相对 A 对角均值）。默认 1e-2。
        recovery (str): 剪枝后补偿方式 —— "ls"(默认, 最小二乘重构 consumer 权重；校准充分时最佳) /
            "bias"(均值折叠到 consumer bias，O(M) 无最小二乘，校准行数 < 中间维 M 时仍稳健) /
            "none"(朴素切片，不补偿)。校准很小时 "ls" 欠定会过拟合，"bias" 更稳。
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
        # 量化感知显著度 (GETA, CVPR'25 Eq.9 的实用核心)：用**量化后**的权重给神经元打分，
        # 使"剪谁"也对齐部署量化方案，而不仅是搜索阶段。int8 影响小、int4/低位才显著。
        quant_aware = bool(spec.kwargs.get("quant_aware", False))
        quant_aware_bits = int(spec.kwargs.get("quant_aware_bits", 8))
        # AMCT 量化配置直传：显著度假量化与 amct.quantize 的 calibrate 完全同式（scale/offset/粒度）
        quant_cfg = spec.kwargs.get("quant_cfg")
        device = infer_model_device(model)

        # 1) 用 hook 收集每个 target 的 consumer 输入激活（中间神经元的激活 H）
        acts, paths, n_batches = _collect_consumer_acts(model, targets, context, device)

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
                recovery,
                quant_aware,
                quant_aware_bits,
                quant_cfg,
                report,
                self.name,
            )
