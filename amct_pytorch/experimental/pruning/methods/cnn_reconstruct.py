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
"""输出保留（重构）的 CNN 通道剪枝 —— 自研、低破坏、无任何外部/内置第三方依赖。

不依赖依赖图库：直接复用 ``cnn`` 域识别出的 (producer conv → bn → consumer) 顺序通道目标，
对每个目标剪掉低显著度的 producer 输出通道后，用最小二乘**重构 consumer 权重**来补偿，
使 consumer 的输出尽量保持（ThiNet/CP 式重构）：

    Y = im2col(X) @ W_flat^T               (X: consumer 输入, Y: 原输出)
    W_kept = argmin || Y - im2col(X[:, keep]) @ W^T || + ridge

相比朴素切片(``variance_channel``)，重构把被移除通道的贡献吸收进保留通道，精度损失更小。
仅处理 ``cnn`` 域能安全识别的顺序耦合（残差/分支边界处自然不被识别，从而避免破坏）。
kwargs: prune_ratio(默认 0.3), ridge(默认 1e-2)。
"""

from __future__ import annotations

import warnings
from typing import Any, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import MethodSpec, PruneConfig
from ..context import PruneContext
from ..domains.base import BasePruningDomain
from ..domains.cnn import CNNChannelTarget, CNNPruningDomain
from ..report import PruneReport
from ..utils import (
    get_submodule,
    infer_model_device,
    move_batch_to_device,
    topk_keep_indices,
)
from .base import BasePruningMethod


def _channel_saliency(acts: torch.Tensor, n_channels: int) -> torch.Tensor:
    """每个 producer 输出通道的显著度 = 该通道激活的 rms（对 4D/2D consumer 输入都稳健）。"""
    a = acts.float()
    if a.dim() == 4:  # conv consumer 输入 [N, C, H, W]
        return a.pow(2).mean(dim=(0, 2, 3)).sqrt()  # [C]
    # linear consumer 输入 [N, C*span]（flatten 后）：按通道分块
    a = a.reshape(a.shape[0], n_channels, -1)
    return a.pow(2).mean(dim=(0, 2)).sqrt()  # [C]


@torch.no_grad()
def _reconstruct_conv_consumer(
    consumer: nn.Conv2d, x_keep: torch.Tensor, y: torch.Tensor, ridge: float
):
    """im2col 最小二乘重构 conv consumer 在保留输入通道上的权重，使输出逼近 y。"""
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
        return None  # 欠定：放弃重构，保留朴素切片
    gram = p.t() @ p
    gram.diagonal().add_(ridge * gram.diagonal().mean())
    w = torch.linalg.solve(gram, p.t() @ target)  # [keep*khkw, out]
    return w.t().reshape(consumer.out_channels, len(x_keep[0]), k[0], k[1])


@torch.no_grad()
def _prune_and_reconstruct(
    model, domain, target, x_full, y, config, prune_ratio, ridge, report, method_name
) -> bool:
    """单个 CNN 目标：选保留通道 → 结构化剪枝 → 重构 conv consumer 权重。"""
    producer = get_submodule(model, target.producer_path)
    consumer = get_submodule(model, target.consumer_path)
    total = int(producer.out_channels)
    if total <= config.min_neurons:
        return False
    _ = consumer  # consumer 仅在重构阶段使用
    saliency = _channel_saliency(x_full, total)
    keep_idx = topk_keep_indices(saliency, prune_ratio, config.min_neurons)
    if len(keep_idx) >= total:
        return False
    keep = sorted(int(i) for i in keep_idx)
    domain.apply_keep_indices(model, target, keep)
    new_consumer = get_submodule(model, target.consumer_path)
    reconstructed = False
    if isinstance(new_consumer, nn.Conv2d):
        x_keep = x_full[:, torch.as_tensor(keep)]
        new_w = _reconstruct_conv_consumer(new_consumer, x_keep, y, ridge)
        if new_w is not None and tuple(new_w.shape) == tuple(new_consumer.weight.shape):
            new_consumer.weight.data = new_w.to(new_consumer.weight.dtype)
            reconstructed = True
    # 不静默：未能重构(欠定/Linear consumer)时**明确告知**退化为朴素切片，而非伪装成"已重构"
    if not reconstructed:
        msg = (
            f"[cnn.reconstruct] target '{target.producer_path}': 未做重构，退回朴素切片"
            f"（欠定: 校准行数 < keep*k*k，或 consumer 非 Conv2d）。增大/多样化校准数据可启用重构。"
        )
        warnings.warn(msg, RuntimeWarning, stacklevel=2)
        report.add_warning(msg)
    verb = "reconstruct" if reconstructed else "naive-slice (no reconstruct)"
    report.add(
        domain.name,
        method_name,
        target.producer_path,
        f"{verb} prune {total} -> {len(keep)}",
    )
    report.record_layer_sparsity(target.producer_path, total, len(keep))
    return True


def _run_calib(model: nn.Module, context: PruneContext, device) -> int:
    """跑校准前向以触发 hook 收集激活；返回处理的 batch 数。"""
    was_training = model.training
    model.eval()
    n = 0
    try:
        with torch.no_grad():
            for args, kwargs in context.iter_model_inputs():
                args, kwargs = move_batch_to_device(args, kwargs, device)
                model(*args, **kwargs)
                n += 1
    finally:
        model.train(was_training)
    return n


class ReconstructChannelPruningMethod(BasePruningMethod):
    """自研输出保留 CNN 通道剪枝：cnn 域顺序目标 + 最小二乘重构补偿（低破坏，无第三方依赖）。

    kwargs: prune_ratio(默认 0.3), ridge(默认 1e-2)。
    """

    domain = "cnn"
    name = "reconstruct"
    requires_data = True

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
        device = infer_model_device(model)

        conv_targets = [t for t in targets if isinstance(t, CNNChannelTarget)]
        caches: dict = {}
        handles = []
        for t in conv_targets:
            mod = get_submodule(model, t.consumer_path)
            caches[id(t)] = {"x": [], "y": []}

            def hook(mod, inp, out, key=id(t)):
                # 累积所有校准批次（而非仅最后一批），使重构样本数随数据量增长
                caches[key]["x"].append(inp[0].detach())
                caches[key]["y"].append(out.detach())

            handles.append(mod.register_forward_hook(hook))

        n_batches = _run_calib(model, context, device)
        for h in handles:
            h.remove()
        if n_batches == 0:
            msg = "Reconstruct CNN pruning got an empty dataset; cnn stage skipped."
            warnings.warn(msg, RuntimeWarning, stacklevel=2)
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
                report,
                self.name,
            )
