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
"""基于依赖图(DepGraph)的 CNN 通道剪枝方法。

简单的“同 parent 兄弟卷积”启发式（cnn_variance）无法跟踪跨模块的数据流，
在 ResNet/ConvNext/MobileNet 等把卷积包进独立 block、且带残差/下采样捷径的
现代结构上会 0 目标或维度不一致而崩溃。

本方法改用 *依赖图*：跟踪一次前向的真实数据流，自动发现彼此耦合、必须一起
剪枝的层组（残差相加的多个生产者、下采样捷径、flatten→linear 等），逐组按
重要度裁剪通道。依赖图实现以 static 方式内置于
``amct_pytorch.experimental.pruning._vendor.torch_pruning``（改编自 MIT 许可的
Torch-Pruning，详见该目录 LICENSE），不引入任何外部依赖。
"""

from __future__ import annotations

from typing import Any, List

import torch.nn as nn

from ..config import MethodSpec, PruneConfig
from ..context import PruneContext
from ..domains.base import BasePruningDomain
from ..report import PruneReport
from ..utils import infer_model_device, move_batch_to_device
from .base import BasePruningMethod


import torch

# 需要梯度的重要度准则。与量化侧 ``TaylorLossSensitivity`` 同源思路：用梯度信息衡量
# "移除该通道对损失的影响"。taylor=一阶 |w·grad|；hessian=二阶(累积平方梯度, OBD)。
_GRAD_BASED_IMPORTANCE = frozenset({"taylor", "hessian"})
# hessian 需要**逐样本**反传 + importance.accumulate_grad()，流程与 taylor 不同。
_PERSAMPLE_GRAD_IMPORTANCE = frozenset({"hessian"})


def _select_importance(name: str):
    from .._vendor import torch_pruning as tp

    table = {
        "magnitude": lambda: tp.importance.MagnitudeImportance(p=2),
        "l1": lambda: tp.importance.MagnitudeImportance(p=1),
        "lamp": lambda: tp.importance.LAMPImportance(),
        "bn": lambda: tp.importance.BNScaleImportance(),
        "random": lambda: tp.importance.RandomImportance(),
        # 一阶泰勒：与量化的 TaylorLossSensitivity 结对齐（需要梯度，见 _populate_gradients）
        "taylor": lambda: tp.importance.GroupTaylorImportance(),
        # 二阶 Optimal Brain Damage（累积逐样本平方梯度，需 accumulate_grad）
        "hessian": lambda: tp.importance.GroupHessianImportance(),
        # FPGM：几何中位数滤波器相似度（仅前向，无需梯度）
        "fpgm": lambda: tp.importance.FPGMImportance(),
    }
    factory = table.get(str(name).lower())
    if factory is None:
        raise ValueError(f"Unknown importance '{name}'. Known: {sorted(table)}")
    return factory()


def _model_logits(model, args, kwargs):
    out = model(*args, **kwargs)
    if hasattr(out, "logits"):
        return out.logits
    return out if isinstance(out, torch.Tensor) else out[0]


def _populate_gradients(
    model: nn.Module,
    context: PruneContext,
    device,
    importance=None,
    per_sample: bool = False,
    max_batches: int = 4,
) -> bool:
    """为基于梯度的重要度在校准数据上反传，填充梯度信息。

    无标签场景用 **surrogate loss = 输出的均方**（对输出幅度的敏感度），等价于 Torch-Pruning
    文档里的 ``loss = loss_fn(model(x), y); loss.backward()``。
      * taylor：累加 ``param.grad``，一次/多批反传即可；
      * hessian(``per_sample=True``)：逐样本 ``backward`` + ``importance.accumulate_grad(model)``
        累积平方梯度（Optimal Brain Damage 流程）。
    返回是否成功产生了可用的梯度信息。
    """
    if per_sample and importance is not None:
        importance.zero_grad()
        n = 0
        for args, kwargs in context.iter_model_inputs():
            args, kwargs = move_batch_to_device(args, kwargs, device)
            logits = _model_logits(model, args, kwargs).float()
            per_sample_loss = logits.pow(2).mean(
                dim=tuple(range(1, logits.dim()))
            )  # 每个样本一个标量
            for li in per_sample_loss:
                model.zero_grad(set_to_none=True)
                li.backward(retain_graph=True)
                importance.accumulate_grad(model)
            n += 1
            if n >= max_batches:
                break
        return bool(getattr(importance, "_accu_grad", {}))

    model.zero_grad(set_to_none=True)
    n = 0
    for args, kwargs in context.iter_model_inputs():
        args, kwargs = move_batch_to_device(args, kwargs, device)
        loss = _model_logits(model, args, kwargs).float().pow(2).mean()
        loss.backward()
        n += 1
        if n >= max_batches:
            break
    if n == 0:
        return False
    return any(p.grad is not None for p in model.parameters())


def _example_inputs(context: PruneContext, device):
    """取校准数据的第一批作为依赖图追踪的样例输入。"""
    first = next(iter(context.iter_model_inputs()), None)
    if first is None:
        return None
    args, kwargs = first
    args, kwargs = move_batch_to_device(args, kwargs, device)
    if kwargs and not args:
        return dict(kwargs)
    if kwargs:
        return (tuple(args), dict(kwargs))
    if len(args) == 1:
        return args[0]
    return tuple(args)


def _auto_ignored_layers(model: nn.Module, skip_layers: List[str]) -> list:
    """自动忽略输出头(最后一个 Linear)与用户 skip_layers 命中的层，避免改变输出维度。"""
    ignored: list = []
    seen = set()

    def _add(m):
        if id(m) not in seen:
            seen.add(id(m))
            ignored.append(m)

    linears = [m for _, m in model.named_modules() if isinstance(m, nn.Linear)]
    if linears:
        _add(linears[-1])  # 分类头：输出维不可变
    if skip_layers:
        for name, m in model.named_modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)) and any(
                p in name for p in skip_layers
            ):
                _add(m)
    return ignored


class DepGraphChannelPruningMethod(BasePruningMethod):
    """依赖图驱动的结构化通道剪枝，支持残差/下采样等现代 CNN 结构。

    kwargs:
        prune_ratio (float): 通道剪枝比例。默认 0.30。
        importance (str): 通道重要度，默认 'magnitude'。可选：
            'magnitude'/'l1'/'lamp'/'bn'/'random'（仅前向）、
            'fpgm'（几何中位数，仅前向）、
            'taylor'（一阶泰勒 |w·grad|，需校准数据反传，思路同量化 TaylorLossSensitivity）、
            'hessian'（二阶 OBD，逐样本累积平方梯度，需校准数据）。
        global_pruning (bool): 全局统一稀疏（跨层按重要度阈值），便于部署。默认 False（逐层）。
        round_to (int|None): 把剪枝后通道数对齐到该倍数（如 8/16），利于 NPU/张量核硬件效率。默认 None。
        iterative_steps (int): 迭代剪枝步数（多步收敛优于一次性）。默认 1。
    """

    domain = "cnn"
    name = "depgraph"
    requires_data = True
    requires_targets = False  # 自行分析整图，不依赖域的兄弟卷积目标

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
        from .._vendor import torch_pruning as tp

        prune_ratio = float(spec.kwargs.get("prune_ratio", 0.30))
        if prune_ratio <= 0:
            return
        importance_name = str(spec.kwargs.get("importance", "magnitude")).lower()
        global_pruning = bool(spec.kwargs.get("global_pruning", False))
        round_to = spec.kwargs.get("round_to", None)
        iterative_steps = int(spec.kwargs.get("iterative_steps", 1))
        if iterative_steps < 1:
            raise ValueError("iterative_steps must be >= 1.")

        device = infer_model_device(model)
        example_inputs = _example_inputs(context, device)
        if example_inputs is None:
            raise ValueError(
                "DepGraph CNN pruning requires calibration data (example inputs)."
            )

        ignored = _auto_ignored_layers(model, config.skip_layers)
        importance = _select_importance(importance_name)
        is_grad = importance_name in _GRAD_BASED_IMPORTANCE
        per_sample = importance_name in _PERSAMPLE_GRAD_IMPORTANCE

        was_training = model.training
        model.eval()
        grad_note = ""
        try:
            pruner = tp.pruner.MetaPruner(
                model,
                example_inputs,
                importance=importance,
                pruning_ratio=prune_ratio,
                ignored_layers=ignored,
                global_pruning=global_pruning,
                round_to=round_to,
                iterative_steps=iterative_steps,
            )
            # 迭代剪枝：每步前为基于梯度的重要度重新填充梯度（权重已变）。
            for _ in range(iterative_steps):
                if is_grad:
                    ok = _populate_gradients(
                        model,
                        context,
                        device,
                        importance=importance,
                        per_sample=per_sample,
                    )
                    if not ok:
                        raise ValueError(
                            f"importance='{importance_name}' needs gradients but none were produced "
                            f"(check that calibration data is provided and the model is differentiable)."
                        )
                    grad_note = ", grads=populated"
                pruner.step()
        finally:
            model.zero_grad(set_to_none=True)
            model.train(was_training)

        opts = []
        if global_pruning:
            opts.append("global")
        if round_to:
            opts.append(f"round_to={round_to}")
        if iterative_steps > 1:
            opts.append(f"iter={iterative_steps}")
        opt_note = (", " + ",".join(opts)) if opts else ""
        report.add(
            self.domain,
            self.name,
            "<depgraph>",
            f"dependency-graph channel prune ratio={prune_ratio}, "
            f"importance={importance_name}, ignored_layers={len(ignored)}{grad_note}{opt_note}",
        )
