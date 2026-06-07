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


def _select_importance(name: str):
    from .._vendor import torch_pruning as tp
    table = {
        "magnitude": lambda: tp.importance.MagnitudeImportance(p=2),
        "l1": lambda: tp.importance.MagnitudeImportance(p=1),
        "lamp": lambda: tp.importance.LAMPImportance(),
        "bn": lambda: tp.importance.BNScaleImportance(),
        "random": lambda: tp.importance.RandomImportance(),
    }
    factory = table.get(str(name).lower())
    if factory is None:
        raise ValueError(f"Unknown importance '{name}'. Known: {sorted(table)}")
    return factory()


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
        _add(linears[-1])                          # 分类头：输出维不可变
    if skip_layers:
        for name, m in model.named_modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)) and any(p in name for p in skip_layers):
                _add(m)
    return ignored


class DepGraphChannelPruningMethod(BasePruningMethod):
    """依赖图驱动的结构化通道剪枝，支持残差/下采样等现代 CNN 结构。

    kwargs:
        prune_ratio (float): 通道剪枝比例。默认 0.30。
        importance (str): 通道重要度，{'magnitude','l1','lamp','bn','random'}。默认 'magnitude'。
    """

    domain = "cnn"
    name = "depgraph"
    requires_data = True
    requires_targets = False     # 自行分析整图，不依赖域的兄弟卷积目标

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
        importance_name = spec.kwargs.get("importance", "magnitude")

        device = infer_model_device(model)
        example_inputs = _example_inputs(context, device)
        if example_inputs is None:
            raise ValueError("DepGraph CNN pruning requires calibration data (example inputs).")

        ignored = _auto_ignored_layers(model, config.skip_layers)
        was_training = model.training
        model.eval()
        try:
            pruner = tp.pruner.MetaPruner(
                model,
                example_inputs,
                importance=_select_importance(importance_name),
                pruning_ratio=prune_ratio,
                ignored_layers=ignored,
            )
            pruner.step()
        finally:
            model.train(was_training)

        report.add(
            self.domain, self.name, "<depgraph>",
            f"dependency-graph channel prune ratio={prune_ratio}, "
            f"importance={importance_name}, ignored_layers={len(ignored)}",
        )
