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
"""剪枝后的**可选**微调（恢复精度）。

结构化剪枝是一次性裁剪，会掉精度；少量微调通常能恢复大部分。本模块提供一个轻量、
与框架无关的 ``finetune`` 工具，作为 prune → (finetune) → quantize 流程中的可选一步，
与既有 graph 子系统的 ``create_prune_retrain_model`` 思路一致，但面向 eager 模型、即插即用。
"""

from __future__ import annotations

from typing import Any, Callable, List, Mapping, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from .context import BatchAdapter
from .utils import infer_model_device, move_batch_to_device

LossFn = Callable[[nn.Module, Any], torch.Tensor]


def _default_loss(
    model: nn.Module, batch: Any, batch_adapter: Optional[BatchAdapter]
) -> torch.Tensor:
    """无需用户指定时的默认损失：

    - dict 且含 ``input_ids``：因果语言建模自监督损失（``labels=input_ids``）；
    - ``(x, y)`` 二元组：交叉熵分类损失；
    - 其他：报错，要求显式提供 ``loss_fn``。
    """
    if batch_adapter is not None:
        args, kwargs = batch_adapter(batch)
        out = model(*args, **kwargs)
        if hasattr(out, "loss") and out.loss is not None:
            return out.loss
        raise ValueError(
            "batch_adapter output produced no .loss; pass an explicit loss_fn."
        )
    if isinstance(batch, Mapping):
        if "labels" in batch:
            return model(**batch).loss
        if "input_ids" in batch:
            return model(**batch, labels=batch["input_ids"]).loss
        raise ValueError(
            "dict batch needs 'labels' or 'input_ids' for default loss; else pass loss_fn."
        )
    if isinstance(batch, (tuple, list)) and len(batch) == 2:
        x, y = batch
        out = model(x)
        logits = out.logits if hasattr(out, "logits") else out
        return F.cross_entropy(logits, y)
    raise ValueError(
        "Cannot derive a default loss from this batch; pass loss_fn(model, batch) -> loss."
    )


def finetune(
    model: nn.Module,
    data: Sequence[Any],
    *,
    loss_fn: Optional[LossFn] = None,
    steps: int = 200,
    lr: float = 1e-4,
    weight_decay: float = 0.0,
    optimizer: Optional[Any] = None,
    batch_adapter: Optional[BatchAdapter] = None,
    device: Any = None,
    grad_clip: Optional[float] = 1.0,
    log_every: int = 0,
) -> dict:
    """对（通常是剪枝后的）模型做少量微调以恢复精度。**可选**步骤。

    Parameters
    ----------
    model : nn.Module
        待微调模型（in-place 更新权重）。
    data : Sequence
        训练批的可迭代集合（会被循环复用直到达到 ``steps``）。
    loss_fn : Callable[[model, batch], Tensor] | None
        损失函数。None 时用 :func:`_default_loss`（因果 LM 自监督 / (x,y) 交叉熵）。
    steps : int
        梯度步数。lr / weight_decay / optimizer / grad_clip 为常规训练超参。
    log_every : int
        >0 时每隔该步数在返回的 history 里记录一次（始终记录每步 loss）。

    Returns
    -------
    dict : ``{"steps", "initial_loss", "final_loss", "loss_history"}``
    """
    if steps <= 0:
        return {
            "steps": 0,
            "initial_loss": None,
            "final_loss": None,
            "loss_history": [],
        }
    batches: List[Any] = list(data)
    if not batches:
        raise ValueError("finetune needs at least one training batch.")

    dev = device if device is not None else infer_model_device(model)
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise ValueError("model has no trainable parameters.")
    opt = (
        optimizer
        if optimizer is not None
        else torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    )

    was_training = model.training
    model.train()
    history: List[float] = []
    step = 0
    try:
        while step < steps:
            for batch in batches:
                if step >= steps:
                    break
                if isinstance(batch, Mapping):
                    moved = {
                        k: (v.to(dev) if torch.is_tensor(v) else v)
                        for k, v in batch.items()
                    }
                elif isinstance(batch, (tuple, list)):
                    args, _ = move_batch_to_device(tuple(batch), {}, dev)
                    moved = type(batch)(args)
                else:
                    moved = batch.to(dev) if torch.is_tensor(batch) else batch

                loss = (
                    loss_fn(model, moved)
                    if loss_fn is not None
                    else _default_loss(model, moved, batch_adapter)
                )
                opt.zero_grad(set_to_none=True)
                loss.backward()
                if grad_clip:
                    torch.nn.utils.clip_grad_norm_(params, grad_clip)
                opt.step()
                history.append(float(loss.detach()))
                step += 1
    finally:
        model.train(was_training)
        model.zero_grad(set_to_none=True)

    return {
        "steps": step,
        "initial_loss": history[0] if history else None,
        "final_loss": history[-1] if history else None,
        "loss_history": history if log_every <= 0 else history[::log_every],
    }
