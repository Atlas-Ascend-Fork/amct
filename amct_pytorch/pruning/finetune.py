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

from itertools import cycle, islice
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
    """Default loss: causal-LM for dict batches, cross-entropy for (x, y) tuples."""
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


def _move_to_device(batch, dev):
    if isinstance(batch, Mapping):
        return {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in batch.items()}
    if isinstance(batch, (tuple, list)):
        args, _ = move_batch_to_device(tuple(batch), {}, dev)
        return type(batch)(args)
    return batch.to(dev) if torch.is_tensor(batch) else batch


def _train_step(model, moved, loss_fn, batch_adapter, opt, params, grad_clip):
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
    return float(loss.detach())


def _set_warmup_lr(opt, lr, step, warmup):
    """Linear warmup: ramp lr from 0 to lr over the first ``warmup`` steps."""
    if warmup <= 0 or step >= warmup:
        return
    scaled = lr * (step + 1) / warmup
    for g in opt.param_groups:
        g["lr"] = scaled


def prune_finetune(
    model: nn.Module,
    data: Sequence[Any],
    loss_fn: Optional[LossFn] = None,
    steps: int = 200,
    lr: float = 1e-4,
    weight_decay: float = 0.0,
    warmup: int = 0,
    optimizer: Optional[Any] = None,
    batch_adapter: Optional[BatchAdapter] = None,
    device: Any = None,
    grad_clip: Optional[float] = 1.0,
    log_every: int = 0,
) -> dict:
    """Briefly fine-tune a (typically pruned) model in place to recover accuracy."""
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
    try:
        for i, batch in enumerate(islice(cycle(batches), steps)):
            _set_warmup_lr(opt, lr, i, warmup)
            moved = _move_to_device(batch, dev)
            history.append(
                _train_step(
                    model, moved, loss_fn, batch_adapter, opt, params, grad_clip
                )
            )
    finally:
        model.train(was_training)
        model.zero_grad(set_to_none=True)
    step = len(history)

    return {
        "steps": step,
        "initial_loss": history[0] if history else None,
        "final_loss": history[-1] if history else None,
        "loss_history": history if log_every <= 0 else history[::log_every],
    }
