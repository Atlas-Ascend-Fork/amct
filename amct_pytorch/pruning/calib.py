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

from typing import Any, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import nn
from .context import BatchAdapter, PruneContext
from .utils import move_batch_to_device


def _unwrap_logits(out: Any) -> torch.Tensor:
    if torch.is_tensor(out):
        return out
    if isinstance(out, (tuple, list)) and out:
        return out[0]
    if hasattr(out, "logits"):
        return out.logits
    raise TypeError(f"cannot extract logits from model output of type {type(out)!r}")


def nll_from_logits(logits: torch.Tensor, ids: torch.Tensor) -> tuple[float, int]:
    """Next-token cross-entropy sum and token count for one batch (logits[:, :-1] vs ids[:, 1:])."""
    target = ids[:, 1:].reshape(-1)
    nll_sum = F.cross_entropy(
        logits[:, :-1].reshape(-1, logits.shape[-1]),
        target,
        reduction="sum",
    )
    return float(nll_sum), int(target.numel())


def calib_nll(
    model: nn.Module, data: Sequence[Any], batch_adapter: Optional[BatchAdapter] = None
) -> float:
    """Mean autoregressive NLL over calibration token batches (label-free, lower is better)."""
    batches = list(data)
    if not batches:
        raise ValueError("calib_nll needs at least one calibration batch.")
    device = next(model.parameters()).device
    model.eval()
    tot, n_tokens = 0.0, 0
    with torch.no_grad():
        context = PruneContext(data=batches, batch_adapter=batch_adapter)
        for args, kwargs in context.iter_model_inputs():
            moved_args, moved_kwargs = move_batch_to_device(args, kwargs, device)
            ids = moved_kwargs.get("input_ids")
            if ids is None and moved_args:
                ids = moved_args[0]
            if not torch.is_tensor(ids) or ids.is_floating_point() or ids.dim() != 2:
                raise ValueError(
                    "calib_nll expects integer token-id batches shaped [batch, sequence] "
                    "as input_ids or the first positional model argument."
                )
            logits = _unwrap_logits(model(*moved_args, **moved_kwargs))
            nll_sum, n = nll_from_logits(logits, ids)
            tot += nll_sum
            n_tokens += n
    return tot / max(n_tokens, 1)
