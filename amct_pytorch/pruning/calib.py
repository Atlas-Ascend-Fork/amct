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
"""Calibration-set NLL: a label-free model-quality proxy (lower is better).

Used where two candidate models must be compared without labels, e.g. the allocation
guard (sensitivity vs uniform). LM autoregressive convention: token[:-1] predicts token[1:].
"""

from __future__ import annotations

from typing import Any, Sequence

import torch
import torch.nn.functional as F
from torch import nn


def _unwrap_logits(out: Any) -> torch.Tensor:
    if torch.is_tensor(out):
        return out
    if isinstance(out, (tuple, list)) and out:
        return out[0]
    if hasattr(out, "logits"):
        return out.logits
    raise TypeError(f"cannot extract logits from model output of type {type(out)!r}")


def calib_nll(model: nn.Module, data: Sequence[Any]) -> float:
    """Mean autoregressive NLL over calibration token batches (label-free, lower is better).

    Device-agnostic: batches are moved to the model's parameter device.
    """
    batches = list(data)
    if not batches:
        raise ValueError("calib_nll needs at least one calibration batch.")
    device = next(model.parameters()).device
    model.eval()
    tot, n = 0.0, 0
    with torch.no_grad():
        for c in batches:
            # Autoregressive NLL needs integer token-id targets; a float batch is a misuse -> raise, not silent skip.
            if c.is_floating_point():
                raise ValueError(
                    "calib_nll expects integer token-id batches (target of "
                    f"cross_entropy); got floating-point batch dtype={c.dtype}."
                )
            c = c.to(device)
            logits = _unwrap_logits(model(c))
            nll = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.shape[-1]), c[:, 1:].reshape(-1)
            )
            tot, n = tot + float(nll), n + 1
    return tot / max(n, 1)
