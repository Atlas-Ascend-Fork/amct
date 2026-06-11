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
"""校准集 NLL（负对数似然）—— 无标签的模型质量代理，越小越好。

用途：allocation 守卫（sensitivity 分配 vs uniform 候选比较）等需要在两个候选模型间
做无标签选择的场合。LM 自回归口径：用 token[:-1] 预测 token[1:]。
"""

from __future__ import annotations

from typing import Any, Sequence

import torch
import torch.nn.functional as F
from torch import nn


def _unwrap_logits(out: Any) -> torch.Tensor:
    """统一解包模型输出：Tensor / (logits, ...) / HF 风格 .logits。"""
    if torch.is_tensor(out):
        return out
    if isinstance(out, (tuple, list)) and out:
        return out[0]
    if hasattr(out, "logits"):
        return out.logits
    raise TypeError(f"cannot extract logits from model output of type {type(out)!r}")


def calib_nll(model: nn.Module, data: Sequence[Any]) -> float:
    """在校准 token 批上计算平均自回归 NLL（无标签，越小越好）。

    设备无关：批数据会被搬到模型参数所在设备（CPU/NPU/GPU 均可）。
    """
    batches = list(data)
    if not batches:
        raise ValueError("calib_nll needs at least one calibration batch.")
    device = next(model.parameters()).device
    model.eval()
    tot, n = 0.0, 0
    with torch.no_grad():
        for c in batches:
            # 自回归 NLL 需要整型 token id 作目标；浮点批必然错用，直接报错不静默
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
