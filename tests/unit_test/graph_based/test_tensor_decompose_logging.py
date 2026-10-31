#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------
"""auto_decomposition 逐层 Processing 日志的单测。

日志已降级为 DEBUG 并保留逐层 Processing 输出，此测试锁定该行为，
防止后续误改回 INFO 造成高频日志回归。原生分解库被替换为 UNCHANGE，
测试不依赖 CANN 原生环境。
"""

import logging

import torch.nn as nn

from amct_pytorch.classic.graph_based.amct_pytorch.tensor_decompose import (
    tensor_decompose as td,
)
from amct_pytorch.classic.graph_based.amct_pytorch.common.decomposition.decomposition import (
    DecomposeMode,
)


class _TinyConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(4, 8, kernel_size=3, bias=False)
        self.linear = nn.Linear(8, 4)

    def forward(self, x):
        return self.linear(self.conv(x).mean(dim=(2, 3)))


def test_auto_decomposition_logs_processing_at_debug_level(monkeypatch, caplog):
    def _unchanged(tensor, stride=1, group=1, dilation=1):
        return {"mode": DecomposeMode.UNCHANGE, "first": None, "last": None}

    monkeypatch.setattr(td, "tensor_decomposition", _unchanged)

    with caplog.at_level(logging.DEBUG, logger="Log"):
        _, changes = td.auto_decomposition(_TinyConvNet())
    # Conv2d 层被遍历到时，Processing 日志必须存在且为 DEBUG 级
    processing = [
        rec
        for rec in caplog.records
        if "Processing" in rec.getMessage() and "conv" in rec.getMessage()
    ]
    assert processing, "Processing log for conv layer is missing"
    assert all(rec.levelno == logging.DEBUG for rec in processing), (
        "Processing log must stay at DEBUG level (downgraded from INFO)"
    )
    # linear 层不是 Conv2d，不应出现对应 Processing 日志
    assert not any("linear" in rec.getMessage() for rec in processing)
    assert changes == {}
