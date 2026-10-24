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
"""Calibration sums must not overflow or round in the model's low precision."""

import math

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from amct_pytorch.pruning.calib import calib_nll, nll_from_logits


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
def test_long_sequence_nll_uses_safe_accumulation(dtype):
    ids = torch.zeros(1, 16384, dtype=torch.long)
    logits = torch.zeros(1, 16384, 128, dtype=dtype)
    losses = F.cross_entropy(
        logits[:, :-1].reshape(-1, 128), ids[:, 1:].reshape(-1), reduction="none"
    )
    expected = float(losses.double().sum())
    total, count = nll_from_logits(logits, ids)
    assert count == 16383
    assert math.isfinite(total)
    assert total == pytest.approx(expected, rel=2e-7)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_public_calibration_long_half_sequence_remains_finite(dtype):
    model = nn.Sequential(
        nn.Embedding(128, 4, dtype=dtype), nn.Linear(4, 128, bias=False, dtype=dtype)
    )
    with torch.no_grad():
        model[1].weight.zero_()
    ids = torch.zeros(1, 16384, dtype=torch.long)
    with torch.no_grad():
        logits = model(ids)
        losses = F.cross_entropy(
            logits[:, :-1].reshape(-1, 128), ids[:, 1:].reshape(-1), reduction="none"
        )
        expected = float(losses.double().mean())
    actual = calib_nll(model, [ids])
    assert math.isfinite(actual)
    assert actual == pytest.approx(expected, rel=2e-7)


def test_short_float64_control_retains_precision():
    logits = torch.tensor(
        [[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]], dtype=torch.float64
    )
    ids = torch.tensor([[0, 1, 2]], dtype=torch.long)
    expected = float(
        F.cross_entropy(
            logits[:, :-1].reshape(-1, 3), ids[:, 1:].reshape(-1), reduction="sum"
        )
    )
    total, count = nll_from_logits(logits, ids)
    assert count == 2 and total == pytest.approx(expected, rel=1e-14)
