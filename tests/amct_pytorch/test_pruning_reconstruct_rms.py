#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
"""Real public pruning regressions for finite calibration amplitudes."""

import pytest
import torch
from torch import nn

from amct_pytorch.pruning import prune


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.bfloat16])
@pytest.mark.parametrize("dead_channel", [0, 1])
@pytest.mark.parametrize(
    "amplitude", [1e20, -1e20, 1e-30, -1e-30, 1.0, -1.0, 1e-3, -1e-3, 0.0]
)
def test_reconstruct_ranks_finite_activation_rms(dtype, dead_channel, amplitude):
    model = nn.Sequential(nn.Linear(1, 2, bias=False), nn.Linear(2, 1, bias=False)).to(
        dtype
    )
    with torch.no_grad():
        model[0].weight.fill_(1)
        model[1].weight.fill_(1)
        model[1].weight[0, dead_channel] = 0
    samples = torch.full((8, 1), amplitude, dtype=dtype)
    expected = model(samples).detach().clone()
    assert torch.isfinite(samples).all()
    assert torch.isfinite(expected).all()

    prune(
        model,
        {
            "min_neurons": 1,
            "methods": {
                "dense": {
                    "name": "reconstruct",
                    "kwargs": {"prune_ratio": 0.5, "recovery": "none"},
                }
            },
        },
        data=[samples[:4], samples[4:]],
    )

    assert model[0].out_features == 1
    assert model[1].in_features == 1
    actual = model(samples).detach()
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    if amplitude != 0:
        torch.testing.assert_close(
            model[1].weight, torch.ones((1, 1), dtype=dtype), rtol=0, atol=0
        )
