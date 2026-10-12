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
"""Zero calibration activations must not defeat positive ridge regularization."""

import pytest
import torch
from torch import nn

from amct_pytorch.pruning.config import MethodSpec, PruneConfig
from amct_pytorch.pruning.context import PruneContext
from amct_pytorch.pruning.domains.dense import DensePruningDomain
from amct_pytorch.pruning.prune_op.dense_reconstruct import (
    ReconstructDensePruningMethod,
)
from amct_pytorch.pruning.report import PruneReport
from amct_pytorch.pruning.utils import ridge_solve


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_positive_ridge_solves_zero_activation_system(dtype):
    gram = torch.zeros(3, 3, dtype=dtype)
    rhs = torch.zeros(3, 2, dtype=dtype)
    result = ridge_solve(gram, rhs, 0.01)
    torch.testing.assert_close(result, rhs, rtol=0, atol=0)
    assert torch.isfinite(result).all()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_nonzero_gram_keeps_relative_ridge(dtype):
    gram = torch.tensor([[4.0, 1.0], [1.0, 2.0]], dtype=dtype)
    rhs = torch.tensor([[1.0], [2.0]], dtype=dtype)
    expected = torch.linalg.solve(gram + torch.eye(2, dtype=dtype) * 0.03, rhs)
    torch.testing.assert_close(ridge_solve(gram, rhs, 0.01), expected)


def test_dense_reconstruct_accepts_dead_relu_calibration():
    model = nn.Sequential(nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 2)).eval()
    with torch.no_grad():
        model[0].weight.zero_()
        model[0].bias.fill_(-1)
    data = [torch.ones(8, 3)]
    expected = model(data[0]).detach()
    config = PruneConfig(min_neurons=1)
    domain = DensePruningDomain()
    targets = domain.find_targets(model, config)
    assert len(targets) == 1
    report = PruneReport()
    spec = MethodSpec(
        "reconstruct", {"prune_ratio": 0.5, "ridge": 0.01, "recovery": "ls"}
    )
    ReconstructDensePruningMethod().apply(
        model, domain, targets, PruneContext(data=data), report, config, spec
    )
    assert model[0].out_features == model[2].in_features == 2
    torch.testing.assert_close(model(data[0]), expected, rtol=0, atol=0)
    assert torch.isfinite(model[2].weight).all()
