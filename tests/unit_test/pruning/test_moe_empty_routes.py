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
"""Empty token batches must leave expert activation counts unchanged."""

import pytest
import torch
from torch import nn

from amct_pytorch.pruning import prune

from amct_pytorch.pruning.prune_op.moe_mass_pruning import ActivationCounter

pytestmark = pytest.mark.cpu


@pytest.mark.parametrize("tuple_output", [False, True])
@pytest.mark.parametrize("initialized", [False, True])
def test_empty_router_output_does_not_crash_or_add_counts(tuple_output, initialized):
    counter = ActivationCounter(num_experts=4)
    if initialized:
        counter.update(torch.tensor([[8.0, 4.0, 2.0, 1.0]]), top_k=2)
    previous = None if counter.counts is None else counter.counts.clone()
    output = (
        (torch.empty(0, 2, dtype=torch.long), torch.empty(0, 2))
        if tuple_output
        else torch.empty(0, 4)
    )
    counter.update(output, top_k=2)
    if previous is None:
        assert counter.counts is None
    else:
        torch.testing.assert_close(counter.counts, previous, rtol=0, atol=0)
    assert counter.num_experts == 4


class _MoE(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = nn.Linear(4, 4, bias=False)
        self.experts = nn.ModuleList(nn.Linear(4, 4) for _ in range(4))
        self.num_experts = 4
        self.top_k = 2

    def forward(self, x):
        self.gate(x)
        return self.experts[0](x)


def test_public_mass_prune_skips_an_empty_token_batch():
    model = _MoE()
    initial_keys = set(model.state_dict())
    config = {
        "methods": {
            "moe": {
                "name": "activation_count",
                "kwargs": {"prune_ratio": 0.5, "top_k": 2},
            }
        }
    }
    prune(model, config, data=[torch.empty(0, 4)])
    assert set(model.state_dict()) == initial_keys
    assert len(model.experts) == 4
