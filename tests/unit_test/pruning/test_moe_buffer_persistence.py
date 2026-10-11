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
"""Expert slicing preserves each buffer's existing checkpoint persistence."""

import pytest
import torch
from torch import nn

from amct_pytorch.pruning.domains.moe import (
    MoEPruningDomain,
    MoETarget,
    _slice_expert_dim,
    _slice_router_experts,
)

pytestmark = pytest.mark.cpu


@pytest.mark.parametrize("router", [False, True])
def test_slice_keeps_transient_buffer_out_of_state_dict(router):
    module = nn.Module()
    values = torch.arange(4, dtype=torch.float32)
    module.register_buffer("temporary_stats", values.clone(), persistent=False)
    module.register_buffer("saved_stats", values.clone())
    keep = torch.tensor([1, 3])
    slice_experts = _slice_router_experts if router else _slice_expert_dim
    slice_experts(module, keep, 4)
    torch.testing.assert_close(module.temporary_stats, values[keep], rtol=0, atol=0)
    torch.testing.assert_close(module.saved_stats, values[keep], rtol=0, atol=0)
    assert "temporary_stats" not in module.state_dict()
    assert "saved_stats" in module.state_dict()


def test_fused_domain_prune_preserves_transient_expert_and_router_buffers():
    model = nn.Module()
    model.experts = nn.Module()
    model.experts.num_experts = 4
    model.experts.weight = nn.Parameter(torch.ones(4, 3, 3))
    model.gate = nn.Linear(3, 4, bias=False)
    for module in (model.experts, model.gate):
        module.register_buffer("temporary_stats", torch.arange(4.0), persistent=False)
        module.register_buffer("saved_stats", torch.arange(4.0))
    target = MoETarget("", "gate", "experts", fused=True)
    MoEPruningDomain().prune_experts(model, target, [1, 3], top_k=2)
    assert model.experts.weight.shape == (2, 3, 3)
    assert model.gate.weight.shape == (2, 3)
    assert not any("temporary_stats" in name for name in model.state_dict())
    for module in (model.experts, model.gate):
        torch.testing.assert_close(
            module.temporary_stats, torch.tensor([1.0, 3.0]), rtol=0, atol=0
        )
