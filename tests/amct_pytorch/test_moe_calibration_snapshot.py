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
"""Collect MoE inputs as snapshots before later tensor mutations."""

import pytest
import torch
from torch import nn

from amct_pytorch.pruning.context import PruneContext
from amct_pytorch.pruning.prune_op.moe_output_merge import _collect_block_inputs

pytestmark = pytest.mark.cpu


def test_block_input_snapshot_precedes_inplace_activation():
    block = nn.Sequential(nn.ReLU(inplace=True), nn.Linear(4, 4, bias=False))
    model = nn.Sequential(block)
    inputs = torch.tensor([[-3.0, 1.0, -2.0, 4.0]])
    expected = inputs.clone()
    captured = _collect_block_inputs(model, [block], PruneContext(data=[inputs]))
    torch.testing.assert_close(captured[id(block)], expected, rtol=0, atol=0)
    assert not torch.equal(inputs, expected)


def test_reused_calibration_buffer_keeps_each_batch_values():
    block = nn.Linear(4, 4, bias=False)
    model = nn.Sequential(block)
    buffer = torch.empty(2, 4)

    def batches():
        for value in (1.0, 3.0, 5.0):
            buffer.fill_(value)
            yield buffer

    captured = _collect_block_inputs(model, [block], PruneContext(data=batches()))
    expected = torch.cat([torch.full((2, 4), value) for value in (1.0, 3.0, 5.0)])
    torch.testing.assert_close(captured[id(block)], expected, rtol=0, atol=0)


def test_ordinary_independent_batches_keep_order_and_values():
    block = nn.Linear(4, 4, bias=False)
    inputs = [torch.full((2, 4), value) for value in (1.0, 3.0)]
    captured = _collect_block_inputs(block, [block], PruneContext(data=inputs))
    torch.testing.assert_close(captured[id(block)], torch.cat(inputs), rtol=0, atol=0)
