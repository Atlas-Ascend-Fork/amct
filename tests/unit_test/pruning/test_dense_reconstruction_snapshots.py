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
"""Dense recovery must use activations observed before later buffer mutations."""

import copy

import pytest
import torch
from torch import nn

from amct_pytorch.pruning.api import prune
from amct_pytorch.pruning.context import PruneContext
from amct_pytorch.pruning.domains.dense import TwoLayerDenseTarget
from amct_pytorch.pruning.prune_op.dense_reconstruct import _collect_consumer_acts


class DenseBlock(nn.Module):
    def __init__(self, dtype=torch.float32, clear_hidden=False):
        super().__init__()
        self.producer = nn.Linear(1, 2, bias=False, dtype=dtype)
        self.consumer = nn.Linear(2, 1, bias=False, dtype=dtype)
        self.clear_hidden = clear_hidden
        with torch.no_grad():
            self.producer.weight.copy_(torch.tensor([[1.0], [2.0]], dtype=dtype))
            self.consumer.weight.copy_(torch.tensor([[1.0, 3.0]], dtype=dtype))

    def forward(self, inputs):
        hidden = self.producer(inputs)
        output = self.consumer(hidden)
        if self.clear_hidden:
            hidden.zero_()
        return output


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_collected_activations_survive_post_use_buffer_clear(dtype):
    model = DenseBlock(dtype=dtype, clear_hidden=True)
    batch = torch.tensor([[1.0], [3.0], [5.0]], dtype=dtype)
    expected = model.producer(batch).detach().float()
    target = TwoLayerDenseTarget("producer", "consumer")
    activations, paths, count = _collect_consumer_acts(
        model, [target], PruneContext(data=[batch]), torch.device("cpu")
    )
    assert count == 1 and paths[id(target)] == "consumer"
    torch.testing.assert_close(activations[id(target)][0], expected)
    assert not model.consumer._forward_hooks


def test_public_dense_recovery_is_independent_of_post_use_buffer_clear():
    reference = DenseBlock()
    cleared = copy.deepcopy(reference)
    cleared.clear_hidden = True
    batch = torch.tensor([[1.0], [3.0], [2.0], [4.0]])
    with torch.no_grad():
        torch.testing.assert_close(reference(batch), cleared(batch))
    config = {
        "methods": {"dense": {"name": "reconstruct", "kwargs": {"prune_ratio": 0.5}}},
        "min_neurons": 1,
    }
    prune(reference, config, data=[batch])
    prune(cleared, config, data=[batch])
    assert reference.consumer.in_features == cleared.consumer.in_features == 1
    for expected, actual in zip(reference.parameters(), cleared.parameters()):
        torch.testing.assert_close(actual, expected)
    with torch.no_grad():
        torch.testing.assert_close(cleared(batch), reference(batch))
