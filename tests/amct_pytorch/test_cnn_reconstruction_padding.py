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
"""Recovery expressions must match the real CPU Conv2d border contract."""

import pytest
import torch
from torch import nn

from amct_pytorch.pruning.prune_op.cnn_reconstruct import (
    _conv_bias_fold_delta,
    _reconstruct_conv_consumer,
)


@pytest.mark.parametrize("mode", ["zeros", "reflect", "replicate", "circular"])
def test_reconstruction_recovers_consumer_with_its_padding_mode(mode):
    generator = torch.Generator().manual_seed(7)
    inputs = torch.randn(3, 1, 5, 6, generator=generator)
    consumer = nn.Conv2d(1, 1, 3, padding=1, padding_mode=mode, bias=True)
    with torch.no_grad():
        consumer.weight.copy_(torch.arange(1, 10).reshape(1, 1, 3, 3) / 10.0)
        consumer.bias.fill_(0.25)
    targets = consumer(inputs).detach()
    weights = _reconstruct_conv_consumer(consumer, inputs, targets, ridge=0.0)
    torch.testing.assert_close(weights, consumer.weight, rtol=1e-4, atol=1e-4)


def test_reconstruction_supports_same_padding_with_even_kernel():
    inputs = torch.randn(3, 1, 5, 6, generator=torch.Generator().manual_seed(9))
    consumer = nn.Conv2d(1, 1, 2, padding="same", bias=False)
    with torch.no_grad():
        consumer.weight.copy_(torch.tensor([[[[0.2, 0.4], [0.6, 0.8]]]]))
    weights = _reconstruct_conv_consumer(
        consumer, inputs, consumer(inputs).detach(), ridge=0.0
    )
    torch.testing.assert_close(weights, consumer.weight, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("mode", ["zeros", "reflect", "replicate", "circular"])
def test_bias_fold_matches_removed_channel_output(mode):
    inputs = torch.arange(1.0, 61.0).reshape(1, 2, 5, 6)
    consumer = nn.Conv2d(2, 1, 3, padding=1, padding_mode=mode, bias=False)
    with torch.no_grad():
        consumer.weight.fill_(1.0)
    removed = inputs.clone()
    removed[:, 0] = 0.0
    expected = consumer(removed).mean(dim=(0, 2, 3)).detach()
    actual = _conv_bias_fold_delta(consumer, inputs, [0], total=2)
    torch.testing.assert_close(actual, expected)
