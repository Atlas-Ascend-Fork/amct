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
"""CNN recovery caches must retain tensors as observed by the consumer hook."""

import copy
import unittest

import torch
from torch import nn

from amct_pytorch.pruning import prune
from amct_pytorch.pruning.domains.cnn import CNNChannelTarget
from amct_pytorch.pruning.prune_op.cnn_reconstruct import _register_consumer_hooks


def make_model(inplace):
    model = nn.Sequential(
        nn.Conv2d(1, 2, 1, bias=False),
        nn.Conv2d(2, 1, 1, bias=False),
        nn.ReLU(inplace=inplace),
    ).eval()
    with torch.no_grad():
        model[0].weight.copy_(torch.tensor([1.0, 0.2]).reshape(2, 1, 1, 1))
        model[1].weight.copy_(torch.tensor([-2.0, -1.0]).reshape(1, 2, 1, 1))
    return model


class TestCnnRecoverySnapshots(unittest.TestCase):
    def test_hook_retains_pre_activation_output(self):
        for inplace in (False, True):
            with self.subTest(inplace=inplace):
                model = make_model(inplace)
                data = torch.tensor([1.0, -1.0, 2.0, -2.0]).reshape(1, 1, 1, 4)
                with torch.no_grad():
                    expected = model[1](model[0](data))
                target = CNNChannelTarget("0", None, "1")
                caches, hooks = _register_consumer_hooks(model, [target])
                try:
                    with torch.no_grad():
                        output = model(data)
                    torch.testing.assert_close(caches[id(target)]["y"][0], expected)
                    self.assertTrue(torch.all(output >= 0))
                    self.assertTrue(torch.any(expected < 0))
                finally:
                    for hook in hooks:
                        hook.remove()

    def test_hook_retains_input_when_a_batch_buffer_is_reused(self):
        model = nn.Sequential(nn.Conv2d(1, 1, 1, bias=False)).eval()
        data = torch.tensor([1.0, 2.0]).reshape(1, 1, 1, 2)
        expected = data.clone()
        target = CNNChannelTarget("unused", None, "0")
        caches, hooks = _register_consumer_hooks(model, [target])
        try:
            with torch.no_grad():
                model(data)
                data.fill_(99.0)
            torch.testing.assert_close(caches[id(target)]["x"][0], expected)
        finally:
            for hook in hooks:
                hook.remove()

    def test_public_prune_is_equivalent_for_inplace_and_out_of_place_relu(self):
        inplace_model = make_model(True)
        reference_model = make_model(False)
        original = copy.deepcopy(inplace_model)
        data = torch.tensor([1.0, -1.0, 2.0, -2.0]).reshape(1, 1, 1, 4)
        config = {
            "methods": {
                "cnn": {
                    "name": "reconstruct",
                    "kwargs": {"prune_ratio": 0.5, "ridge": 0.0, "recovery": "ls"},
                }
            },
            "min_channels": 1,
        }
        prune(reference_model, config, data=[data])
        prune(inplace_model, config, data=[data])
        self.assertEqual(inplace_model[0].out_channels, 1)
        self.assertEqual(inplace_model[1].in_channels, 1)
        torch.testing.assert_close(inplace_model[1].weight, reference_model[1].weight)
        with torch.no_grad():
            torch.testing.assert_close(inplace_model(data), original(data))


if __name__ == "__main__":
    unittest.main()
