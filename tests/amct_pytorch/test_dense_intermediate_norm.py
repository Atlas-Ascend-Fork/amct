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
"""Do not narrow a dense pair across a normalization that is not resized."""

import unittest

import torch
from torch import nn

from amct_pytorch.pruning import prune


def inputs():
    return torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [2.0, -1.0, 1.0],
        ]
    )


def chain(middle):
    model = nn.Sequential(nn.Linear(3, 4), middle, nn.Linear(4, 2)).eval()
    with torch.no_grad():
        model[0].weight.copy_(
            torch.tensor(
                [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0], [4.0, 1.0, 0.0]]
            )
        )
        model[0].bias.zero_()
        model[2].weight.copy_(torch.arange(1.0, 9.0).reshape(2, 4) / 10.0)
        model[2].bias.zero_()
    return model


def config(method):
    kwargs = {"prune_ratio": 0.5}
    if method == "reconstruct":
        kwargs.update(recovery="ls", ridge=0.0)
    return {"methods": {"dense": {"name": method, "kwargs": kwargs}}, "min_neurons": 1}


class TestDenseIntermediateNormalization(unittest.TestCase):
    def assert_normalized_chain_unchanged(self, factory):
        for method in ("low_variance", "reconstruct"):
            with self.subTest(method=method):
                model = chain(factory())
                data = inputs()
                with torch.no_grad():
                    expected = model(data)
                before = {
                    name: value.clone() for name, value in model.state_dict().items()
                }
                prune(model, config(method), data=[data])
                with torch.no_grad():
                    torch.testing.assert_close(model(data), expected)
                self.assertEqual(model[0].out_features, 4)
                self.assertEqual(model[2].in_features, 4)
                for name, value in model.state_dict().items():
                    torch.testing.assert_close(value, before[name])

    def test_layer_norm_is_not_treated_as_width_independent(self):
        self.assert_normalized_chain_unchanged(lambda: nn.LayerNorm(4))

    def test_layer_norm_without_affine_still_constrains_width(self):
        self.assert_normalized_chain_unchanged(
            lambda: nn.LayerNorm(4, elementwise_affine=False)
        )

    def test_batch_norm_statistics_are_not_left_at_the_old_width(self):
        self.assert_normalized_chain_unchanged(lambda: nn.BatchNorm1d(4))

    def test_group_norm_parameters_are_not_left_at_the_old_width(self):
        self.assert_normalized_chain_unchanged(lambda: nn.GroupNorm(2, 4))

    def test_activations_and_identity_still_allow_pruning(self):
        for method in ("low_variance", "reconstruct"):
            for factory in (nn.ReLU, nn.GELU, nn.Identity):
                with self.subTest(method=method, middle=factory.__name__):
                    model = chain(factory())
                    data = inputs()
                    prune(model, config(method), data=[data])
                    self.assertEqual(model[0].out_features, 2)
                    self.assertEqual(model[2].in_features, 2)
                    with torch.no_grad():
                        output = model(data)
                    self.assertEqual(tuple(output.shape), (5, 2))
                    self.assertTrue(torch.all(torch.isfinite(output)))


if __name__ == "__main__":
    unittest.main()
