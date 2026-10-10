#!/usr/bin/env python3
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
from types import SimpleNamespace
import unittest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from amct_pytorch.common.optimization.factory import build_optimizer, build_lr_scheduler


class OptimizerConfigurationTest(unittest.TestCase):
    def test_adamw_flat_parameters_use_requested_learning_rate_and_decay(self):
        parameter = nn.Parameter(torch.tensor([1.0]))
        args = SimpleNamespace(optimizer='adamw', base_lr=0.02, weight_decay=0.25)
        optimizer = build_optimizer(args, [parameter])
        reference = nn.Parameter(parameter.detach().clone())
        expected = torch.optim.AdamW(
            [reference], lr=args.base_lr, weight_decay=args.weight_decay
        )
        for item in (parameter, reference):
            item.grad = torch.tensor([0.5])
        optimizer.step()
        expected.step()
        torch.testing.assert_close(parameter, reference, rtol=0, atol=0)

    def test_adamw_group_lr_preserved_but_zero_decay_respected(self):
        parameter = nn.Parameter(torch.tensor([1.0]))
        args = SimpleNamespace(optimizer='adamw', base_lr=1e-3, weight_decay=0.0)
        optimizer = build_optimizer(args, [{'params': [parameter], 'lr': 0.2}])
        parameter.grad = torch.zeros_like(parameter)
        optimizer.step()
        torch.testing.assert_close(parameter, torch.tensor([1.0]), rtol=0, atol=0)
        self.assertEqual(optimizer.param_groups[0]['lr'], 0.2)

    def test_adamw_missing_options_use_factory_defaults(self):
        optimizer = build_optimizer(SimpleNamespace(), [nn.Parameter(torch.ones(1))])
        self.assertEqual(optimizer.defaults['lr'], 1e-5)
        self.assertEqual(optimizer.defaults['weight_decay'], 0.0)

    def test_adamw_explicit_group_decay_keeps_precedence(self):
        optimizer = build_optimizer(
            SimpleNamespace(base_lr=1e-3, weight_decay=0.25),
            [{'params': [nn.Parameter(torch.ones(1))], 'lr': 0.2, 'weight_decay': 0.3}],
        )
        self.assertEqual(optimizer.param_groups[0]['lr'], 0.2)
        self.assertEqual(optimizer.param_groups[0]['weight_decay'], 0.3)

    def check_cosine(self, count, batch_size, epochs):
        parameter = nn.Parameter(torch.ones(1))
        optimizer = torch.optim.SGD([parameter], lr=0.1)
        loader = DataLoader(TensorDataset(torch.ones(count, 1)), batch_size=batch_size)
        args = SimpleNamespace(
            lr_scheduler='cosine',
            base_lr=0.1,
            nsamples=count,
            cali_bsz=batch_size,
            epochs=epochs,
        )
        scheduler = build_lr_scheduler(args, optimizer)
        rates = [optimizer.param_groups[0]['lr']]
        for _ in range(epochs):
            for _ in loader:
                parameter.grad = torch.ones_like(parameter)
                optimizer.step()
                scheduler.step()
                rates.append(optimizer.param_groups[0]['lr'])
        self.assertEqual(scheduler.T_max, len(loader) * epochs)
        self.assertAlmostEqual(rates[-1], scheduler.eta_min)
        self.assertTrue(all(a >= b for a, b in zip(rates, rates[1:])))

    def test_cosine_short_nonempty_dataset(self):
        self.check_cosine(3, 4, 2)

    def test_cosine_remainder_batch(self):
        self.check_cosine(5, 2, 2)

    def test_cosine_divisible_control(self):
        self.check_cosine(8, 4, 2)

    def test_other_optimizer_configuration_control(self):
        args = SimpleNamespace(optimizer='adam', base_lr=0.02, weight_decay=0.25)
        optimizer = build_optimizer(args, [nn.Parameter(torch.ones(1))])
        self.assertEqual(optimizer.defaults['lr'], args.base_lr)
        self.assertEqual(optimizer.defaults['weight_decay'], args.weight_decay)


if __name__ == '__main__':
    unittest.main()
