#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) gcw_Z57idahf 2026. All rights reserved.
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
from amct_pytorch.common.optimization.blockwise_solver import BlockwiseSolver


class TrainableLinear(nn.Linear):
    def trainable_params(self):
        return [self.weight, self.bias]


class BlockwiseZeroLossTest(unittest.TestCase):
    def run_calibration(self, optimizer, tuple_output=False, later_error=False):
        class Layer(TrainableLinear):
            def forward(self, x):
                value = super().forward(x)
                return (value, None) if tuple_output else value

        layer = Layer(2, 2)
        with torch.no_grad():
            layer.weight.copy_(torch.eye(2))
            layer.bias.zero_()
        before = {k: v.clone() for k, v in layer.state_dict().items()}
        inputs = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        target = inputs.clone()
        if later_error:
            target[1] += 1.0
        loader = DataLoader(TensorDataset(inputs, target), batch_size=1)
        args = SimpleNamespace(
            quant_target=['mlp'],
            device=torch.device('cpu'),
            epochs=1,
            base_lr=1e-3,
            optimizer=optimizer,
            weight_decay=0.0,
            momentum=0.0,
            lr_scheduler='none',
        )
        solver = BlockwiseSolver(args, layer_idx=0, model=layer)
        solver.solve(loader)
        self.assertEqual(solver.current_iter, len(loader))
        self.assertTrue(all(torch.isfinite(p).all() for p in layer.parameters()))
        for value in solver.optimizer.state.values():
            for tensor in value.values():
                if isinstance(tensor, torch.Tensor):
                    self.assertTrue(torch.isfinite(tensor).all())
        if not later_error:
            for name, value in layer.state_dict().items():
                torch.testing.assert_close(value, before[name], rtol=0, atol=0)
        else:
            self.assertFalse(torch.equal(layer.weight, before['weight']))

    def test_exact_reconstruction_adam(self):
        self.run_calibration('adam')

    def test_exact_reconstruction_sgd(self):
        self.run_calibration('sgd')

    def test_exact_reconstruction_signsgd(self):
        self.run_calibration('signsgd')

    def test_tuple_output_zero_then_nonzero_can_still_learn(self):
        self.run_calibration('adam', tuple_output=True, later_error=True)

    def test_nonzero_gradient_normalization_unchanged(self):
        layer = TrainableLinear(1, 1, bias=False)
        with torch.no_grad():
            layer.weight.fill_(2.0)
        args = SimpleNamespace(
            quant_target=['mlp'],
            device=torch.device('cpu'),
            epochs=1,
            base_lr=1e-3,
            optimizer='sgd',
            weight_decay=0.0,
            momentum=0.0,
            lr_scheduler='none',
        )
        solver = BlockwiseSolver(args, 0, layer)
        solver.solve(
            DataLoader(TensorDataset(torch.ones(1, 1), torch.zeros(1, 1)), batch_size=1)
        )
        torch.testing.assert_close(
            layer.weight, torch.tensor([[1.99]]), rtol=1e-6, atol=0
        )


if __name__ == '__main__':
    unittest.main()
