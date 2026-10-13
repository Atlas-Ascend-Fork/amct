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

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from amct_pytorch.common.optimization.calibration_solver import CalibrationSolver


class _CalibrationModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, inputs, multiplier=1):
        self.calls.append((inputs, multiplier))
        return inputs * multiplier


def _solver(module):
    args = SimpleNamespace(device="cpu", quant_target=["mlp"])
    return CalibrationSolver(args, 0, module)


def test_calibration_solver_runs_each_batch_without_gradient_state():
    module = _CalibrationModule()
    data = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    loader = DataLoader(TensorDataset(data), batch_size=2, shuffle=False)

    with torch.no_grad():
        _solver(module).solve(loader, forward_kwargs={"multiplier": 3})

    assert [item[0].tolist() for item in module.calls] == [
        [[0.0, 1.0], [2.0, 3.0]],
        [[4.0, 5.0], [6.0, 7.0]],
    ]
    assert all(item[1] == 3 for item in module.calls)


def test_calibration_solver_exports_ptq_params_on_finalize():
    module = _CalibrationModule()
    module.export_ptq_params = lambda: {"value": torch.tensor(2.0)}
    assert _solver(module).finalize() == {"value": torch.tensor(2.0)}


def test_calibration_solver_rejects_multi_input_batches():
    module = _CalibrationModule()
    with pytest.raises(ValueError, match="one input tensor"):
        _solver(module).solve([(torch.zeros(1), torch.zeros(1))])
