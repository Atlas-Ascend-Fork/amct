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
"""Temporary calibration/fine-tuning restores each module's prior mode."""

import pytest
import torch
import torch.nn as nn

from amct_pytorch.pruning.context import PruneContext
from amct_pytorch.pruning.finetune import prune_finetune
from amct_pytorch.pruning.utils import run_calibration_epoch


@pytest.mark.parametrize("operation", ["calibration", "finetune"])
@pytest.mark.parametrize("fail", [False, True])
def test_mixed_module_modes_are_restored(operation, fail):
    model = nn.Sequential(nn.Linear(2, 2), nn.Dropout(), nn.Linear(2, 2))
    model.train()
    model[1].eval()
    before = [module.training for module in model.modules()]

    def check_forward(module, args):
        if fail:
            raise RuntimeError("forward aborted")

    handle = model.register_forward_pre_hook(check_forward)
    x = torch.ones(2, 2)
    try:
        if operation == "calibration":

            def call():
                return run_calibration_epoch(model, PruneContext(data=[x]))
        else:

            def call():
                return prune_finetune(
                    model, [(x, torch.zeros(2, dtype=torch.long))], steps=1
                )

        if fail:
            with pytest.raises(RuntimeError, match="forward aborted"):
                call()
        else:
            call()
        assert [module.training for module in model.modules()] == before
    finally:
        handle.remove()


def test_homogeneous_eval_calibration_remains_eval():
    model = nn.Linear(2, 2).eval()
    assert run_calibration_epoch(model, PruneContext(data=[torch.ones(1, 2)])) == 1
    assert not model.training
