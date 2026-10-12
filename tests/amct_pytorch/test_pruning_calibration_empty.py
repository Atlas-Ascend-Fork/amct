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
"""Calibration NLL must not assign zero loss to batches with no targets."""

import math

import pytest
import torch

from amct_pytorch.pruning.calib import calib_nll


def uniform_model():
    model = torch.nn.Embedding(8, 8)
    with torch.no_grad():
        model.weight.zero_()
    return model


@pytest.mark.parametrize("shape", [(2, 1), (0, 3), (2, 0)])
def test_calibration_rejects_no_next_token_targets(shape):
    ids = torch.ones(shape, dtype=torch.int64)
    with pytest.raises(ValueError, match="next-token target"):
        calib_nll(uniform_model(), [ids])


def test_calibration_counts_real_targets_across_batches():
    batches = [torch.tensor([[1, 2, 3]]), torch.tensor([[2, 3]])]
    assert calib_nll(uniform_model(), batches) == pytest.approx(math.log(8))


def test_empty_target_batch_does_not_hide_valid_batch():
    batches = [torch.ones(2, 1, dtype=torch.int64), torch.tensor([[1, 2]])]
    assert calib_nll(uniform_model(), batches) == pytest.approx(math.log(8))
