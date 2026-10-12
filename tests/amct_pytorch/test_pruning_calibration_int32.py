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
"""CPU token-id dtype regression for pruning calibration."""

import math

import pytest
import torch

from amct_pytorch.pruning.calib import calib_nll, nll_from_logits


def uniform_model():
    model = torch.nn.Embedding(8, 8)
    with torch.no_grad():
        model.weight.zero_()
    return model


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("entry", ["logits", "model"])
def test_nll_accepts_integer_token_ids(dtype, entry):
    ids = torch.tensor([[1, 2, 3]], dtype=dtype)
    if entry == "logits":
        total, count = nll_from_logits(torch.zeros(1, 3, 8), ids)
        assert count == 2
        assert total == pytest.approx(2 * math.log(8))
    else:
        assert calib_nll(uniform_model(), [ids]) == pytest.approx(math.log(8))
    assert ids.dtype == dtype
