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
import pytest
import torch

from amct_pytorch.common.utils.data_utils import (
    check_linear_input_dim,
    float_to_fp4e2m1,
)


@pytest.mark.parametrize("dim", [2, 3, 4, 5, 6])
def test_check_linear_input_dim_accepts_supported_ranks(dim):
    check_linear_input_dim(torch.zeros(*([2] * dim)))


@pytest.mark.parametrize("dim", [1, 7])
def test_check_linear_input_dim_rejects_out_of_range(dim):
    with pytest.raises(RuntimeError, match="dim from 2 to 6"):
        check_linear_input_dim(torch.zeros(*([2] * dim)))


@pytest.mark.parametrize(
    "value,expected",
    [
        (0.0, 0.0),
        (0.1, 0.0),
        (0.25, 0.0),     # boundary: 0.25 rounds to 0
        (0.4, 0.5),
        (0.5, 0.5),
        (0.74, 0.5),
        (0.75, 1.0),     # boundary
        (1.0, 1.0),
        (1.25, 1.0),
        (1.5, 1.5),
        (1.74, 1.5),
        (1.75, 2.0),
        (2.5, 2.0),
        (3.0, 3.0),
        (3.5, 4.0),
        (5.0, 4.0),
        (6.0, 6.0),     # > 5.0 saturates to 6
        (100.0, 6.0),
    ],
)
def test_float_to_fp4e2m1_quantization_levels(value, expected):
    out = float_to_fp4e2m1(torch.tensor([value]))
    assert out.item() == pytest.approx(expected)


def test_float_to_fp4e2m1_preserves_sign():
    x = torch.tensor([-1.0, -2.5, -100.0])
    out = float_to_fp4e2m1(x)
    assert (out < 0).all()
    assert out.tolist() == [-1.0, -2.0, -6.0]


def test_float_to_fp4e2m1_preserves_shape():
    x = torch.randn(3, 4)
    assert float_to_fp4e2m1(x).shape == x.shape
