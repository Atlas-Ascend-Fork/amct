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
"""E8M0 scales are piecewise constant; zero blocks must not poison gradients."""

import pytest
import torch
from amct_pytorch.quantization.dtypes.fp import QuantDequantFP

pytestmark = pytest.mark.cpu


@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("block_size", [1, 128])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_e8m0_zero_block_has_finite_gradient(bits, block_size, dtype):
    x = torch.zeros((2, 32), dtype=dtype, requires_grad=True)
    q = QuantDequantFP(bits=bits, block_size_col=block_size, scale_dtype="e8m0")
    y = q(x)
    assert torch.isfinite(y).all()
    torch.testing.assert_close(y, torch.zeros_like(x), rtol=0, atol=0)
    y.sum().backward()
    torch.testing.assert_close(x.grad, torch.ones_like(x), rtol=0, atol=0)


def test_e8m0_mixed_zero_and_nonzero_blocks_have_finite_gradients():
    x = torch.cat([torch.zeros((128, 32)), torch.ones((1, 32))]).requires_grad_()
    q = QuantDequantFP(bits=8, scale_dtype="e8m0")
    y = q(x)
    assert torch.isfinite(y).all()
    y.sum().backward()
    torch.testing.assert_close(x.grad, torch.ones_like(x), rtol=0, atol=0)


@pytest.mark.parametrize("bits", [4, 8])
def test_e8m0_positive_block_keeps_forward_and_ste_gradient(bits):
    x = torch.ones((2, 32), requires_grad=True)
    q = QuantDequantFP(bits=bits, scale_dtype="e8m0")
    y = q(x)
    torch.testing.assert_close(y, x.detach(), rtol=0, atol=0)
    y.sum().backward()
    torch.testing.assert_close(x.grad, torch.ones_like(x), rtol=0, atol=0)
