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
"""Preserve shared layer references and empty activation batches in MXFP4 QAT."""

import pytest
import torch
from torch import nn

from amct_pytorch.experimental.fakequant.mxfp4_qat import (
    MXFP4QATConfig,
    MXFP4QATLinear,
    convert_to_mxfp4_qat,
    mxfp4_fake_quant,
    mxfp4_saturation_mask,
)

pytestmark = pytest.mark.cpu


class SharedLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.first = nn.Linear(32, 4)
        self.second = self.first


def test_conversion_preserves_both_references_to_a_shared_linear():
    model = SharedLinear()
    weight, bias = model.first.weight, model.first.bias
    config = MXFP4QATConfig(backend="torch")
    assert convert_to_mxfp4_qat(model, config) is model
    assert isinstance(model.first, MXFP4QATLinear)
    assert isinstance(model.second, MXFP4QATLinear)
    assert model.first is model.second
    assert model.first.weight is weight
    assert model.first.bias is bias
    assert convert_to_mxfp4_qat(model, config).first is model.first


def test_conversion_preserves_shared_linear_across_different_containers():
    shared = nn.Linear(32, 4, bias=False)
    model = nn.ModuleDict(
        {"left": nn.Sequential(shared), "right": nn.Sequential(shared)}
    )
    convert_to_mxfp4_qat(model, MXFP4QATConfig(backend="torch"))
    assert model["left"][0] is model["right"][0]
    assert isinstance(model["left"][0], MXFP4QATLinear)


def test_conversion_keeps_a_shared_reference_float_when_any_path_is_skipped():
    model = SharedLinear()
    original = model.first
    convert_to_mxfp4_qat(model, MXFP4QATConfig(backend="torch"), skip_names=("second",))
    assert model.first is original
    assert model.second is original
    assert not isinstance(model.first, MXFP4QATLinear)
    assert not isinstance(model.second, MXFP4QATLinear)


class SharedContainer(nn.Module):
    def __init__(self):
        super().__init__()
        self.first = nn.Sequential(nn.Linear(32, 4))
        self.second = self.first


def test_conversion_skips_shared_container_descendants_through_every_alias():
    model = SharedContainer()
    original = model.first[0]
    convert_to_mxfp4_qat(model, MXFP4QATConfig(backend="torch"), skip_names=("first",))
    assert model.first[0] is original
    assert model.second[0] is original
    assert not isinstance(model.first[0], MXFP4QATLinear)


class UnhashableLinear(nn.Linear):
    def __eq__(self, other):
        return isinstance(other, UnhashableLinear)

    __hash__ = None  # type: ignore[assignment]


def test_conversion_handles_linear_subclasses_that_define_eq_without_hash():
    shared = UnhashableLinear(32, 4)
    weight, bias = shared.weight, shared.bias
    model = nn.ModuleDict(
        {"left": nn.Sequential(shared), "right": nn.Sequential(shared)}
    )
    assert convert_to_mxfp4_qat(model, MXFP4QATConfig(backend="torch")) is model
    assert isinstance(model["left"][0], MXFP4QATLinear)
    assert model["left"][0] is model["right"][0]
    assert model["left"][0].weight is weight
    assert model["left"][0].bias is bias


@pytest.mark.parametrize("shape", [(0, 32), (2, 0, 64), (0, 17), (2, 0), (0,)])
@pytest.mark.parametrize("clip_grad", [False, True])
def test_empty_fake_quant_preserves_tensor_and_backward_contract(shape, clip_grad):
    x = torch.empty(shape, requires_grad=True)
    out = mxfp4_fake_quant(x, clip_grad=clip_grad, backend="torch")
    assert out.shape == x.shape
    assert out.dtype == x.dtype
    assert out.device == x.device
    assert mxfp4_saturation_mask(x).shape == x.shape
    out.sum().backward()
    assert x.grad.shape == x.shape
    assert x.grad.numel() == 0


def test_w4a4_linear_accepts_the_same_empty_batch_as_linear():
    float_layer = nn.Linear(32, 4)
    layer = MXFP4QATLinear.from_linear(
        float_layer, MXFP4QATConfig(quantize_input=True, backend="torch")
    )
    x = torch.empty(0, 32, requires_grad=True)
    out = layer(x)
    assert out.shape == float_layer(x).shape == (0, 4)
    out.sum().backward()
    assert x.grad.shape == x.shape


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_nonempty_exactly_representable_input_keeps_its_values_and_ste(dtype):
    x = torch.full((2, 17), 1.5, dtype=dtype, requires_grad=True)
    out = mxfp4_fake_quant(x, backend="torch")
    torch.testing.assert_close(out, x, rtol=0, atol=0)
    out.sum().backward()
    torch.testing.assert_close(x.grad, torch.ones_like(x), rtol=0, atol=0)


def test_conversion_skips_registered_none_without_losing_shared_linear_aliases():
    model = SharedLinear()
    model.opt = nn.Linear(32, 4)
    model.opt = None
    weight, bias = model.first.weight, model.first.bias
    assert "opt" in model._modules
    config = MXFP4QATConfig(backend="torch")
    assert convert_to_mxfp4_qat(model, config) is model
    assert model.opt is None
    assert "opt" in model._modules
    assert isinstance(model.first, MXFP4QATLinear)
    assert model.first is model.second
    assert model.first.weight is weight
    assert model.first.bias is bias
    assert convert_to_mxfp4_qat(model, config) is model
    assert model.opt is None
