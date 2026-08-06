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

from amct_pytorch.common.utils.nvfp4_format import (
    NVFP4_BLOCK_SIZE,
    is_nvfp4_scale_key,
    nvfp4_weight_dequant,
    unpack_nvfloat4_to_fp32,
)

# ---- unpack_nvfloat4_to_fp32 ---------------------------------------------


def test_unpack_splits_each_byte_into_two_e2m1_codes():
    # 0x21 -> low nibble 1 (0.5), high nibble 2 (1.0); 0x43 -> 3 (1.5), 4 (2.0)
    packed = torch.tensor([[0x21, 0x43]], dtype=torch.uint8)
    out = unpack_nvfloat4_to_fp32(packed)
    assert out.shape == (1, 4)
    assert torch.equal(out, torch.tensor([[0.5, 1.0, 1.5, 2.0]]))


def test_unpack_reads_int8_bytes_as_unsigned():
    # 0xFF stored as int8 is -1; both nibbles are code 15 (-6.0)
    packed = torch.tensor([[-1]], dtype=torch.int8)
    out = unpack_nvfloat4_to_fp32(packed)
    assert torch.equal(out, torch.tensor([[-6.0, -6.0]]))


# ---- nvfp4_weight_dequant -------------------------------------------------


def _packed_weight(rows, cols):
    return torch.randint(0, 256, (rows, cols // 2), dtype=torch.uint8)


def test_dequant_applies_block_and_global_scales():
    rows, cols = 2, 32
    weight = _packed_weight(rows, cols)
    weight_scale = torch.full((rows, cols // NVFP4_BLOCK_SIZE), 2.0).to(
        torch.float8_e4m3fn
    )
    weight_scale_2 = torch.tensor(0.5)

    out = nvfp4_weight_dequant(weight, weight_scale, weight_scale_2)

    assert out.shape == (rows, cols)
    assert torch.allclose(out, unpack_nvfloat4_to_fp32(weight) * 2.0 * 0.5)


def test_dequant_accepts_uint8_view_of_e4m3_scale():
    rows, cols = 2, 32
    weight = _packed_weight(rows, cols)
    weight_scale = torch.full((rows, cols // NVFP4_BLOCK_SIZE), 2.0).to(
        torch.float8_e4m3fn
    )
    weight_scale_2 = torch.tensor(1.0)

    out = nvfp4_weight_dequant(weight, weight_scale.view(torch.uint8), weight_scale_2)

    assert torch.allclose(out, nvfp4_weight_dequant(weight, weight_scale, weight_scale_2))


def test_dequant_reshapes_flattened_block_scale():
    # A flattened block scale is laid out over a padded 128-row x 4-column tiling.
    rows, cols = 2, 32
    weight = _packed_weight(rows, cols)
    flat_scale = torch.full((128 * 4,), 2.0).to(torch.float8_e4m3fn)
    weight_scale_2 = torch.tensor(0.5)

    out = nvfp4_weight_dequant(weight, flat_scale, weight_scale_2)

    assert out.shape == (rows, cols)
    assert torch.allclose(out, unpack_nvfloat4_to_fp32(weight) * 2.0 * 0.5)


def test_dequant_rejects_3d_block_scale():
    weight = _packed_weight(2, 32)
    weight_scale = torch.ones(2, 2, 2).to(torch.float8_e4m3fn)
    with pytest.raises(ValueError, match="1-D or 2-D"):
        nvfp4_weight_dequant(weight, weight_scale, torch.tensor(1.0))


# ---- is_nvfp4_scale_key ---------------------------------------------------


@pytest.mark.parametrize(
    "name,expected",
    [
        ("layer.weight_scale", True),
        ("layer.weight_scale_2", True),
        ("layer.weight", False),
        ("layer.weight_scale_inv", False),
    ],
)
def test_is_nvfp4_scale_key(name, expected):
    assert is_nvfp4_scale_key(name) is expected
