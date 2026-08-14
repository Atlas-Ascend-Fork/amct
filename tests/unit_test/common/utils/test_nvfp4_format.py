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
    nvfp4_weight_dequant,
)
from amct_pytorch.quantization.dtypes.mxfp_impl import unpack_mxfloat4_to_fp32


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
    assert torch.allclose(out, unpack_mxfloat4_to_fp32(weight) * 2.0 * 0.5)


def test_dequant_reads_int8_packed_weight_as_unsigned_nibbles():
    # 0xFF stored as int8 is -1; both nibbles are E2M1 code 15 (-6.0).
    weight = torch.full((1, NVFP4_BLOCK_SIZE // 2), -1, dtype=torch.int8)
    weight_scale = torch.ones(1, 1).to(torch.float8_e4m3fn)

    out = nvfp4_weight_dequant(weight, weight_scale, torch.tensor(1.0))

    assert torch.equal(out, torch.full((1, NVFP4_BLOCK_SIZE), -6.0))


def test_dequant_accepts_uint8_view_of_e4m3_scale():
    rows, cols = 2, 32
    weight = _packed_weight(rows, cols)
    weight_scale = torch.full((rows, cols // NVFP4_BLOCK_SIZE), 2.0).to(
        torch.float8_e4m3fn
    )
    weight_scale_2 = torch.tensor(1.0)

    out = nvfp4_weight_dequant(weight, weight_scale.view(torch.uint8), weight_scale_2)

    assert torch.allclose(
        out, nvfp4_weight_dequant(weight, weight_scale, weight_scale_2)
    )


def test_dequant_accepts_int8_storage_of_e4m3_scale():
    rows, cols = 2, 32
    weight = _packed_weight(rows, cols)
    weight_scale = torch.full((rows, cols // NVFP4_BLOCK_SIZE), 2.0).to(
        torch.float8_e4m3fn
    )

    out = nvfp4_weight_dequant(weight, weight_scale.view(torch.int8), torch.tensor(1.0))

    assert torch.allclose(out, unpack_mxfloat4_to_fp32(weight) * 2.0)


def test_dequant_reshapes_exact_flattened_block_scale():
    rows, cols = 2, 32
    weight = _packed_weight(rows, cols)
    flat_scale = torch.full((rows * cols // NVFP4_BLOCK_SIZE,), 2.0).to(
        torch.float8_e4m3fn
    )

    out = nvfp4_weight_dequant(weight, flat_scale, torch.tensor(0.5))

    assert out.shape == (rows, cols)
    assert torch.allclose(out, unpack_mxfloat4_to_fp32(weight) * 2.0 * 0.5)


def test_dequant_reshapes_trtllm_padded_flattened_block_scale():
    # TRT-LLM flattens SF over pad(rows, 128) x pad(cols/16, 4).
    rows, cols = 2, 32
    weight = _packed_weight(rows, cols)
    flat_scale = torch.full((128 * 4,), 2.0).to(torch.float8_e4m3fn)

    out = nvfp4_weight_dequant(weight, flat_scale, torch.tensor(0.5))

    assert out.shape == (rows, cols)
    assert torch.allclose(out, unpack_mxfloat4_to_fp32(weight) * 2.0 * 0.5)


def test_dequant_trims_row_padded_block_scale():
    rows, cols = 2, 32
    weight = _packed_weight(rows, cols)
    padded_scale = torch.full((8, cols // NVFP4_BLOCK_SIZE), 2.0).to(
        torch.float8_e4m3fn
    )

    out = nvfp4_weight_dequant(weight, padded_scale, torch.tensor(1.0))

    assert out.shape == (rows, cols)
    assert torch.allclose(out, unpack_mxfloat4_to_fp32(weight) * 2.0)


def test_dequant_rejects_flattened_block_scale_of_unexpected_length():
    weight = _packed_weight(2, 32)
    weight_scale = torch.ones(7).to(torch.float8_e4m3fn)
    with pytest.raises(ValueError, match="length 7"):
        nvfp4_weight_dequant(weight, weight_scale, torch.tensor(1.0))


@pytest.mark.parametrize("scale_shape", [(2, 2, 2), (2, 1)])
def test_dequant_rejects_block_scale_that_does_not_cover_the_weight(scale_shape):
    weight = _packed_weight(2, 32)
    weight_scale = torch.ones(*scale_shape).to(torch.float8_e4m3fn)
    with pytest.raises(RuntimeError):
        nvfp4_weight_dequant(weight, weight_scale, torch.tensor(1.0))


def test_dequant_rejects_non_2d_weight():
    """Fused expert weights are not supported and must not be silently mangled."""
    weight = torch.zeros(2, 2, 16, dtype=torch.uint8)
    weight_scale = torch.ones(2, 2).to(torch.float8_e4m3fn)
    with pytest.raises(ValueError):
        nvfp4_weight_dequant(weight, weight_scale, torch.tensor(1.0))


def test_dequant_rejects_non_scalar_global_scale():
    weight = _packed_weight(2, 32)
    weight_scale = torch.ones(2, 2).to(torch.float8_e4m3fn)
    with pytest.raises(RuntimeError):
        nvfp4_weight_dequant(weight, weight_scale, torch.ones(2))


def test_dequant_round_trips_a_modelopt_style_quantization():
    """A weight quantized with the reference NVFP4 recipe comes back within FP4 error."""
    torch.manual_seed(0)
    weight = (torch.randn(8, 64) * 3).to(torch.bfloat16)
    positive_codes = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
    e2m1 = torch.tensor(positive_codes + [-value for value in positive_codes])
    rows, cols = weight.shape

    # Reference recipe: per-tensor scale over E4M3 x E2M1 range, then per-block scale.
    weight_scale_2 = weight.abs().max().float() / (6.0 * 448.0)
    blocks = weight.float().reshape(rows, cols // NVFP4_BLOCK_SIZE, NVFP4_BLOCK_SIZE)
    weight_scale = (blocks.abs().amax(dim=-1) / 6.0 / weight_scale_2).to(
        torch.float8_e4m3fn
    )
    effective_scale = weight_scale.float() * weight_scale_2
    scaled = (blocks / effective_scale.unsqueeze(-1)).unsqueeze(-1)
    codes = (scaled - e2m1).abs().argmin(dim=-1).to(torch.uint8).reshape(rows, cols)
    packed = codes[:, 0::2] | (codes[:, 1::2] << 4)

    out = nvfp4_weight_dequant(packed.contiguous(), weight_scale, weight_scale_2)

    # Worst case is half the gap between the two largest E2M1 codes of a block.
    tolerance = effective_scale.max().item()
    assert (out - weight.float()).abs().max().item() < tolerance
