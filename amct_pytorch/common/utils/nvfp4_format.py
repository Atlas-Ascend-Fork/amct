# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------
"""NVFP4 dequantization on top of the shared MXFP4 ``weight_dequant``.

NVFP4 packs two 4-bit E2M1 codes per byte exactly like MXFP4. The NVFP4 specifics
live in the scale: an E4M3 block scale covering 16 elements of a row, times one
per-tensor FP32 scale. Folding those two into the single FP32 block scale
``weight_dequant`` expects is all this module does.
"""

import torch

from amct_pytorch.quantization.dtypes.mxfp_impl import weight_dequant

NVFP4_BLOCK_SIZE = 16
NVFP4_PACKED_DTYPES = (torch.uint8, torch.int8)
NVFP4_SCALE_DTYPES = (torch.uint8, torch.int8, torch.float8_e4m3fn)
_SF_ROW_ALIGN = 128
_SF_COL_ALIGN = 4


def _nvfp4_scale_shape(weight: torch.Tensor, block_size: int):
    """Return (rows, scale_cols, padded_rows, padded_cols) for a packed NVFP4 weight."""
    rows = weight.shape[0]
    unpacked_cols = weight.shape[1] * 2
    scale_cols = unpacked_cols // block_size
    padded_rows = -(-rows // _SF_ROW_ALIGN) * _SF_ROW_ALIGN
    padded_cols = -(-scale_cols // _SF_COL_ALIGN) * _SF_COL_ALIGN
    return rows, scale_cols, padded_rows, padded_cols


def is_nvfp4_scale(
    weight: torch.Tensor,
    scale: torch.Tensor,
    block_size: int = NVFP4_BLOCK_SIZE,
) -> bool:
    """Whether `scale` is an NVFP4 E4M3 block scale for this packed weight.

    NVFP4: weight dtype=uint8/int8, shape=[M, N//2];
           scale dtype=e4m3/uint8, shape=[M, N//16] or a 1-D buffer of that
           length / the padded tile pad(M, 128) * pad(N//16, 4).
    """
    if weight.dtype not in NVFP4_PACKED_DTYPES or weight.dim() != 2:
        return False
    if scale.dtype not in NVFP4_SCALE_DTYPES:
        return False
    unpacked_cols = weight.shape[1] * 2
    if unpacked_cols % block_size != 0:
        return False
    rows, scale_cols, padded_rows, padded_cols = _nvfp4_scale_shape(weight, block_size)
    if scale.dim() == 2:
        return tuple(scale.shape) == (rows, scale_cols) or (
            scale.shape[0] >= rows and scale.shape[1] == scale_cols
        )
    if scale.dim() == 1:
        return scale.numel() in (rows * scale_cols, padded_rows * padded_cols)
    return False


def nvfp4_weight_dequant(
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_scale_2: torch.Tensor,
    block_size: int = NVFP4_BLOCK_SIZE,
) -> torch.Tensor:
    """Dequantize a packed NVFP4 weight of shape (M, N // 2) into a float tensor (M, N)."""
    if weight_scale.dtype in NVFP4_PACKED_DTYPES:
        weight_scale = weight_scale.view(torch.float8_e4m3fn)
    scale = weight_scale.to(torch.float32)
    if scale.dim() == 1:
        rows, scale_cols, padded_rows, padded_cols = _nvfp4_scale_shape(
            weight, block_size
        )
        if scale.numel() == rows * scale_cols:
            scale = scale.view(rows, scale_cols)
        elif scale.numel() == padded_rows * padded_cols:
            scale = scale.view(padded_rows, padded_cols)
        else:
            raise ValueError(
                f"NVFP4 weight_scale length {scale.numel()} does not match "
                f"({rows}, {scale_cols}) or the padded tile "
                f"({padded_rows}, {padded_cols})."
            )
    return weight_dequant(
        weight,
        scale * weight_scale_2.to(torch.float32).reshape(()),
        block_size=block_size,
        is_mx=True,
        is_packed=True,
    )
