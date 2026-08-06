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
"""NVFP4 unpack / dequant helpers.

NVFP4 packs two 4-bit E2M1 codes per byte, exactly like MXFP4, so the nibble codec is
shared with ``mxfp_impl``. The formats differ in the scale: NVFP4 pairs a per-16-element
E4M3 block scale (``weight_scale``) with one FP32 per-tensor scale (``weight_scale_2``),
while MXFP4 uses a single E8M0 per-32-element scale. That second scale is also what lets
``convert_state_dict`` tell the two apart, since both arrive as 1-byte packed tensors.
"""

import torch

from amct_pytorch.quantization.dtypes.mxfp_impl import unpack_mxfloat4_to_fp32

NVFP4_BLOCK_SIZE = 16
NVFP4_SCALE_SUFFIXES = (".weight_scale", ".weight_scale_2")

# The block-scale tensor is stored over a padded row/column tiling when it is flattened.
_SF_ROW_ALIGN = 128
_SF_COL_ALIGN = 4


def is_nvfp4_scale_key(weight_name: str) -> bool:
    """Whether `weight_name` is an NVFP4 scale tensor.

    Scales are consumed by the dequantization and must not reach the decoded checkpoint.
    """
    return weight_name.endswith(NVFP4_SCALE_SUFFIXES)


def unpack_nvfloat4_to_fp32(packed_tensor: torch.Tensor) -> torch.Tensor:
    """Decode a packed NVFP4 tensor: every byte holds two 4-bit E2M1 codes.

    The last dimension doubles, so a [M, N // 2] packed weight becomes [M, N]. The bytes
    are viewed as uint8 first because the codes are unsigned; reading them out of a signed
    int8 tensor would sign-extend the high nibble.
    """
    if packed_tensor.dtype != torch.uint8:
        packed_tensor = packed_tensor.view(torch.uint8)
    return unpack_mxfloat4_to_fp32(packed_tensor)


def _expand_block_scale(weight_scale: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    """Decode the E4M3 block scale to FP32 and lay it out as [rows, cols]."""
    if weight_scale.dtype == torch.uint8:
        weight_scale = weight_scale.view(torch.float8_e4m3fn)
    scale = weight_scale.to(torch.float32)
    if scale.dim() == 1:
        padded_rows = -(-rows // _SF_ROW_ALIGN) * _SF_ROW_ALIGN
        padded_cols = -(-cols // _SF_COL_ALIGN) * _SF_COL_ALIGN
        scale = scale.view(padded_rows, padded_cols)
    elif scale.dim() != 2:
        raise ValueError(
            f"NVFP4 weight_scale must be 1-D or 2-D, got {scale.dim()} dimensions."
        )
    return scale[:rows, :cols]


def nvfp4_weight_dequant(
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_scale_2: torch.Tensor,
    block_size: int = NVFP4_BLOCK_SIZE,
) -> torch.Tensor:
    """Dequantize an NVFP4 packed weight into a float tensor.

    Each element is reconstructed as `e2m1_code * e4m3_block_scale * fp32_global_scale`,
    where the block scale covers `block_size` consecutive elements of a row.

    Args:
        weight (torch.Tensor): Packed E2M1 weight of shape (M, N // 2).
        weight_scale (torch.Tensor): E4M3 block scale, (M, N // block_size) or flattened.
        weight_scale_2 (torch.Tensor): FP32 per-tensor scale.
        block_size (int, optional): Elements sharing one block scale. Defaults to 16.

    Returns:
        torch.Tensor: The dequantized weight of shape (M, N), in the default dtype.
    """
    unpacked = unpack_nvfloat4_to_fp32(weight)
    rows, cols = unpacked.shape
    scale = _expand_block_scale(weight_scale, rows, cols // block_size)
    scale_expanded = scale.repeat_interleave(block_size, dim=1)[:rows, :cols]

    dequantized_weight = unpacked * scale_expanded * weight_scale_2.to(torch.float32)
    return dequantized_weight.to(torch.get_default_dtype())
