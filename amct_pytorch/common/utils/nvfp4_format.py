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

NVFP4 packs two 4-bit E2M1 codes per byte exactly like MXFP4, so unpacking the nibbles,
broadcasting the scale over each block and trimming padding are all reused from
``mxfp_impl``. The NVFP4 specifics live in the scale only: an E4M3 block scale covering 16
elements of a row, times one per-tensor FP32 scale. Folding those two into the single FP32
block scale ``weight_dequant`` expects is all this module does. TRT-LLM may store the
block scale flattened over a padded tile ``pad(rows, 128) * pad(cols, 4)``.

Scale tensor names are checkpoint specific, so resolving them is left to the model
adapters (``get_nvfp4_scale_name``).
"""

import torch

from amct_pytorch.quantization.dtypes.mxfp_impl import weight_dequant

NVFP4_BLOCK_SIZE = 16
# Packed 4-bit weights arrive as raw bytes, unlike FP8 weights which keep a float dtype.
NVFP4_PACKED_DTYPES = (torch.uint8, torch.int8)
_SF_ROW_ALIGN = 128
_SF_COL_ALIGN = 4


def nvfp4_weight_dequant(
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_scale_2: torch.Tensor,
    block_size: int = NVFP4_BLOCK_SIZE,
) -> torch.Tensor:
    """Dequantize a packed NVFP4 weight of shape (M, N // 2) into a float tensor (M, N).

    Args:
        weight (torch.Tensor): Packed E2M1 weight, two elements per byte.
        weight_scale (torch.Tensor): E4M3 block scale, 2-D or flattened over the padded
            TRT-LLM tile.
        weight_scale_2 (torch.Tensor): Per-tensor FP32 scale.
        block_size (int, optional): Elements sharing one block scale. Defaults to 16.

    Returns:
        torch.Tensor: The dequantized weight, in the default dtype.
    """
    if weight_scale.dtype in NVFP4_PACKED_DTYPES:
        weight_scale = weight_scale.view(torch.float8_e4m3fn)
    scale = weight_scale.to(torch.float32)
    if scale.dim() == 1:
        rows, cols = weight.shape[0], weight.shape[1] * 2 // block_size
        padded_rows = -(-rows // _SF_ROW_ALIGN) * _SF_ROW_ALIGN
        padded_cols = -(-cols // _SF_COL_ALIGN) * _SF_COL_ALIGN
        if scale.numel() == rows * cols:
            scale = scale.view(rows, cols)
        elif scale.numel() == padded_rows * padded_cols:
            scale = scale.view(padded_rows, padded_cols)
        else:
            raise ValueError(
                f"NVFP4 weight_scale length {scale.numel()} does not match "
                f"({rows}, {cols}) or the TRT-LLM padded tile "
                f"({padded_rows}, {padded_cols})."
            )
    return weight_dequant(
        weight,
        scale * weight_scale_2.to(torch.float32).reshape(()),
        block_size=block_size,
        is_mx=True,
        is_packed=True,
    )
