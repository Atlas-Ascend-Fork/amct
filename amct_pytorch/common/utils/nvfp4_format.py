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
"""NVFP4 unpack / dequant helpers used by convert_state_dict.

NVFP4 packs two 4-bit E2M1 codes per byte (same packing as MXFP4). It is identified by
the companion scales ``weight_scale`` (E4M3, per 16 elements) and ``weight_scale_2``
(per-tensor FP32). TRT-LLM may store ``weight_scale`` as a 1-D buffer over a padded tile
``pad(rows, 128) * pad(cols, 4)`` before cropping.
"""

import torch

from amct_pytorch.quantization.dtypes.mxfp_impl import unpack_mxfloat4_to_fp32

NVFP4_BLOCK_SIZE = 16
_PACKED_DTYPES = (torch.uint8, torch.int8)
_SF_ROW_ALIGN = 128
_SF_COL_ALIGN = 4


def _nvfp4_scale_name_candidates(weight_name: str):
    """Possible NVFP4 scale key pairs; naming is not unified across checkpoints."""
    return (
        # ``xxx.weight`` -> ``xxx.weight_scale`` / ``xxx.weight_scale_2``
        (f"{weight_name}_scale", f"{weight_name}_scale_2"),
        # ``xxx.weight`` -> ``xxx.scale`` / ``xxx.scale_2``
        (
            weight_name.replace(".weight", ".scale"),
            weight_name.replace(".weight", ".scale_2"),
        ),
    )


def nvfp4_scale_names(weight_name: str, weight_map=None):
    """Resolve block / global scale keys for `weight_name`.

    If ``weight_map`` is given, return the first candidate pair that exists in it.
    Otherwise default to the ``_scale`` / ``_scale_2`` layout.
    """
    candidates = _nvfp4_scale_name_candidates(weight_name)
    if weight_map is not None:
        for block_name, global_name in candidates:
            if block_name in weight_map and global_name in weight_map:
                return block_name, global_name
    return candidates[0]


def is_nvfp4_weight(weight: torch.Tensor, weight_name: str, weight_map) -> bool:
    """Whether `weight_name` is an NVFP4 weight rather than FP8 or MXFP4."""
    if weight.dtype not in _PACKED_DTYPES:
        return False
    for block_name, global_name in _nvfp4_scale_name_candidates(weight_name):
        if block_name in weight_map and global_name in weight_map:
            return True
    return False


def unpack_nvfloat4_to_fp32(packed_tensor: torch.Tensor) -> torch.Tensor:
    """Decode packed NVFP4: every byte holds two 4-bit E2M1 codes; last dim doubles."""
    if packed_tensor.element_size() != 1:
        raise ValueError(
            f"NVFP4 weights are packed one byte per two elements, "
            f"got dtype {packed_tensor.dtype}."
        )
    if packed_tensor.dtype != torch.uint8:
        packed_tensor = packed_tensor.view(torch.uint8)
    return unpack_mxfloat4_to_fp32(packed_tensor)


def _expand_block_scale(
    weight_scale: torch.Tensor, rows: int, cols: int
) -> torch.Tensor:
    """Decode E4M3 block scale to FP32 laid out as [rows, cols]."""
    if weight_scale.dtype in _PACKED_DTYPES:
        weight_scale = weight_scale.view(torch.float8_e4m3fn)
    scale = weight_scale.to(torch.float32)
    if scale.dim() == 1:
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
    elif scale.dim() != 2:
        raise ValueError(
            f"NVFP4 weight_scale must be 1-D or 2-D, got {scale.dim()} dimensions."
        )
    scale = scale[:rows, :cols]
    if tuple(scale.shape) != (rows, cols):
        raise ValueError(
            f"NVFP4 weight_scale must hold ({rows}, {cols}) block scales for this "
            f"weight, got shape {tuple(weight_scale.shape)}."
        )
    return scale


def nvfp4_weight_dequant(
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_scale_2: torch.Tensor,
    block_size: int = NVFP4_BLOCK_SIZE,
) -> torch.Tensor:
    """Dequantize packed NVFP4 weight: e2m1 * e4m3_block_scale * fp32_global_scale."""
    unpacked = unpack_nvfloat4_to_fp32(weight)
    if unpacked.dim() != 2:
        raise ValueError(
            f"NVFP4 dequantization expects a 2-D weight, got shape {tuple(weight.shape)}."
        )
    rows, cols = unpacked.shape
    if cols % block_size != 0:
        raise ValueError(
            f"NVFP4 weight has {cols} columns, which is not a multiple of the "
            f"{block_size}-element block size."
        )
    scale = _expand_block_scale(weight_scale, rows, cols // block_size)
    scale_expanded = scale.repeat_interleave(block_size, dim=1)
    global_scale = weight_scale_2.to(torch.float32)
    if global_scale.numel() != 1:
        raise ValueError(
            f"NVFP4 weight_scale_2 must be a per-tensor scalar, "
            f"got shape {tuple(weight_scale_2.shape)}."
        )

    dequantized_weight = unpacked * scale_expanded * global_scale.reshape(())
    return dequantized_weight.to(torch.get_default_dtype())
