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
# Parameters stored next to an NVFP4 weight: the E4M3 block scale, the per-tensor scale
# and, when activations are quantized too, the activation scale. All of them are either
# consumed by the dequantization or meaningless once the weight is a float tensor.
NVFP4_PARAM_SUFFIXES = (".weight_scale", ".weight_scale_2", ".input_scale")
# `weight_scale_2` is what makes a checkpoint recognisable as NVFP4.
NVFP4_MARKER_SUFFIX = ".weight_scale_2"
# Packed 4-bit weights arrive as raw bytes, unlike FP8 weights which keep a float dtype.
_PACKED_DTYPES = (torch.uint8, torch.int8)


def nvfp4_scale_names(weight_name: str):
    """Checkpoint keys of the block scale and the per-tensor scale of `weight_name`."""
    return f"{weight_name}_scale", f"{weight_name}_scale_2"


def is_nvfp4_weight(weight: torch.Tensor, weight_name: str, weight_map) -> bool:
    """Whether `weight_name` is an NVFP4 weight rather than an FP8 or MXFP4 one.

    NVFP4 packs two elements per byte just like MXFP4, so the dtype alone cannot tell
    them apart. The discriminator is the per-tensor `weight_scale_2`, which only NVFP4
    carries next to the E4M3 block scale.
    """
    block_scale_name, global_scale_name = nvfp4_scale_names(weight_name)
    return (
        weight.dtype in _PACKED_DTYPES
        and block_scale_name in weight_map
        and global_scale_name in weight_map
    )


def is_nvfp4_scale_key(weight_name: str, weight_map) -> bool:
    """Whether `weight_name` is a quantization parameter of an NVFP4 weight.

    Requiring a sibling `weight_scale_2` keeps identically named parameters of other
    formats, which have to survive the conversion, out of the match.
    """
    for suffix in NVFP4_PARAM_SUFFIXES:
        if weight_name.endswith(suffix):
            base_name = weight_name[: -len(suffix)]
            return f"{base_name}{NVFP4_MARKER_SUFFIX}" in weight_map
    return False


def unpack_nvfloat4_to_fp32(packed_tensor: torch.Tensor) -> torch.Tensor:
    """Decode a packed NVFP4 tensor: every byte holds two 4-bit E2M1 codes.

    The last dimension doubles, so a [M, N // 2] packed weight becomes [M, N]. The bytes
    are viewed as uint8 first because the codes are unsigned; reading them out of a signed
    int8 tensor would sign-extend the high nibble.
    """
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
    """Decode the E4M3 block scale to FP32 and lay it out as [rows, cols]."""
    if weight_scale.dtype in _PACKED_DTYPES:
        weight_scale = weight_scale.view(torch.float8_e4m3fn)
    scale = weight_scale.to(torch.float32)
    if scale.dim() == 1 and scale.numel() == rows * cols:
        scale = scale.view(rows, cols)
    if scale.dim() == 2:
        # Checkpoints may pad the block-scale rows/columns up to a tile size.
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
