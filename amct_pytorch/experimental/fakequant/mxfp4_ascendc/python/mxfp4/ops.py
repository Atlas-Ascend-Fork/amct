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
"""Thin Python wrappers around torch.ops.amct.quant_dequant_mxfp4."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def quant_dequant_mxfp4(
    x: torch.Tensor,
    block_size: int = 32,
    inv_scale_factor_scale: float = 1.0,
) -> torch.Tensor:
    """MXFP4 quant-dequant on Ascend NPU.

    Drop-in replacement for the PyTorch reference ``reference.mxfp4.quant_dequant_mxfp4()``.
    Input tensor must be on an NPU device. AIV core count is queried at
    runtime via ``PlatformAscendC::GetCoreNumAiv()`` (no manual ``num_cores``).

    Args:
        x: Any shape, float32 (or auto-cast), on ``npu`` device.
        block_size: Quantisation block width. Must be 32.
        inv_scale_factor_scale: Runtime multiplier applied on top of the
            kernel's built-in ``INV_SCALE_FACTOR`` (1/6.0). Default ``1.0``
            reproduces the standard MXFP4 behaviour.

    Returns:
        Same shape / dtype / device as *x*.

    Raises:
        ValueError: If ``block_size != 32``, scale is non-positive, or input
            is not on NPU.
        RuntimeError: If the native extension is not loaded or kernel fails.

    Example:
        >>> y = quant_dequant_mxfp4(x_npu)
        >>> y = torch.ops.amct.quant_dequant_mxfp4(x_flat, 1.0)  # flat float32
    """
    if block_size != 32:
        raise ValueError(
            f"only block_size=32 supported by the NPU kernel, got {block_size}"
        )

    if inv_scale_factor_scale <= 0:
        raise ValueError(
            f"inv_scale_factor_scale must be positive, got {inv_scale_factor_scale}"
        )

    orig_dtype = x.dtype
    if not x.is_npu:
        raise ValueError(f"input must be on an NPU device, got device={x.device}")
    # Pad along the last dim first (per-row blocks), matching the PyTorch
    # reference. Flattening before pad would merge adjacent rows into the
    # same MXFP4 block when last_dim is not a multiple of block_size.
    x_fp = x.to(torch.float32)
    if x_fp.ndim == 0:
        raise ValueError("x must have at least 1 dimension")
    last_dim = x_fp.shape[-1]
    pad = (block_size - last_dim % block_size) % block_size
    if pad:
        x_fp = F.pad(x_fp, (0, pad))
    padded_shape = x_fp.shape
    x_flat = x_fp.reshape(-1).contiguous()

    y_flat = torch.ops.amct.quant_dequant_mxfp4(x_flat, float(inv_scale_factor_scale))

    y_fp = y_flat.reshape(padded_shape)
    if pad:
        y_fp = y_fp[..., :last_dim]

    return y_fp.to(orig_dtype)
