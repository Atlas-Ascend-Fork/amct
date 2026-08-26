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

import numpy as np
import torch
from torch import Tensor


def round_ste(x: torch.Tensor) -> torch.Tensor:
    return (torch.round(x) - x).detach() + x


def dynamic_per_token_quant(tensor: torch.Tensor, bits: int) -> torch.Tensor:
    dtype = tensor.dtype
    qmax = 2 ** (bits - 1) - 1
    qmin = -(2 ** (bits - 1))
    abs_max = torch.abs(tensor).max(dim=-1, keepdim=True)[0].clamp(min=1e-5)
    scale = (abs_max / qmax).clamp_min(1e-9)
    tensor = round_ste(tensor / scale)
    tensor = torch.clamp(tensor, qmin, qmax)
    tensor = tensor * scale
    return tensor.to(dtype)


def scale_fp32_to_u64(weight_scale):
    """
    Convert FP32 scale to UINT64 scale for W4A8 MoEGMM.
    """
    k, n = weight_scale.shape
    scale_np = weight_scale.detach().float().cpu().numpy()
    scale_uint32 = scale_np.astype(np.float32)
    scale_uint32.dtype = np.uint32
    scale_uint64 = np.zeros((k, n * 2), dtype=np.uint32)
    scale_uint64[..., ::2] = scale_uint32
    scale_uint64.dtype = np.uint64
    scale_uint64 = torch.from_numpy(scale_uint64).to(torch.uint64)
    return scale_uint64


def pack_4bit(x: torch.Tensor):
    """
    Pack int4 weight for W4A8 MoEGMM. Each two int4 numbers are packed into one byte.
    """
    assert x.dtype == torch.int8
    x = x.T.contiguous()  # pack along output channel dim.
    shape = x.shape
    x = x.view(-1, 2)
    # for example, 5(0b00000101) << 4 -> 0b01010000, -7 (0b11111001) & 0b00001111 -> 0b00001001,
    # then 0b01010000 | 0b00001001 -> 0b01011001
    x1 = x[:, 0]
    x2 = x[:, 1]
    y_x2 = torch.bitwise_left_shift(x2, 4)
    y_x1 = x1 & 0b00001111
    y = torch.bitwise_or(y_x1, y_x2)
    y = y.view(shape[0], shape[1] // 2)
    return y.T.contiguous()


def weight_quant(tensor: torch.Tensor, bits=8, real_quant=False, v: torch.Tensor = 0.0):
    assert tensor.dim() == 2
    qmax = 2 ** (bits - 1) - 1
    abs_max = torch.abs(tensor).max(dim=1, keepdim=True)[0]
    scale = (abs_max / qmax).clamp_min(1e-9)
    assert scale.shape == (tensor.shape[0], 1)
    if isinstance(v, torch.Tensor):
        v = v.to(device=tensor.device, dtype=tensor.dtype)
    quantized = round_ste(tensor / scale + v)
    quantized = torch.clamp(quantized, -qmax, qmax)
    if real_quant:
        if bits == 4:
            # pack 4bit for W4A8 MoEGMM
            quantized = quantized.to(torch.int8)
            bias = int4_assistance_bias(quantized, scale)
            quantized = pack_4bit(quantized)
            scale = scale_fp32_to_u64(scale)
            return quantized, scale, bias
        else:
            return quantized.to(torch.int8), scale.to(torch.float32), None
    else:
        return quantized * scale


def int4_assistance_bias(weight, weight_scale):
    """
    Calculate the int4 weight assistance bias matrix for W4A8 MoEGEMM.
    """
    repeat_times = weight.shape[1] // weight_scale.shape[1]
    expanded_scale = weight_scale.repeat_interleave(repeat_times, dim=1)
    # 8 is the max value of INT4, for normalizing the quantization range of assistance bias.
    weight_assistant_matrix = (expanded_scale * weight * 8).sum(dim=1).float()
    return weight_assistant_matrix


# --- decode side ----------------------------------------------------------------------
# Reconstruct a *symmetric* integer weight (INT8 / INT4, no zero-point) to float, i.e. the
# "normal" int quant matrix cell in docs/official_docs/input_formats_packing_overview_CN.md:
# integer weight + float scale, w = int * scale. This is NOT the W4A8 MoEGMM packing above
# (pack_4bit / scale_fp32_to_u64) — those are a hardware-specific encode format.


def unpack_int4(packed: Tensor) -> Tensor:
    """Unpack a symmetric INT4 weight: 2 signed 4-bit ints per byte, low-first (even index in
    the low nibble), sign-extended. ``[..., N/2] -> int8 [..., N]``.

    Packed along the last dim (in-features), matching MXFP4/HiF4 packing in this repo — *not*
    the output-channel ``pack_4bit`` used by the W4A8 encode path.
    """
    u = packed.contiguous().view(torch.uint8)
    lo = (u & 0x0F).to(torch.int16)
    hi = ((u >> 4) & 0x0F).to(torch.int16)
    # sign-extend 4-bit: codes 8..15 represent -8..-1
    lo = torch.where(lo >= 8, lo - 16, lo)
    hi = torch.where(hi >= 8, hi - 16, hi)
    out = torch.stack([lo, hi], dim=-1).reshape(*packed.shape[:-1], packed.shape[-1] * 2)
    return out.to(torch.int8)


def int_weight_dequant(weight: Tensor, scale: Tensor, bits: int = 8) -> Tensor:
    """Decode a symmetric integer weight to float: ``w = int * scale`` (no zero-point).

    Args:
        weight: INT8 ``[M, N]`` (1 elem/byte) when ``bits==8``; packed INT4 ``[M, N/2]``
            (2 signed nibbles/byte, low-first) when ``bits==4``.
        scale: float multiplier — per-tensor (scalar/``[1]``), per-channel ``[M, 1]`` or
            per-group ``[M, N/g]``. Broadcast over its group along the last dim.
        bits: 8 or 4.

    Returns:
        The dequantized weight cast to the default (float) dtype.
    """
    if bits == 4:
        weight = unpack_int4(weight)
    elif bits != 8:
        raise ValueError(f"int_weight_dequant supports bits in (4, 8), got {bits}")
    w = weight.to(torch.float32)
    s = scale.to(torch.float32)
    # Expand a per-group scale to the weight's last dim; per-channel [M,1], per-tensor scalar
    # and per-element scales broadcast as-is.
    if s.dim() == w.dim() and s.shape[-1] not in (0, 1, w.shape[-1]):
        if w.shape[-1] % s.shape[-1] != 0:
            raise ValueError(
                f"INT scale last dim {s.shape[-1]} does not tile weight last dim "
                f"{w.shape[-1]}."
            )
        s = s.repeat_interleave(w.shape[-1] // s.shape[-1], dim=1)
    return (w * s).to(torch.get_default_dtype())
