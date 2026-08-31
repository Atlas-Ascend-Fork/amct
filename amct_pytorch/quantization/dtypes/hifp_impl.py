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

import functools
from typing import NamedTuple

import numpy as np
import torch


_HIFLOAT8_BACKEND_ERRORS = (RuntimeError, OSError)
_HIFLOAT8_BACKEND_REQUIRED = (
    "HiFloat8 fake quantization is unavailable. To use HiFloat8, enable "
    "native HiFloat8 cast support in torch_npu or install amct_ops."
)


@functools.lru_cache(maxsize=1)
def is_native_hifloat8_cast_available():
    """Check native HiFloat8 cast support with a minimal round trip."""
    try:
        import torch_npu
    except ImportError:
        return False

    if not hasattr(torch_npu, "hifloat8") or not hasattr(torch_npu, "npu_dtype_cast"):
        return False

    try:
        fp_tensor = torch.zeros(1, dtype=torch.float16).npu()
        hifloat8_tensor = torch_npu.npu_dtype_cast(fp_tensor, torch_npu.hifloat8)
        torch_npu.npu_dtype_cast(
            hifloat8_tensor,
            torch.float16,
            input_dtype=torch_npu.hifloat8,
        )
    except Exception:
        return False
    return True


def _native_hifloat8_fake_quant(fp_tensor):
    """Run a native FP-to-HiFloat8-to-FP cast without fallback."""
    import torch_npu

    hifloat8_tensor = torch_npu.npu_dtype_cast(fp_tensor, torch_npu.hifloat8)
    return torch_npu.npu_dtype_cast(
        hifloat8_tensor,
        fp_tensor.dtype,
        input_dtype=torch_npu.hifloat8,
    )


def _load_amct_ops_cast():
    """Load amct_ops HiFloat8 cast functions only when requested."""
    try:
        from amct_ops.hifloat8_cast import (
            decode_from_hifloat8,
            encode_to_hifloat8,
        )
    except ImportError:
        return None
    return encode_to_hifloat8, decode_from_hifloat8


def _amct_ops_hifloat8_fake_quant(fp_tensor, encode, decode):
    """Run an amct_ops HiFloat8 round trip and restore input metadata."""
    work_dtype = (
        fp_tensor.dtype
        if fp_tensor.dtype in (torch.float16, torch.bfloat16)
        else torch.bfloat16
    )
    work_tensor = fp_tensor.to(work_dtype)
    if work_tensor.device.type != "npu":
        work_tensor = work_tensor.npu()
    codes = encode(work_tensor)
    return decode(codes, work_dtype).to(device=fp_tensor.device, dtype=fp_tensor.dtype)


@torch.no_grad()
def hifloat8_fake_quant(fp_tensor):
    """Run HiFloat8 fake quant with native cast and amct_ops fallback."""
    if is_native_hifloat8_cast_available():
        try:
            return _native_hifloat8_fake_quant(fp_tensor)
        except _HIFLOAT8_BACKEND_ERRORS:
            pass

    try:
        ops = _load_amct_ops_cast()
    except _HIFLOAT8_BACKEND_ERRORS:
        ops = None
    if ops is not None:
        try:
            return _amct_ops_hifloat8_fake_quant(fp_tensor, *ops)
        except _HIFLOAT8_BACKEND_ERRORS:
            pass

    raise RuntimeError(_HIFLOAT8_BACKEND_REQUIRED)


# ---- HiF4 encode / pack ------------------------------------------------
E6_OFFSET = 48
E6M2_MBITS = 2
N_FMT = 4
NG = N_FMT - 2  # S1PNg, Ng=2 -> in-group code 3 bits

# Bit-exact primitives (aligned with HiFloat4-private _exact.py)
FP32_ABS_MASK = 0x7FFFFFFF
FP32_EXP_MASK = 0x7F800000
FP32_MANTISSA_MASK = 0x007FFFFF


def _floor_log2_fp32(x):
    """Exact floor(log2(abs(x))) for fp32, via IEEE-754 bit extraction."""
    if x.dtype != torch.float32:
        raise TypeError("_floor_log2_fp32 expects torch.float32")
    bits = x.contiguous().view(torch.int32) & FP32_ABS_MASK
    exp_field = (bits & FP32_EXP_MASK) >> 23
    mantissa = bits & FP32_MANTISSA_MASK
    mantissa_msb = torch.zeros_like(mantissa)
    for bit in range(1, 23):
        mantissa_msb = torch.where(
            mantissa >= (1 << bit),
            torch.full_like(mantissa_msb, bit),
            mantissa_msb,
        )
    normal_exp = exp_field - 127
    subnormal_exp = mantissa_msb - 149
    exponent = torch.where(exp_field == 0, subnormal_exp, normal_exp).to(torch.float32)
    exponent = torch.where(bits == 0, torch.full_like(exponent, -torch.inf), exponent)
    return torch.where(torch.isfinite(x), exponent, x)


def _ldexp_fp32(x, exp):
    """Exact x * 2**exp via ldexp (bit-exact on NPU)."""
    return torch.ldexp(x, exp.to(torch.int32))


def _pow2(e):
    """Exact 2**e via ldexp (bit-exact on NPU)."""
    return _ldexp_fp32(torch.ones_like(e, dtype=torch.float32), e)


def _to_bf16(x):
    """Round to bf16 (round-half-to-even) and widen back to fp32.

    Aligned with round_to_bf16_fp32 in the HiFloat4 reference (_exact.py): the
    bf16 cast is RNE on CPU and bit-exact RNE on Ascend NPU. A hand-rolled int32
    bit trick is not used here because it mis-rounds values whose low 16 bits
    exceed half-ULP while the bf16 mantissa LSB is set.
    """
    return x.to(torch.float32).to(torch.bfloat16).to(torch.float32)


class Hif4Encoded(NamedTuple):
    sign_bit: torch.Tensor
    e6m2: torch.Tensor
    de: torch.Tensor
    code: torch.Tensor
    e1_8: torch.Tensor
    e1_16: torch.Tensor
    e_e6: torch.Tensor
    q: torch.Tensor
    block_nan: torch.Tensor


@torch.no_grad()
def hif4_encode(xg, ng=2, qdim=-1):
    """HiF4 encode: 3-level block scale + S1P2 in-group code."""
    finite_mask = torch.isfinite(xg)
    block_nan = (~finite_mask).flatten(start_dim=qdim - 2).any(dim=-1)
    xf = torch.where(finite_mask, xg, torch.zeros_like(xg))
    tmp = xf.abs()
    sign_bit = (xf < 0).to(torch.int32)

    v16 = tmp.amax(dim=qdim, keepdim=True)
    v8 = v16.amax(dim=qdim - 1, keepdim=True)
    vmax = v8.amax(dim=qdim - 2, keepdim=True)

    const_rec = _to_bf16(torch.tensor(1.0 / 7.0, dtype=torch.float32, device=xg.device))
    sf = _to_bf16(vmax * const_rec)
    sf_c = sf.clamp(2.0**-E6_OFFSET, 2.0**15 * 1.5)
    e_e6 = _floor_log2_fp32(sf_c)
    q = torch.round(sf_c * _pow2(2.0 - e_e6))
    e6m2_val = q * _pow2(e_e6 - 2.0)
    rec = _to_bf16(1.0 / e6m2_val)

    e1_8 = v8 * rec >= 4.0
    pl = e1_8.ndim + qdim - 1
    upper_start = pl + 1
    e1_8x2 = e1_8.expand(*e1_8.shape[:pl], 2, *e1_8.shape[upper_start:])
    e1_16 = v16 * rec * _pow2(-e1_8x2.to(torch.float32)) >= 2.0

    de = e1_16.to(torch.float32) + e1_8x2.to(torch.float32)
    in_grp = tmp * rec * _pow2(-de.to(tmp.dtype))
    in_grp = _to_bf16(in_grp) * (2**ng)
    code = torch.floor(in_grp + 0.5).clamp(0, (1 << (ng + 1)) - 1).to(torch.int32)

    e1_8_sq = e1_8.squeeze(qdim).squeeze(qdim)
    e6m2_sq = e6m2_val.squeeze(qdim).squeeze(qdim).squeeze(qdim)
    e_e6_sq = e_e6.squeeze(qdim).squeeze(qdim).squeeze(qdim)
    q_sq = q.squeeze(qdim).squeeze(qdim).squeeze(qdim)
    return Hif4Encoded(
        sign_bit,
        e6m2_sq,
        de.squeeze(qdim),
        code,
        e1_8_sq,
        e1_16.squeeze(qdim),
        e_e6_sq,
        q_sq,
        block_nan,
    )


@torch.no_grad()
def hif4_decode(sign_bit, e6m2, de, code, ng=2, block_nan=None, qdim=-1):
    """HiF4 decode: reconstruct fp32 from encode outputs."""
    de64 = de.unsqueeze(qdim)
    target = list(de64.shape)
    target[de64.ndim + qdim] = 4
    de64 = de64.expand(target)

    sign = 1.0 - 2.0 * sign_bit.to(torch.float32)
    e6m2_exp = e6m2
    nd_target = sign.ndim
    for pos in sorted([nd_target + qdim - 2, nd_target + qdim - 1, nd_target + qdim]):
        e6m2_exp = e6m2_exp.unsqueeze(pos)

    out = sign * e6m2_exp * _pow2(de64) * (code.to(torch.float32) / (2**ng))

    if block_nan is not None:
        bn_exp = block_nan
        for pos in sorted(
            [nd_target + qdim - 2, nd_target + qdim - 1, nd_target + qdim]
        ):
            bn_exp = bn_exp.unsqueeze(pos)
        while bn_exp.ndim < nd_target:
            bn_exp = bn_exp.unsqueeze(-1)
        out = torch.where(bn_exp, torch.full_like(out, float("nan")), out)
    return out


@torch.no_grad()
def hif4_pack(x):
    """Pack [n_rows, n_cols] tensor to HiF4 bytes (scale, value)."""
    x = x.detach().to(torch.float32)
    if x.dim() != 2:
        raise RuntimeError(
            "hif4_pack: input must be 2-D [n_rows, n_cols], got {} dims".format(x.dim())
        )
    n_rows, n_cols = x.shape
    if n_cols % 64 != 0:
        raise RuntimeError(
            "hif4_pack: n_cols must be a multiple of 64, got {}".format(n_cols)
        )
    gn = n_cols // 64

    xg = x.reshape(n_rows, gn, 8, 2, 4)
    enc = hif4_encode(xg, ng=NG)

    nibble = (enc.sign_bit * 8 + enc.code).reshape(n_rows, n_cols // 2, 2)
    value = (nibble[..., 0] + nibble[..., 1] * 16).to(torch.uint8)

    byte_e6m2 = (enc.e_e6 + E6_OFFSET).to(torch.int32) * (2**E6M2_MBITS) + (
        enc.q.to(torch.int32) - 4
    )
    byte_e6m2 = torch.where(enc.block_nan, torch.full_like(byte_e6m2, 0xFF), byte_e6m2)

    pw8 = _pow2(torch.arange(7, -1, -1, device=x.device, dtype=torch.float32)).to(
        torch.int32
    )
    byte_l1 = (enc.e1_8.to(torch.int32) * pw8).sum(dim=-1)
    e1_16f = enc.e1_16.reshape(n_rows, gn, 16).to(torch.int32)
    byte_l2_hi = (e1_16f[..., 0:8] * pw8).sum(dim=-1)
    byte_l2_lo = (e1_16f[..., 8:16] * pw8).sum(dim=-1)

    scale = torch.stack(
        [byte_e6m2.to(torch.int32), byte_l1, byte_l2_hi, byte_l2_lo], dim=-1
    ).to(torch.uint8)
    return scale, value


def hif4_unpack(scale, value):
    """Reconstruct [n_rows, n_cols] from packed HiF4 (scale, value) bytes."""
    if isinstance(scale, torch.Tensor):
        scale = scale.cpu().numpy()
    if isinstance(value, torch.Tensor):
        value = value.cpu().numpy()
    scale = np.asarray(scale, dtype=np.int64)
    value = np.asarray(value, dtype=np.int64)
    n_rows, gn, _ = scale.shape
    n_cols = gn * 64

    b_e6m2 = scale[..., 0]
    b_l1 = scale[..., 1]
    b_l2_hi = scale[..., 2]
    b_l2_lo = scale[..., 3]

    e = (b_e6m2 >> E6M2_MBITS) - E6_OFFSET
    mant = b_e6m2 & ((1 << E6M2_MBITS) - 1)
    e6m2 = torch.from_numpy(2.0**e * (1.0 + mant / (2**E6M2_MBITS))).float()

    a_idx = np.arange(8)
    e1_8 = torch.from_numpy(((b_l1[..., None] >> (7 - a_idx)) & 1).astype(np.float32))
    hi = torch.from_numpy(((b_l2_hi[..., None] >> (7 - a_idx)) & 1).astype(np.float32))
    lo = torch.from_numpy(((b_l2_lo[..., None] >> (7 - a_idx)) & 1).astype(np.float32))
    de = torch.cat([hi, lo], dim=-1).reshape(n_rows, gn, 8, 2) + e1_8.unsqueeze(-1)

    # Unpack nibbles: even element in low nibble
    lo = value & 0xF
    hi = (value >> 4) & 0xF
    nib = np.stack([lo, hi], axis=-1).reshape(n_rows, gn, 8, 2, 4)
    sign_bit = torch.from_numpy(((nib >> 3) & 1).astype(np.int32))
    code = torch.from_numpy((nib & 0x7).astype(np.int32))
    block_nan = torch.from_numpy((b_e6m2 == 0xFF))

    out = hif4_decode(sign_bit, e6m2, de, code, ng=NG, block_nan=block_nan)
    return out.reshape(n_rows, n_cols).to(torch.get_default_dtype())


def _load_amct_ops_hif4_cast():
    """Load amct_ops HiF4 fake-quant kernel only when requested."""
    try:
        from amct_ops.hifloat4_cast import hifloat4_fake_quant as _npu_fq
    except ImportError:
        return None
    return _npu_fq


@torch.no_grad()
def hifloat4_fake_quant(fp_tensor, qdim=-1):
    """HiF4 fake-quant: NPU kernel when available, else pure-torch reference."""
    if fp_tensor.device.type == "npu" and fp_tensor.dtype in (
        torch.float16,
        torch.bfloat16,
    ):
        _npu_fq = _load_amct_ops_hif4_cast()
        if _npu_fq is not None:
            return _npu_fq(fp_tensor, qdim=qdim)

    return _hif4_fake_quant_reference(fp_tensor, qdim)


@torch.no_grad()
def _hif4_fake_quant_reference(fp_tensor, qdim=-1):
    """Pure-torch HiF4 fake-quant, aligned with HiFloat4-private quant_hifx."""
    dtype_ori = fp_tensor.dtype
    x = fp_tensor.to(torch.float32)

    if qdim >= 0:
        qdim = qdim - len(x.shape)

    # The quant dim must be a multiple of 64, consistent with hif4_pack and the
    # NPU kernel: non-aligned input raises instead of being padded.
    dim_len = x.shape[qdim]
    if dim_len % 64 != 0:
        raise RuntimeError(
            "hifloat4_fake_quant: quant dim length must be a multiple of 64, "
            "got {}".format(dim_len)
        )

    out = _hif4_reference_quantize(x, qdim)

    return out.to(dtype_ori)


@torch.no_grad()
def _hif4_reference_quantize(x, qdim):
    """quant_hifx three-level block quantization on the padded fp32 tensor."""
    # Match quant_hifx algorithm exactly (HiFloat4-private QFuncs/hifx.py)
    xg = x.unflatten(qdim, (-1, 8, 2, 4))
    special_mask = ~torch.isfinite(xg)
    x_finite = torch.where(special_mask, torch.zeros_like(xg), xg)
    x_unsigned = torch.abs(x_finite)
    sign = torch.sign(x_finite)

    max_lv3 = torch.amax(x_unsigned, dim=qdim, keepdim=True)
    max_lv2 = torch.amax(max_lv3, dim=qdim - 1, keepdim=True)
    max_lv1 = torch.amax(max_lv2, dim=qdim - 2, keepdim=True)
    zero_group = max_lv1 == 0

    div7 = _to_bf16(torch.ones_like(max_lv1) / 7.0)
    scale_factor = max_lv1 * div7
    safe_scale = torch.where(
        scale_factor > 0, scale_factor, torch.ones_like(scale_factor)
    )
    e_sf = _floor_log2_fp32(safe_scale)
    mant_sf = _ldexp_fp32(safe_scale, -e_sf + 7.0)
    scale_factor = _ldexp_fp32(torch.round(mant_sf), e_sf - 7.0)
    scale_factor = scale_factor.clamp(min=2.0 ** (-48), max=49152.0)

    e_sf = _floor_log2_fp32(scale_factor)
    scale_factor = _ldexp_fp32(
        torch.round(_ldexp_fp32(scale_factor, 2.0 - e_sf)), e_sf - 2.0
    )
    scale_factor = torch.where(
        zero_group, torch.full_like(scale_factor, 2.0 ** (-48)), scale_factor
    )

    rec_sf = (1.0 / scale_factor).to(torch.bfloat16).to(x.dtype)

    scale_lv2_exp = torch.floor((max_lv2 * rec_sf).clamp(0, 4) * 0.25)
    scale_lv2 = _pow2(scale_lv2_exp)
    scale_lv3_input = _ldexp_fp32(max_lv3 * rec_sf, -scale_lv2_exp)
    scale_lv3_exp = torch.floor(scale_lv3_input.clamp(0, 2) * 0.5)
    scale_lv3 = _pow2(scale_lv3_exp)

    mant = _ldexp_fp32(x_unsigned * rec_sf, -scale_lv2_exp - scale_lv3_exp)
    mant = _to_bf16(mant)
    mant = _ldexp_fp32(
        torch.floor(_ldexp_fp32(mant, torch.full_like(mant, NG)) + 0.5),
        torch.full_like(mant, -float(NG)),
    )
    mant = mant.clamp(min=-2.0 + 2.0 ** (-NG), max=2.0 - 2.0 ** (-NG))

    out = sign * mant * scale_lv2 * scale_lv3 * scale_factor

    group_nan = special_mask.any(dim=qdim, keepdim=True)
    group_nan = group_nan.any(dim=qdim - 1, keepdim=True)
    group_nan = group_nan.any(dim=qdim - 2, keepdim=True)
    out = torch.where(
        torch.broadcast_to(group_nan, out.shape), torch.full_like(out, torch.nan), out
    )

    return out.flatten(qdim - 3, qdim)
