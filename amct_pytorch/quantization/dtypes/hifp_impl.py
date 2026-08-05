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
import re
from copy import deepcopy
from typing import NamedTuple

import numpy as np
import torch
import torch.nn.functional as F


class QType():
    """Floating-point quant-format descriptor.

    Parses ``e*m*k*b*`` / ``e*m*K*k*B*b*`` universal descriptors as well as
    special format names (``hifx*``, ``mxfp*``, ``nvf4``, ``fp16``, etc.).
    ``hifx4`` rewrites to ``e0m3K1k4B1b64Coff38`` (exp_bits=0, man_bits=3,
    64-element hierarchical block).
    """
    desc: str
    exp_bits: int = -1
    man_bits: int = -1
    k_bits: int = -1
    k_outer_bits: int = 0
    blk_size: int = 1
    blk_outer_size: int = 1
    exp_max: int = -1
    exp_min: int = -1
    k_max: int = -1
    fp_val_max: float = -1
    q_dim: int = -1
    man_shift_bit: int = -1
    exp_offset: int = 0
    do_carry: bool = True

    def __init__(self, desc: str):
        self.desc = desc
        if desc in ['fp16', 'fp32', 'bf16', 'int8sym']:
            ...
        elif 'nf4' in desc:
            res = re.match(r'^nf4B([0-9]+)b([0-9]+)$', desc)
            if res is None:
                raise ValueError('Quant type string must be [nf4B*b*]')
            self.blk_outer_size = int(res.group(1))
            self.blk_size = int(res.group(2))
        elif desc.lower()[:3] == 'tmx':
            res = re.match(r'^tmx([0-9]+)$', desc)
            self.blk_size = 16
            if res is None:
                raise ValueError('Specify bit number for tmx*')
            self.man_bits = int(res.group(1)) - 2
        elif desc.lower() == 'nvf4':
            self.blk_size = 16
        else:
            if desc.lower() == 'mxfp4':
                desc = 'e2m1k8b32c'
            elif desc.lower() == 'mxfp6e3m2':
                desc = 'e2m3k8b32c'
            elif desc.lower() == 'mxfp8e4m3':
                desc = 'e4m3k8b32c'
            elif desc.lower() == 'mxfp8e5m2':
                desc = 'e5m2k8b32c'
            elif desc.lower()[:4] == 'hifx':
                res = re.match(r'^hifx([2345])$', desc.lower())
                if res is None:
                    raise ValueError('HiFx only supports hifx*')
                n_bit_tmp = int(res.group(1)) - 1
                desc = 'e0m%dK1k4B1b%dCoff38' % (n_bit_tmp, 64)

            res = re.match(r'^e([0-9]+)m([0-9]+)k([0-9]+)b([0-9]+)([Cc]?)$', desc)
            res2 = re.match(
                r'^e([0-9]+)m([0-9]+)K([0-9]+)k([0-9]+)B([0-9]+)b([0-9]+)([Cc]?)(off[0-9]+)?$',
                desc,
            )
            if res is not None:
                self.exp_bits = int(res.group(1))
                self.man_bits = int(res.group(2))
                self.k_bits = int(res.group(3))
                self.blk_size = int(res.group(4))
                offset_number = None
                if res.group(5) is None:
                    self.do_carry = False
                else:
                    self.do_carry = str(res.group(5)) == 'C'
            elif res2 is not None:
                self.exp_bits = int(res2.group(1))
                self.man_bits = int(res2.group(2))
                self.k_outer_bits = int(res2.group(3))
                self.k_bits = int(res2.group(4))
                self.blk_outer_size = int(res2.group(5))
                self.blk_size = int(res2.group(6))
                if res2.group(7) is None:
                    self.do_carry = False
                else:
                    self.do_carry = str(res2.group(7)) == 'C'
                if res2.group(8) is not None:
                    offset_number = int(res2.group(8)[3:])
                else:
                    offset_number = None
            else:
                raise ValueError(
                    'Quant type string must be [e*m*k*b* or e*m*K*k*B*b*], '
                    'or special float types [fp16, fp32, bf16, int8sym]'
                )

            assert self.exp_bits != 1, 'exp_bits==1 is not supported. E1Mx is equivalent to E0M(x+1)'
            assert self.man_bits >= 1, 'man_bits should >=1'

            if self.exp_bits == 0:
                self.exp_max = self.man_bits - 1
                self.exp_min = 0
            else:
                self.exp_max = 2 ** (self.exp_bits - 1)
                if self.exp_bits == 5 and self.man_bits == 2:
                    self.exp_max -= 1
                self.exp_min = - 2 ** (self.exp_bits - 1) + 2

            self.k_max = 2 ** (self.k_bits + self.k_outer_bits - 1) - 1

            if offset_number is None:
                self.exp_offset = self.exp_max
            else:
                self.exp_offset = offset_number - self.k_max - 1 + self.exp_max

            self.man_shift_bit = self.man_bits

            if self.exp_bits == 4 and self.man_bits == 3:
                self.fp_val_max = 448
            else:
                self.fp_val_max = (
                    2 ** self.exp_max
                    * float(2 ** (self.man_bits + 1) - 1)
                    / 2 ** self.man_bits
                )
                self.fp_val_max = min(self.fp_val_max, 1e38)

    def dim_(self, dim: int):
        self.q_dim = dim
        return self

    def dim(self, dim: int):
        out = deepcopy(self)
        out.q_dim = dim
        return out

    def copy(self):
        return deepcopy(self)

    def __repr__(self) -> str:
        return f'QType: {self.desc}   Dim: {self.q_dim}  ExpOffset: {self.exp_offset}'


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



E6_OFFSET = 48          # E6M2 exponent bias: unbiased in [-48, 15] -> biased in [0, 63]
E6M2_MBITS = 2          # E6M2 mantissa bits
N_FMT = 4               # HiF4
NG = N_FMT - 2          # S1PNg, Ng = 2 -> in-group code is 3 bits, value = code / 2**Ng

def _pow2(e):
    """Exact 2**e via ldexp -- writes the exponent field only (bit-exact on NPU)."""
    return torch.ldexp(torch.ones_like(e, dtype=torch.float32), e.to(torch.int32))


def _to_bf16(x):
    """Round to bf16 and back to fp32 (round-half-to-even), matching the hardware scale path."""
    return x.bfloat16().float()



class Hif4Encoded(NamedTuple):
    """Per-64-group HiF4 encode outputs. ``B`` = all batch dims before the (8, 2, 4) block."""
    sign_bit: torch.Tensor    # int32 [*B, 8, 2, 4]  1 where x < 0 (zeros -> 0)
    e6m2: torch.Tensor        # fp32  [*B]           level-1 E6M2 scale value per 64-group
    de: torch.Tensor          # fp32  [*B, 8, 2]     lv2 + lv3 exponent in {0, 1, 2}
    code: torch.Tensor        # int32 [*B, 8, 2, 4]  S1P2 magnitude index in [0, 2**(ng+1) - 1]
    e1_8: torch.Tensor        # bool  [*B, 8]        level-2 (8x) exponent bit (for byte packing)
    e1_16: torch.Tensor       # bool  [*B, 8, 2]     level-3 (16x) exponent bit (for byte packing)
    e_e6: torch.Tensor        # fp32  [*B]           E6M2 unbiased exponent (for byte packing)
    q: torch.Tensor           # fp32  [*B]           E6M2 mantissa quotient in {4..7} (for packing)
    block_nan: torch.Tensor   # bool  [*B]           True where the 64-group has any non-finite (NaN/Inf)


@torch.no_grad()
def hif4_encode(xg, ng=2, qdim=-1):
    """HiF4 encode: 3-level block scale + S1P2 in-group code."""
    # Non-finite handling -- HiF4 block-poison: zero non-finite values before the max
    # reduction; if any element in a 64-group is non-finite the whole group becomes NaN.
    finite_mask = torch.isfinite(xg)
    block_nan = (~finite_mask).flatten(start_dim=qdim - 2).any(dim=-1)
    xf = torch.where(finite_mask, xg, torch.zeros_like(xg))
    tmp = xf.abs()
    sign_bit = (xf < 0).to(torch.int32)

    # Max hierarchy with keepdim=True (matching private hifx.py).
    # All tensors below share xg.ndim; 1s in reduced dims make broadcast with rec trivial.
    v16 = tmp.amax(dim=qdim, keepdim=True)
    v8 = v16.amax(dim=qdim - 1, keepdim=True)
    vmax = v8.amax(dim=qdim - 2, keepdim=True)

    # E6M2 (level-1 scale)
    const_rec = _to_bf16(torch.tensor(1.0 / 7.0, dtype=torch.float32, device=xg.device))
    sf = _to_bf16(vmax * const_rec)
    sf_c = sf.clamp(2.0 ** -E6_OFFSET, 2.0 ** 15 * 1.5)
    e_e6 = torch.floor(torch.log2(sf_c))
    q = torch.round(sf_c * _pow2(2.0 - e_e6))
    e6m2_val = q * _pow2(e_e6 - 2.0)
    rec = _to_bf16(1.0 / e6m2_val)

    # Level-2 (8x) and level-3 (16x) exponent bits — all same ndim, rec broadcasts naturally.
    e1_8 = (v8 * rec >= 4.0)                                  # [*B, 8, 1, 1, …]
    # Expand the "2" keepdim (at qdim-1) from 1 → 2.
    pl = e1_8.ndim + qdim - 1                                  # split before "2" keepdim
    e1_8x2 = e1_8.expand(*e1_8.shape[:pl], 2, *e1_8.shape[pl + 1:])
    e1_16 = (v16 * rec * _pow2(-e1_8x2.to(torch.float32)) >= 2.0)

    # In-group S1P2 code
    de = e1_16.to(torch.float32) + e1_8x2.to(torch.float32)
    in_grp = tmp * rec * _pow2(-de.to(tmp.dtype))
    in_grp = _to_bf16(in_grp) * (2 ** ng)
    code = torch.floor(in_grp + 0.5).clamp(0, (1 << (ng + 1)) - 1).to(torch.int32)

    # Collapse keepdim-1s back to the shapes expected by hif4_decode / packer.
    # ─ sign_bit, code: original xg shape, no squeeze needed.
    # ─ e6m2_val, e_e6, q: 3 keepdim 1s at qdim-2, qdim-1, qdim → squeeze backwards.
    # ─ e1_8: 2 keepdim 1s at qdim-1, qdim → squeeze backwards.
    # ─ e1_16, de: 1 keepdim 1 at qdim → squeeze that one.
    e1_8_sq = e1_8.squeeze(qdim).squeeze(qdim)       # squeeze backwards
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
    # Expand de from [*B, 8, 2] to [*B, 8, 2, 4].
    de64 = de.unsqueeze(qdim)
    target = list(de64.shape)
    target[de64.ndim + qdim] = 4
    de64 = de64.expand(target)

    sign = 1.0 - 2.0 * sign_bit.to(torch.float32)

    # Broadcast e6m2 [*B] to match sign's ndim: insert 1s at positions
    # qdim-2 (8), qdim-1 (2), qdim (4) of the target tensor.
    e6m2_exp = e6m2
    nd_target = sign.ndim
    for pos in sorted([nd_target + qdim - 2, nd_target + qdim - 1, nd_target + qdim]):
        e6m2_exp = e6m2_exp.unsqueeze(pos)

    out = sign * e6m2_exp * _pow2(de64) * (code.to(torch.float32) / (2 ** ng))

    if block_nan is not None:
        bn_exp = block_nan
        # Insert 1s at the block-keepdim positions, then pad any trailing dims.
        for pos in sorted([nd_target + qdim - 2, nd_target + qdim - 1, nd_target + qdim]):
            bn_exp = bn_exp.unsqueeze(pos)
        while bn_exp.ndim < nd_target:
            bn_exp = bn_exp.unsqueeze(-1)
        out = torch.where(bn_exp, torch.full_like(out, float("nan")), out)
    return out



@torch.no_grad()
def quant_dequant_float(x, Q, force_py=False, force_fp32=False, **kwargs):
    """Pure-torch hifx4 fake-quant: pad → encode → decode → unpad."""
    if not re.match(r'^hifx[0-9]*$', Q.desc):
        raise ValueError(
            f"hifloat4_quant only implements the 'hifx*' fake-quant path, got '{Q.desc}'. "
            "Add the corresponding QFuncs reference if another format is needed."
        )
    assert Q.exp_bits == 0

    dtype_ori = x.dtype
    if force_fp32:
        x = x.to(torch.float32)

    C = x.shape[Q.q_dim]
    blk_size_total = Q.blk_size * Q.blk_outer_size
    padC = (blk_size_total - C % blk_size_total) % blk_size_total
    qdim = Q.q_dim
    if qdim >= 0:
        qdim = qdim - len(x.shape)
    if padC > 0:
        pads = [0] * (-qdim * 2 - 1) + [padC]
        x = F.pad(x, pads, value=0.0)

    ng = Q.man_bits - 1                       # HiF4: man_bits=3 -> ng=2
    xb = x.unflatten(qdim, (-1, 8, 2, 4)).to(torch.float32)
    enc = hif4_encode(xb, ng=ng, qdim=qdim)
    out = hif4_decode(enc.sign_bit, enc.e6m2, enc.de, enc.code, ng=ng,
                      block_nan=enc.block_nan, qdim=qdim).flatten(qdim - 3, qdim)

    if padC > 0:
        slices = [slice(0, out.shape[i]) for i in range(len(x.shape) + qdim)] + [slice(0, C), ]
        out = out[tuple(slices)]

    if out.dtype != dtype_ori:
        out = out.to(dtype_ori)
    return out


@torch.no_grad()
def hif4_cast(x, output="packed"):
    """Pack [M, N] tensor to HiF4 bytes (scale + value) or dequant round-trip."""
    if output not in ("packed", "dequant"):
        raise ValueError(f"output must be 'packed' or 'dequant', got {output!r}")

    dtype_ori = x.dtype
    x = x.detach().to(torch.float32)
    assert x.dim() == 2, "input must be [M, N]"
    M, N = x.shape
    assert N % 64 == 0, "N must be a multiple of 64"
    gn = N // 64

    xg = x.reshape(M, gn, 8, 2, 4)
    enc = hif4_encode(xg, ng=NG)

    if output == "dequant":
        dequant = hif4_decode(enc.sign_bit, enc.e6m2, enc.de, enc.code, ng=NG,
                              block_nan=enc.block_nan)
        return dequant.reshape(M, N).to(dtype_ori)

    # pack value: nibble = sign*8 + code, pair adjacent along N (even -> low nibble)
    nibble = (enc.sign_bit * 8 + enc.code).reshape(M, N // 2, 2)
    value = (nibble[..., 0] + nibble[..., 1] * 16).to(torch.uint8)      # [M, N//2]

    # pack scale (integer arithmetic, no bitwise ops)
    byte_e6m2 = ((enc.e_e6 + E6_OFFSET).to(torch.int32) * (2 ** E6M2_MBITS)
                 + (enc.q.to(torch.int32) - 4))                        # [M,gn]
    byte_e6m2 = torch.where(enc.block_nan,
                            torch.full_like(byte_e6m2, 0xFF), byte_e6m2)

    pw8 = _pow2(torch.arange(7, -1, -1, device=x.device, dtype=torch.float32)).to(torch.int32)
    byte_l1 = (enc.e1_8.to(torch.int32) * pw8).sum(dim=-1)              # [M,gn]
    e1_16f = enc.e1_16.reshape(M, gn, 16).to(torch.int32)              # k = a*2+b
    byte_l2_hi = (e1_16f[..., 0:8] * pw8).sum(dim=-1)
    byte_l2_lo = (e1_16f[..., 8:16] * pw8).sum(dim=-1)

    scale = torch.stack([byte_e6m2.to(torch.int32), byte_l1,
                         byte_l2_hi, byte_l2_lo], dim=-1).to(torch.uint8)
    return scale, value


def hif4_bit_reconstruct(scale, value):
    """Reconstruct fp32 [M, N] from packed HiF4 (scale, value) bytes."""
    if isinstance(scale, torch.Tensor):
        scale = scale.cpu().numpy()
    if isinstance(value, torch.Tensor):
        value = value.cpu().numpy()
    scale = np.asarray(scale, dtype=np.int64)
    value = np.asarray(value, dtype=np.int64)
    M, gn, _ = scale.shape
    N = gn * 64

    b_e6m2 = scale[..., 0]
    b_l1 = scale[..., 1]
    b_l2_hi = scale[..., 2]
    b_l2_lo = scale[..., 3]

    e = (b_e6m2 >> E6M2_MBITS) - E6_OFFSET
    mant = b_e6m2 & ((1 << E6M2_MBITS) - 1)
    e6m2 = (2.0 ** e * (1.0 + mant / (2 ** E6M2_MBITS)))                  # [M,gn]

    a_idx = np.arange(8)
    e1_8 = ((b_l1[..., None] >> (7 - a_idx)) & 1).astype(np.float64)      # [M,gn,8]
    hi = (b_l2_hi[..., None] >> (7 - a_idx)) & 1
    lo = (b_l2_lo[..., None] >> (7 - a_idx)) & 1
    e1_16 = np.concatenate([hi, lo], axis=-1).reshape(M, gn, 8, 2).astype(np.float64)

    nib = np.stack([value & 0xF, value >> 4], axis=-1).reshape(M, N)      # [M,N] element 0 in low nibble
    nib = nib.reshape(M, gn, 8, 2, 4)
    sign = np.where((nib >> 3) & 1 == 1, -1.0, 1.0)
    code = (nib & 0x7).astype(np.float64)
    in_grp_val = code / (2 ** NG)

    de = e1_16 + e1_8[..., None]                                          # [M,gn,8,2]
    dequant = sign * e6m2[..., None, None, None] * 2.0 ** de[..., None] * in_grp_val
    nan_block = (b_e6m2 == 0xFF)                                          # [M,gn]
    dequant = np.where(nan_block[..., None, None, None], np.float32(np.nan), dequant)
    return dequant.reshape(M, N).astype(np.float32)


@torch.no_grad()
def hifloat4_fake_quant(fp_tensor, qdim=-1):
    """FP -> HiFloat4 -> FP fake-quant round trip.

    The hifx4 format does its own 64-element block scaling along ``qdim``. Only
    ``float16``/``bfloat16`` tensors resident on NPU use the AscendC kernel;
    everything else falls back to the pure-torch reference.
    """
    if fp_tensor.device.type == "npu" and fp_tensor.dtype in (torch.float16, torch.bfloat16):
        try:
            from amct_ops.hifloat4_cast import hifloat4_fake_quant as _npu_hifloat4_fake_quant
        except ImportError:
            _npu_hifloat4_fake_quant = None
        if _npu_hifloat4_fake_quant is not None:
            return _npu_hifloat4_fake_quant(fp_tensor, qdim=qdim)

    qtype = QType('hifx4').dim(qdim)
    return quant_dequant_float(fp_tensor, qtype, force_py=True, force_fp32=True).to(fp_tensor.dtype)
