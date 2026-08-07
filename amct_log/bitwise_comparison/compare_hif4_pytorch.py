#!/usr/bin/env python3
"""
Compare HiFloat4-private PyTorch golden vs AMCT PyTorch HiF4 on real Qwen3-8B weights.

Golden: To_HiFX row-wise (1D 64-el block, E6M2+E1_8+E1_16+S1P2) using _exact helpers
AMCT:   hifloat4_fake_quant (hif4_encode + hif4_decode)

Both: quantize + dequantize back to fp32, compare element-wise.
"""
import time, sys
from pathlib import Path
import numpy as np
import torch
import safetensors

# ===========================================================================
# 1. Bit-exact helpers from HiFloat4-private _exact.py
# ===========================================================================
FP32_ABS_MASK = 0x7FFFFFFF
FP32_EXP_MASK = 0x7F800000
FP32_MANTISSA_MASK = 0x007FFFFF

def floor_log2_fp32(x):
    if x.dtype != torch.float32:
        raise TypeError("floor_log2_fp32 expects torch.float32")
    bits = x.contiguous().view(torch.int32) & FP32_ABS_MASK
    exp_field = (bits & FP32_EXP_MASK) >> 23
    mantissa = bits & FP32_MANTISSA_MASK
    mantissa_msb = torch.zeros_like(mantissa)
    for bit in range(1, 23):
        mantissa_msb = torch.where(mantissa >= (1 << bit), torch.full_like(mantissa_msb, bit), mantissa_msb)
    normal_exp = exp_field - 127
    subnormal_exp = mantissa_msb - 149
    exponent = torch.where(exp_field == 0, subnormal_exp, normal_exp).to(torch.float32)
    exponent = torch.where(bits == 0, torch.full_like(exponent, -torch.inf), exponent)
    return torch.where(torch.isfinite(x), exponent, x)

def ldexp_fp32(x, exp):
    return torch.ldexp(x, exp.to(torch.int32))

def pow2_fp32(exp):
    return ldexp_fp32(torch.ones_like(exp, dtype=torch.float32), exp)

def round_to_bf16_fp32(x):
    x32 = x.to(torch.float32)
    rounded = (x32.view(torch.int32) + 0x8000) & ~0xFFFF
    rounded = torch.where((x32.view(torch.int32) & 0x10000) != 0,
                          rounded - 0x10000, rounded)
    return rounded.view(torch.float32)

# ===========================================================================
# 2. HiFloat4-private To_HiFX in PyTorch (row-wise, uses _exact helpers)
#    Column-wise golden: groups 64 elements along dim 0.
#    We apply along dim 1 (row-wise) for AMCT compatibility.
# ===========================================================================
E6M2_MAX = 2.0**15 * 1.5
E6M2_MIN = 2.0**-48
E6_OFFSET = 48

def _to_bf16(x):
    return round_to_bf16_fp32(x.float())

def _to_e6m2(x):
    xc = x.float().clamp(min=E6M2_MIN, max=E6M2_MAX)
    e = floor_log2_fp32(xc)
    units = torch.round(ldexp_fp32(xc, -e + 2))
    return ldexp_fp32(units, e - 2)

def _to_e1m2(x):
    a = _to_bf16(torch.abs(x))
    q = ldexp_fp32(torch.floor(ldexp_fp32(a, torch.full_like(a, 2.0)) + 0.5), torch.full_like(a, -2.0))
    q = torch.where(q >= 2.0, torch.full_like(q, 1.75), q)
    sign = torch.where(x < 0, torch.full_like(x, -1.0), torch.ones_like(x))
    return sign * q

@torch.no_grad()
def hif4_golden_pytorch(x, qdim=-1):
    """HiFloat4-private To_HiFX golden in PyTorch, applied along qdim (row-wise 1D blocks)."""
    ndim = x.dim()
    if qdim < 0:
        qdim += ndim
    if qdim != ndim - 1:
        x_work = x.transpose(qdim, ndim - 1).contiguous()
    else:
        x_work = x.contiguous()

    shape = tuple(x_work.shape)
    n = shape[-1]
    m = x_work.numel() // n
    xf = x_work.reshape(m, n).float()

    # 1D blocks of 64 along last dim
    gn = n // 64
    # Reshape to [M, gn, 8, 2, 4] — 64 elements = 8 * 2 * 4
    xg = xf.reshape(m, gn, 8, 2, 4)

    finite_mask = torch.isfinite(xg)
    xf_clean = torch.where(finite_mask, xg, torch.zeros_like(xg))
    tmp = xf_clean.abs()

    # Level 1: max over 64 → E6M2 scale
    v16 = tmp.amax(dim=-1, keepdim=True)    # max over 4
    v8 = v16.amax(dim=-2, keepdim=True)     # max over 2 → 8
    vmax = v8.amax(dim=-3, keepdim=True)    # max over 8 → 64

    rec7 = _to_bf16(torch.tensor(1.0 / 7.0, dtype=torch.float32, device=x.device))
    sf = _to_bf16(vmax * rec7)
    e6m2 = _to_e6m2(sf)
    e6m2_inv = _to_bf16(1.0 / e6m2)

    # Level 2: per-8 octave
    e1_8 = (v8 * e6m2_inv >= 4.0).float()

    # Level 3: per-4 octave
    e1_8_x2 = torch.repeat_interleave(e1_8, 2, dim=-2)
    e1_16 = (v16 * e6m2_inv * pow2_fp32(-e1_8_x2.float()) >= 2.0).float()

    de = e1_16.float() + e1_8_x2.float()

    # In-group quant
    igp = _to_bf16(tmp * e6m2_inv * pow2_fp32(-de))
    code = torch.floor(igp * 4.0 + 0.5).clamp(0, 7).float() / 4.0  # S1P2, Ng=2

    # Dequant
    out = e6m2 * pow2_fp32(de) * code
    sign = torch.where(xg < 0, -1.0, 1.0)
    out = out * sign

    # NaN block poisoning: any non-finite in 64-element group poisons the group
    block_nan = (~finite_mask).flatten(start_dim=-3).any(dim=-1)  # [M, gn] — True if block has NaN
    if block_nan.any():
        # Broadcast back: [M, gn] → [M, gn, 8, 2, 4]
        bn = block_nan.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        out = torch.where(bn.expand_as(out), torch.full_like(out, float('nan')), out)

    out = out.reshape(m, n).reshape(shape)
    if qdim != ndim - 1:
        out = out.transpose(qdim, ndim - 1).contiguous()
    return out


# ===========================================================================
# 3. AMCT HiF4
# ===========================================================================
from amct_pytorch.quantization.dtypes.hifp_impl import hifloat4_fake_quant


# ===========================================================================
# 4. Compare on Qwen3-8B weights
# ===========================================================================
QWEN3_8B = "/home/ma-user/work/l84414662/models/Qwen3-8B"
LAYERS_TO_TEST = [0, 17, 35]  # first, middle, last

shard_files = sorted(Path(QWEN3_8B).glob("model*.safetensors"))

print(f"HiF4 PyTorch: HiFloat4-private golden vs AMCT")
print(f"  On Qwen3-8B weight matrices (layers {LAYERS_TO_TEST})")
print(f"{'='*80}")

all_diffs = []
for shard in shard_files:
    with safetensors.safe_open(str(shard), framework="pt") as sf:
        for key in sorted(sf.keys()):
            # Only test specific layers' 2D weights
            parts = key.split(".")
            if "layers" not in parts:
                continue
            try:
                layer_idx = int(parts[parts.index("layers") + 1])
            except (ValueError, IndexError):
                continue
            if layer_idx not in LAYERS_TO_TEST:
                continue

            t = sf.get_tensor(key)
            if t.ndim != 2:
                continue
            if key.endswith((".bias", "_bias")):
                continue

            M, N = t.shape
            # Both need last dim div by 64 → pad if needed
            pad_n = (64 - N % 64) % 64
            if pad_n:
                w = torch.nn.functional.pad(t.float(), (0, pad_n))
            else:
                w = t.float()

            # HiFloat4-private golden (PyTorch, CPU)
            w_bf16 = w.to(torch.bfloat16)
            y_golden = hif4_golden_pytorch(w_bf16).float()

            # AMCT PyTorch
            y_amct = hifloat4_fake_quant(w_bf16, qdim=-1).float()

            # Unpad
            if pad_n:
                y_golden = y_golden[:, :N]
                y_amct = y_amct[:, :N]

            # Compare
            diff = (y_golden - y_amct).abs()
            max_diff = diff.max().item()
            mean_diff = diff.mean().item()
            n_mismatch = (diff > 0).sum().item()
            total = diff.numel()

            all_diffs.append(max_diff)

            # Also compute per-element SQNR between the two implementations
            sig = y_golden.square().sum()
            noise = (y_golden - y_amct).square().sum()
            sqnr = float("inf") if noise == 0 else float(10 * torch.log10(sig / noise))

            status = "BIT-EXACT" if max_diff == 0 else f"DIFF"
            print(f"  [{status}] L{layer_idx} {key.split('layers.'+str(layer_idx)+'.')[1]:<40s} "
                  f"[{M}×{N}] max={max_diff:.2e} mean={mean_diff:.2e} "
                  f"mismatch={n_mismatch}/{total} ({100*n_mismatch/total:.1f}%) "
                  f"impl_SQNR={sqnr:.1f}dB")

if all_diffs:
    arr = np.array(all_diffs)
    print(f"\n{'='*80}")
    print(f"  Summary: {len(all_diffs)} weights tested")
    print(f"  Bit-exact matches: {(arr == 0).sum()}")
    print(f"  Max element diff:  {arr.max():.2e}")
    print(f"  Mean max diff:     {arr.mean():.2e}")
    if (arr > 0).sum() > 0:
        print(f"  >0 diff weights:   {(arr > 0).sum()}")
    print(f"{'='*80}")
