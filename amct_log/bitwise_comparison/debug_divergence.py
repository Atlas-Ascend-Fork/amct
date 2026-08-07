#!/usr/bin/env python3
"""Step-by-step comparison: find where AMCT and HiFloat4-private PyTorch golden diverge."""
import torch
import numpy as np

# ===========================================================================
# Helpers from both sides
# ===========================================================================
# AMCT helpers (from hifp_impl.py)
E6_OFFSET = 48
E6M2_MBITS = 2

def amct_pow2(e):
    return torch.ldexp(torch.ones_like(e, dtype=torch.float32), e.to(torch.int32))

def amct_to_bf16(x):
    return x.bfloat16().float()

# HiFloat4-private helpers (from _exact.py)
FP32_ABS_MASK = 0x7FFFFFFF
FP32_EXP_MASK = 0x7F800000
FP32_MANTISSA_MASK = 0x007FFFFF

def hif4_floor_log2_fp32(x):
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

def hif4_ldexp_fp32(x, exp):
    return torch.ldexp(x, exp.to(torch.int32))

def hif4_pow2_fp32(exp):
    return hif4_ldexp_fp32(torch.ones_like(exp, dtype=torch.float32), exp)

def hif4_round_to_bf16_fp32(x):
    x32 = x.to(torch.float32)
    shifted = x32.view(torch.int32) + 0x8000
    rounded = shifted & ~0xFFFF
    rounded2 = torch.where((x32.view(torch.int32) & 0x10000) != 0, rounded - 0x10000, rounded)
    return rounded2.view(torch.float32)

# Create a small test case with values that stress rounding
torch.manual_seed(1)
x = torch.randn(4, 128, dtype=torch.float32) * 0.3

# Both use the same input shape [M, gn, 8, 2, 4]
M, N = x.shape
gn = N // 64
xg = x.reshape(M, gn, 8, 2, 4)
print(f"Input shape: {x.shape}, reshaped: {xg.shape}")

# ===========================================================================
# Step 1: abs
# ===========================================================================
amct_abs = xg.abs()
hif4_abs = xg.abs()
print(f"Step 1 (abs): max diff = {(amct_abs - hif4_abs).abs().max().item():.2e} — MATCH")

# ===========================================================================
# Step 2: v16 = max over last dim (4 elements)
# ===========================================================================
amct_v16 = amct_abs.amax(dim=-1, keepdim=True)
hif4_v16 = hif4_abs.amax(dim=-1, keepdim=True)
print(f"Step 2 (v16): max diff = {(amct_v16 - hif4_v16).abs().max().item():.2e} — MATCH")

# ===========================================================================
# Step 3: v8 = max over dim -2 (8 elements cumulative)
# ===========================================================================
amct_v8 = amct_v16.amax(dim=-2, keepdim=True)
hif4_v8 = hif4_v16.amax(dim=-2, keepdim=True)
print(f"Step 3 (v8): max diff = {(amct_v8 - hif4_v8).abs().max().item():.2e} — MATCH")

# ===========================================================================
# Step 4: vmax = max over dim -3 (64 elements cumulative)
# ===========================================================================
amct_vmax = amct_v8.amax(dim=-3, keepdim=True)
hif4_vmax = hif4_v8.amax(dim=-3, keepdim=True)
print(f"Step 4 (vmax): max diff = {(amct_vmax - hif4_vmax).abs().max().item():.2e} — MATCH")

# ===========================================================================
# Step 5: sf = bf16(vmax * 1/7)
# ===========================================================================
rec7 = 1.0 / 7.0

amct_rec7_bf16 = amct_to_bf16(torch.tensor(rec7, dtype=torch.float32))
hif4_rec7_bf16 = hif4_round_to_bf16_fp32(torch.tensor(rec7, dtype=torch.float32))
print(f"Step 5a (rec7 bf16): amct={amct_rec7_bf16.item():.10f} hif4={hif4_rec7_bf16.item():.10f} diff={(amct_rec7_bf16 - hif4_rec7_bf16).abs().item():.2e}")

amct_sf = amct_to_bf16(amct_vmax * amct_rec7_bf16)
hif4_sf = hif4_round_to_bf16_fp32(hif4_vmax * hif4_rec7_bf16)
print(f"Step 5b (sf): max diff = {(amct_sf - hif4_sf).abs().max().item():.2e}")
if (amct_sf - hif4_sf).abs().max().item() > 0:
    diff_mask = (amct_sf - hif4_sf).abs() > 0
    n_diff = diff_mask.sum().item()
    print(f"  *** DIVERGENCE: {n_diff} elements differ ***")
    # Show an example
    idx = torch.where(diff_mask.flatten())[0][:3]
    for i in idx:
        a = amct_sf.flatten()[i].item()
        b = hif4_sf.flatten()[i].item()
        print(f"  amct={a:.10f} hif4={b:.10f} vmax={amct_vmax.flatten()[i].item():.10f}")

# ===========================================================================
# Step 5c: Compare bf16 rounding on simple values
# ===========================================================================
test_vals = torch.tensor([0.1428571492433548, 0.042857144, 0.5, 1.0/3.0, 0.10000000149011612], dtype=torch.float32)
print(f"\nBF16 rounding comparison:")
for v in test_vals:
    amct_bf = amct_to_bf16(v)
    hif4_bf = hif4_round_to_bf16_fp32(v)
    diff = (amct_bf - hif4_bf).abs().item()
    mark = " <-- DIFF" if diff > 0 else ""
    print(f"  fp32={v.item():.15f}  amct_bf16={amct_bf.item():.10f}  hif4_bf16={hif4_bf.item():.10f}  diff={diff:.2e}{mark}")

# Test on a sweep of values to find divergence patterns
print(f"\nSweeping random values for bf16 diff...")
torch.manual_seed(42)
sweep = torch.randn(100000, dtype=torch.float32) * 3.0
amct_sweep = amct_to_bf16(sweep)
hif4_sweep = hif4_round_to_bf16_fp32(sweep)
diffs = (amct_sweep - hif4_sweep).abs()
n_diff = (diffs > 0).sum().item()
print(f"  {n_diff}/{100000} values differ ({100*n_diff/100000:.2f}%)")
if n_diff > 0:
    idx = torch.where(diffs > 0)[0][:5]
    for i in idx:
        v = sweep[i]
        print(f"  fp32={v.item():.15f} amct={amct_sweep[i].item():.10f} hif4={hif4_sweep[i].item():.10f}")

# Check if the diff is systematic (always one direction)
if n_diff > 0:
    diff_vals = amct_sweep[diffs > 0] - hif4_sweep[diffs > 0]
    print(f"  Sign of diff: all positive={torch.all(diff_vals > 0).item()}, all negative={torch.all(diff_vals < 0).item()}")

# ===========================================================================
# Step 6: Compare e6m2 rounding
# ===========================================================================
print(f"\nE6M2 rounding comparison:")
# Quick comparison of e6m2 on sf
amct_sf_c = amct_sf.clamp(2.0**-E6_OFFSET, 2.0**15 * 1.5)

# AMCT e6m2
amct_e_e6 = torch.floor(torch.log2(amct_sf_c))
amct_q = torch.round(amct_sf_c * amct_pow2(2.0 - amct_e_e6))
amct_e6m2 = amct_q * amct_pow2(amct_e_e6 - 2.0)

# HiF4 e6m2 (using hif4_floor_log2 and hif4_ldexp)
hif4_sf_c = hif4_sf.clamp(2.0**-E6_OFFSET, 2.0**15 * 1.5)
hif4_e_e6 = hif4_floor_log2_fp32(hif4_sf_c)
hif4_q = torch.round(hif4_ldexp_fp32(hif4_sf_c, -hif4_e_e6 + 2.0))
hif4_e6m2 = hif4_ldexp_fp32(hif4_q, hif4_e_e6 - 2.0)

print(f"  log2: diff = {(amct_e_e6 - hif4_e_e6).abs().max().item():.2e}")
print(f"  q:    diff = {(amct_q - hif4_q).abs().max().item():.2e}")
print(f"  e6m2: diff = {(amct_e6m2 - hif4_e6m2).abs().max().item():.2e}")

# Check log2 divergence specifically
log2_diff = (amct_e_e6 - hif4_e_e6).abs()
if log2_diff.max() > 0:
    idx = torch.where(log2_diff.flatten() > 0)[0][:3]
    print(f"  log2 diverges at {len(torch.where(log2_diff.flatten() > 0)[0])} points:")
    for i in idx:
        v = amct_sf_c.flatten()[i].item()
        print(f"    sf={v:.10f} amct_log2={amct_e_e6.flatten()[i].item():.6f} hif4_log2={hif4_e_e6.flatten()[i].item():.6f}")
else:
    print(f"  log2 IDENTICAL — divergence is from bf16 rounding only")
