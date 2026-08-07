#!/usr/bin/env python3
"""
HiF4 bitwise alignment: AMCT NPU kernel vs AMCT PyTorch vs HiFloat4-private To_HiFX golden.

The golden reference To_HiFX is the column-wise (axis-0) 64-element block format.
AMCT's HiF4 is row-wise (axis=-1). We compare by transposing appropriately.
"""
import sys, time
from pathlib import Path
import numpy as np
import torch

# ===========================================================================
# 1. HiFloat4-private To_HiFX golden (from np_ref/hifx4.py + _common.py)
#    Column-wise 64-element block, three-level scale: E6M2 + E1_8 + E1_16, S1P2 code.
# ===========================================================================

import torch as _torch_for_bf16

def To_BF16_vec(x):
    """Round to bf16 using torch.bfloat16 (round-half-to-even)."""
    return _torch_for_bf16.from_numpy(np.asarray(x, dtype=np.float32)).bfloat16().float().numpy()

def BF16_to_E6MX(x, M=2):
    """Convert bf16 value to E6Mx format.
    E6M2: 2^[-48, 15] * 1.M2, E6_offset=48, max=2^15*1.50=49152, min=2^-48.
    """
    E6_OFFSET = 48
    E6M2_MAX_VAL = 2.0 ** 15 * 1.5
    E6M2_MIN_VAL = 2.0 ** -E6_OFFSET
    xf = np.asarray(x, dtype=np.float64).clip(E6M2_MIN_VAL, E6M2_MAX_VAL)
    e = np.floor(np.log2(xf))
    mant = np.round(xf * 2.0 ** (-e + M))
    return mant * 2.0 ** (e - M)


def To_HiFX(x, N=4, M=2, G=64):
    """NumPy golden: column-wise HiF4/HiF5 quantize+dequantize.

    Groups G=64 elements along axis 0 (column), three-level E6M2+E1_8+E1_16 scale,
    S1PNg in-group code. N=4 → Ng=2 → code range [0, 7].

    Input: float32/fp64 array of shape (M_i, N_i).
    Returns: float32 array of same shape.
    """
    x = np.asarray(x, dtype=np.float64)
    Mi, Ni = x.shape
    Mcnt = int(np.ceil(Mi / G))
    Ng = N - 2  # 2 for HiF4
    res = np.zeros((Mi, Ni), dtype=np.float32)

    for i in range(Mcnt):
        row_start = i * G
        row_end = min(row_start + G, Mi)
        actual_G = row_end - row_start
        if actual_G < G:
            # pad: not needed since we only call on multiple-of-64
            continue

        for j in range(Ni):
            ori = x[row_start:row_end, j]
            finite_mask = np.isfinite(ori)
            ori_finite = np.where(finite_mask, ori, 0.0)

            S = np.where(ori_finite < 0, -1.0, 1.0)
            tmpG = np.abs(ori_finite)

            V16 = np.array([np.max(tmpG[k*4:k*4+4]) for k in range(16)], dtype=np.float32)
            V8 = np.array([np.max(V16[k*2:k*2+2]) for k in range(8)], dtype=np.float32)
            Vmax = np.max(V8)

            # Level-1: E6M2 scale factor
            Const_rec = To_BF16_vec(np.float32(1.0 / 7.0))
            SF = To_BF16_vec(Vmax * Const_rec)
            E6M2 = BF16_to_E6MX(SF, M=M)

            # Level-2: 8-element octave
            REC_E6M2 = To_BF16_vec(1.0 / E6M2)
            E1_8 = (V8 * REC_E6M2 >= 4.0).astype(np.float32)

            # Level-3: 4-element octave
            E1_8x2 = np.repeat(E1_8, 2)
            E1_16 = (V16 * REC_E6M2 * 2.0**(-E1_8x2) >= 2.0).astype(np.float32)

            # Reconstruct
            DE16 = E1_16 + E1_8x2
            DE64 = np.repeat(DE16, 4)

            igpv_bf16 = To_BF16_vec(tmpG * REC_E6M2 * 2.0**(-DE64))
            in_grp = np.floor(igpv_bf16 * 2.0**Ng + 0.5) * 2.0**(-Ng)  # RHA
            in_grp = np.where(in_grp >= 2, 2 - 2.0**(-Ng), in_grp)      # clip overflow

            grp = E6M2 * 2.0**DE64 * in_grp * S
            if not np.all(finite_mask):
                grp[:] = np.nan

            res[row_start:row_end, j] = grp.astype(np.float32)

    return res


def hif4_golden_rowwise(x_np):
    """Apply To_HiFX row-wise (like AMCT's qdim=-1) by transposing.

    To_HiFX groups 64 elements along axis 0 (column). We need grouping along
    axis -1 (row). So: transpose → To_HiFX → transpose back.
    """
    return To_HiFX(x_np.T.astype(np.float64), N=4).T


# ===========================================================================
# 2. AMCT implementations
# ===========================================================================
from amct_pytorch.quantization.dtypes.hifp_impl import hifloat4_fake_quant

try:
    from amct_ops.hifloat4_cast import hifloat4_fake_quant as hif4_npu_kernel
    import torch_npu
    HAS_NPU = torch.npu.is_available()
except Exception:
    HAS_NPU = False
    hif4_npu_kernel = None


# ===========================================================================
# 3. Test harness
# ===========================================================================

def pad64(x):
    """Pad last dim to multiple of 64."""
    C = x.shape[-1]
    pad = (64 - C % 64) % 64
    if pad:
        return np.pad(x, ((0, 0), (0, pad)), mode='constant')
    return x

def compare(label, x, tol=0):
    """Three-way comparison: golden vs AMCT-PyTorch vs AMCT-NPU."""
    x_np = x.numpy().astype(np.float32)
    x_np_pad = pad64(x_np)
    M, N_pad = x_np_pad.shape

    golden = hif4_golden_rowwise(x_np_pad)
    # Unpad
    N_orig = x_np.shape[-1]
    if N_pad != N_orig:
        golden = golden[:, :N_orig]

    golden_t = torch.from_numpy(golden.copy())

    # AMCT PyTorch (CPU) — handles padding internally
    x_bf16 = torch.from_numpy(x_np).to(torch.bfloat16)
    amct_py = hifloat4_fake_quant(x_bf16, qdim=-1).float()

    diff_py = (golden_t - amct_py).abs().max().item()

    if HAS_NPU:
        x_npu = x_bf16.npu()
        amct_npu = hif4_npu_kernel(x_npu, qdim=-1).cpu().float()
        diff_npu = (golden_t - amct_npu).abs().max().item()
        diff_py_npu = (amct_py - amct_npu).abs().max().item()
    else:
        diff_npu = None; diff_py_npu = None

    ok = diff_py < 1e-7
    status = "OK" if ok else "MISMATCH"
    parts = [f"[{status}] {label}: shape={tuple(x.shape)} golden-vs-AMCT-Py={diff_py:.2e}"]
    if HAS_NPU:
        parts.append(f"golden-vs-NPU={diff_npu:.2e} Py-vs-NPU={diff_py_npu:.2e}")
    print(" ".join(parts))
    return ok


# ===========================================================================
# 4. Run
# ===========================================================================
if __name__ == "__main__":
    t0 = time.time()
    rng = np.random.default_rng(42)

    test_cases = [
        ("gauss 64x64",         rng.standard_normal((64, 64), dtype=np.float32) * 0.2),
        ("gauss 64x128",        rng.standard_normal((64, 128), dtype=np.float32) * 0.2),
        ("gauss 128x512",       rng.standard_normal((128, 512), dtype=np.float32) * 3.0),
        ("gauss 256x1024",      rng.standard_normal((256, 1024), dtype=np.float32) * 5.0),
        ("gauss 4096x4096",     rng.standard_normal((4096, 4096), dtype=np.float32) * 0.5),
        ("zeros 64x64",         np.zeros((64, 64), dtype=np.float32)),
        ("zeros 128x256",       np.zeros((128, 256), dtype=np.float32)),
        ("dynamic range 64x64", np.concatenate([
            np.full((16, 64), 30.0, dtype=np.float32),
            (rng.standard_normal((48, 64)) * 0.01).astype(np.float32)])),
        ("large values",        (rng.standard_normal((128, 256)) * 10.0).astype(np.float32)),
        ("near zero",           (rng.standard_normal((64, 128)) * 0.001).astype(np.float32)),
        ("like in_proj_a 32x4096", (rng.standard_normal((32, 4096)) * 0.1).astype(np.float32)),
        ("like mlp 12288x4096", (rng.standard_normal((12288, 4096)) * 0.1).astype(np.float32)),
    ]

    print(f"HiF4 Three-Way Alignment Test")
    print(f"  Golden: HiFloat4-private To_HiFX (1D 64-element block)")
    print(f"  AMCT PyTorch: hifloat4_fake_quant (CPU, hif4_encode+hif4_decode)")
    print(f"  AMCT NPU: {'amct_ops.hifloat4_cast' if HAS_NPU else 'NOT AVAILABLE'}")
    print(f"{'='*80}")

    results = []
    all_ok = True
    for label, x_np in test_cases:
        x = torch.from_numpy(x_np.astype(np.float32))
        ok = compare(label, x)
        results.append((label, ok))
        if not ok:
            all_ok = False

    elapsed = time.time() - t0
    print(f"{'='*80}")
    print(f"  Time: {elapsed:.1f}s")
    print(f"  {'ALL PASSED' if all_ok else 'SOME FAILED'}")
    print(f"{'='*80}")
    print("\nSummary:")
    for label, ok in results:
        print(f"  {'PASS' if ok else 'FAIL'}: {label}")
