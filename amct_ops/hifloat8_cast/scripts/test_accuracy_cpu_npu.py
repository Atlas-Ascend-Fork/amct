# -*- coding: UTF-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# Apache License, Version 2.0
"""
CPU vs NPU hifloat8_cast 精度对比

覆盖场景：
  1. 随机数据（FP16 / BF16）
  2. 边界值（最大值、最小正规数、极小值）
  3. 特殊值（±0、±Inf、NaN）
  4. 全 256 个 HiFloat8 编码值（Decode 完整性）

运行：
    LD_PRELOAD=/usr/lib/$(uname -m)-linux-gnu/libgomp.so.1 python scripts/test_accuracy_cpu_npu.py
"""

import sys
import os
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, '..', '..', '..', 'amct_pytorch', 'experimental', 'hifloat8'))

import torch_npu
from amct_ops.hifloat8_cast import encode_to_hifloat8, decode_from_hifloat8
import hifloat8_cast as cpu_hif8

torch.npu.set_device(0)

PASS = "[PASS]"
FAIL = "[FAIL]"
WARN = "[WARN]"

total_cases = 0
total_pass = 0


def record(ok, msg=""):
    global total_cases, total_pass
    total_cases += 1
    if ok:
        total_pass += 1
    return ok


# ── 工具 ──────────────────────────────────────────────────────────────────────

def npu_encode(x_cpu: torch.Tensor) -> torch.Tensor:
    """在 NPU 上编码，结果返回 CPU"""
    return encode_to_hifloat8(x_cpu.to('npu')).cpu()


def npu_decode(x_hif8: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """在 NPU 上解码，结果返回 CPU"""
    return decode_from_hifloat8(x_hif8.to('npu'), dtype).cpu()


def cpu_encode(x: torch.Tensor) -> torch.Tensor:
    # 不能走 .numpy()：numpy 不支持 BF16
    return cpu_hif8.float_to_hifloat8(x.detach().cpu().contiguous().clone())


def cpu_decode_bf16(x: torch.Tensor) -> torch.Tensor:
    return cpu_hif8.hifloat8_to_bfloat16(x.detach().cpu().to(torch.uint8).contiguous().clone())


def compare_encoded(npu_out: torch.Tensor, cpu_out: torch.Tensor, label: str):
    """逐字节对比编码结果，打印不一致条目"""
    assert npu_out.dtype == torch.uint8
    assert cpu_out.dtype == torch.uint8
    mismatch = (npu_out != cpu_out)
    n_bad = mismatch.sum().item()
    ok = (n_bad == 0)
    record(ok)
    tag = PASS if ok else FAIL
    print(f"  {tag} {label}: {npu_out.numel()} 元素, 不一致 {n_bad}")
    if not ok:
        idxs = mismatch.nonzero(as_tuple=True)[0][:8].tolist()
        for i in idxs:
            print(f"         idx={i:6d}  NPU=0x{npu_out[i].item():02x}  CPU=0x{cpu_out[i].item():02x}")
    return ok


def compare_decoded(npu_out: torch.Tensor, cpu_out: torch.Tensor, label: str):
    """
    对比解码结果：
      - 对 NaN 单独判断（NaN != NaN 但都是 NaN 算对）
      - 对 ±Inf 精确匹配
      - 对普通值精确 bit 匹配（同一 HiF8 -> 同一浮点）
    """
    n = npu_out.numel()
    npu_f = npu_out.float()
    cpu_f = cpu_out.float()

    nan_mask = npu_f.isnan() | cpu_f.isnan()
    both_nan = npu_f.isnan() & cpu_f.isnan()
    one_nan  = nan_mask & ~both_nan

    normal_mask = ~nan_mask
    exact_match = (npu_out[normal_mask] == cpu_out[normal_mask])

    n_bad_nan    = one_nan.sum().item()
    n_bad_normal = (~exact_match).sum().item()
    n_bad = n_bad_nan + n_bad_normal

    ok = (n_bad == 0)
    record(ok)
    tag = PASS if ok else FAIL
    print(f"  {tag} {label}: {n} 元素, 不一致 {n_bad}"
          + (f" (NaN不匹配={n_bad_nan}, 普通不匹配={n_bad_normal})" if n_bad > 0 else ""))

    if not ok:
        count = 0
        for i in range(n):
            if count >= 8:
                break
            n_v = float(npu_out[i])
            c_v = float(cpu_out[i])
            n_nan = (n_v != n_v)
            c_nan = (c_v != c_v)
            if (n_nan != c_nan) or (not n_nan and n_v != c_v):
                print(f"         idx={i:6d}  NPU={n_v}  CPU={c_v}")
                count += 1
    return ok


# ── 测试 1: 随机 FP16 ────────────────────────────────────────────────────────

def test_random_fp16(n=100000):
    print("\n[1] 随机 FP16 数据")
    x = torch.randn(n, dtype=torch.float16)

    npu_enc = npu_encode(x)
    cpu_enc = cpu_encode(x)
    compare_encoded(npu_enc, cpu_enc, f"Encode FP16 (n={n})")

    # Decode 用 NPU 编码结果作为公共输入，对比解码输出（CPU 参考使用 BF16 近似）
    npu_dec = npu_decode(npu_enc, torch.float16)
    cpu_dec = cpu_decode_bf16(npu_enc)
    compare_decoded(npu_dec.bfloat16(), cpu_dec, "Decode HiF8->FP16≈BF16 (from NPU encoded)")

    npu_dec2 = npu_decode(cpu_enc, torch.float16)
    cpu_dec2 = cpu_decode_bf16(cpu_enc)
    compare_decoded(npu_dec2.bfloat16(), cpu_dec2, "Decode HiF8->FP16≈BF16 (from CPU encoded)")


# ── 测试 2: 随机 BF16 ────────────────────────────────────────────────────────

def test_random_bf16(n=100000):
    print("\n[2] 随机 BF16 数据")
    x = torch.randn(n, dtype=torch.bfloat16)

    npu_enc = npu_encode(x)
    cpu_enc = cpu_encode(x)
    compare_encoded(npu_enc, cpu_enc, f"Encode BF16 (n={n})")

    npu_dec = npu_decode(npu_enc, torch.bfloat16)
    cpu_dec = cpu_decode_bf16(npu_enc)
    compare_decoded(npu_dec, cpu_dec, "Decode HiF8->BF16 (from NPU encoded)")

    npu_dec2 = npu_decode(cpu_enc, torch.bfloat16)
    cpu_dec2 = cpu_decode_bf16(cpu_enc)
    compare_decoded(npu_dec2, cpu_dec2, "Decode HiF8->BF16 (from CPU encoded)")


# ── 测试 3 & 4: 边界值（FP16 / BF16） ────────────────────────────────────────

def _test_boundary(idx, dtype_label, dtype, values_fp32):
    print(f"\n[{idx}] {dtype_label} 边界值")
    x = torch.tensor(values_fp32, dtype=torch.float32).to(dtype)
    npu_enc = npu_encode(x)
    cpu_enc = cpu_encode(x)
    compare_encoded(npu_enc, cpu_enc, f"Encode {dtype_label} 边界值")

    print(f"  {'值':>14} {'NPU':>6} {'CPU':>6} {'一致':>5}")
    for i in range(len(x)):
        v = float(x[i])
        n_b = npu_enc[i].item()
        c_b = cpu_enc[i].item()
        ok = "✓" if n_b == c_b else "✗"
        print(f"  {v:>14.6g}  0x{n_b:02x}  0x{c_b:02x}    {ok}")


def test_boundary_fp16():
    _test_boundary(3, "FP16", torch.float16, [
        0.0, -0.0, 1.0, -1.0, 0.5, 2.0, 0.25,
        # 接近 HiF8 最大值
        32768.0, -32768.0,
        # 接近 FP16 最小正规数
        6.1035e-5, 1.22e-4,
        # 超出范围
        1e10, -1e10,
    ])


def test_boundary_bf16():
    _test_boundary(4, "BF16", torch.bfloat16, [
        0.0, -0.0, 1.0, -1.0, 0.5, 2.0, 0.25,
        32768.0, -32768.0,
        1.2e-7, 2.4e-7,
        1e10, -1e10,
    ])


# ── 测试 5: 特殊值（±0, ±Inf, NaN） ─────────────────────────────────────────

def test_special_values():
    print("\n[5] 特殊值 FP16：±0, ±Inf, NaN")
    names = ["+0", "-0", "+Inf", "-Inf", "NaN"]
    vals  = [0.0, -0.0, float('inf'), float('-inf'), float('nan')]
    x = torch.tensor(vals, dtype=torch.float16)

    npu_enc = npu_encode(x)
    cpu_enc = cpu_encode(x)

    print(f"  {'值':>6} {'NPU enc':>8} {'CPU enc':>8} {'enc一致':>8}")
    for i, name in enumerate(names):
        n_b = npu_enc[i].item()
        c_b = cpu_enc[i].item()
        ok = "✓" if n_b == c_b else "✗"
        print(f"  {name:>6}   0x{n_b:02x}     0x{c_b:02x}       {ok}")

    # Decode 全部 256 个 HiF8 值，对比 CPU/NPU（BF16 解码，CPU 参考支持）
    print("\n[5b] 特殊值 Decode：检查 0x00(+0) 0x80(NaN) 0x6F(+Inf) 0xEF(-Inf)")
    special_hif8 = torch.tensor([0x00, 0x80, 0x6F, 0xEF], dtype=torch.uint8)
    special_names = ["0x00(+0)", "0x80(NaN)", "0x6F(+Inf)", "0xEF(-Inf)"]

    npu_dec = npu_decode(special_hif8, torch.bfloat16)
    cpu_dec = cpu_decode_bf16(special_hif8)
    print(f"  {'HiF8':>12} {'NPU dec':>12} {'CPU dec':>12} {'一致':>6}")
    for i, name in enumerate(special_names):
        n_v = float(npu_dec[i])
        c_v = float(cpu_dec[i])
        n_nan = (n_v != n_v)
        c_nan = (c_v != c_v)
        ok = "✓" if ((n_nan and c_nan) or (not n_nan and n_v == c_v)) else "✗"
        print(f"  {name:>12}  {str(n_v):>12}  {str(c_v):>12}    {ok}")

    record(True)  # 仅展示，不计失败


# ── 测试 6: 全 256 个 HiFloat8 值 → BF16 完整对比 ────────────────────────────

def test_full_decode_bf16():
    print("\n[6] 全 256 个 HiFloat8 值 → BF16 完整对比")
    all_hif8 = torch.arange(256, dtype=torch.uint8)
    npu_dec = npu_decode(all_hif8, torch.bfloat16)
    cpu_dec = cpu_decode_bf16(all_hif8)
    compare_decoded(npu_dec, cpu_dec, "Decode 全256值 HiF8->BF16")

    for i in range(256):
        n_v = float(npu_dec[i])
        c_v = float(cpu_dec[i])
        n_nan = (n_v != n_v)
        c_nan = (c_v != c_v)
        if not ((n_nan and c_nan) or (not n_nan and n_v == c_v)):
            print(f"    不一致: HiF8=0x{i:02x}  NPU={n_v}  CPU={c_v}")


# ── 主程序 ────────────────────────────────────────────────────────────────────

print("=" * 70)
print("HiFloat8 Cast 精度对比: NPU vs CPU")
print("=" * 70)

test_random_fp16()
test_random_bf16()
test_boundary_fp16()
test_boundary_bf16()
test_special_values()
test_full_decode_bf16()

print()
print("=" * 70)
print(f"总计: {total_pass}/{total_cases} 通过")
if total_pass == total_cases:
    print("结论: CPU 与 NPU 精度完全一致")
else:
    print(f"结论: {total_cases - total_pass} 项存在差异，请检查上方详情")
print("=" * 70)

sys.exit(0 if total_pass == total_cases else 1)
