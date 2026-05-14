# -*- coding: UTF-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# Apache License, 2.0
"""
HiFloat8 Cast 精度测试（仅 NPU）

测试覆盖：
  1. 类型推断（FP16/BF16 输入自动识别，非法类型报错）
  2. 编解码往返（随机数据、边界值）
  3. 特殊值（±0、±Inf、NaN）
  4. 非规格化数（Denormal/Subnormal）
  5. 全 256 个 HiFloat8 值 decode 完备性

运行：
    python tests/test_accuracy.py
"""

import torch
import torch_npu
from amct_ops.hifloat8_cast import encode_to_hifloat8, decode_from_hifloat8

torch.npu.set_device(0)


# ── 工具函数 ──────────────────────────────────────────────────────────────
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


# ── 测试 1: 类型推断 ──────────────────────────────────────────────────────
def test_type_inference():
    print("\n[1] 类型推断测试")
    print("-" * 60)

    # FP16 输入
    x_fp16 = torch.randn(1000, dtype=torch.float16, device='npu')
    y_fp16 = encode_to_hifloat8(x_fp16)
    ok = y_fp16.dtype == torch.uint8
    record(ok)
    print(f"  {PASS if ok else FAIL} FP16 输入 → uint8 输出")

    # BF16 输入
    x_bf16 = torch.randn(1000, dtype=torch.bfloat16, device='npu')
    y_bf16 = encode_to_hifloat8(x_bf16)
    ok = y_bf16.dtype == torch.uint8
    record(ok)
    print(f"  {PASS if ok else FAIL} BF16 输入 → uint8 输出")

    # 非法类型
    for bad_dtype, name in [(torch.float32, "FP32"), (torch.int32, "INT32")]:
        try:
            x_bad = torch.randn(1000, dtype=bad_dtype, device='npu') if bad_dtype != torch.int32 \
                else torch.zeros(1000, dtype=bad_dtype, device='npu')
            encode_to_hifloat8(x_bad)
            record(False)
            print(f"  {FAIL} {name} 输入应报错但未报错")
        except (ValueError, RuntimeError):
            record(True)
            print(f"  {PASS} {name} 输入正确报错")


# ── 测试 2: Decode 输出类型 ────────────────────────────────────────────────
def test_decode_output_type():
    print("\n[2] Decode 输出类型测试")
    print("-" * 60)

    y = torch.randint(0, 255, (1000,), dtype=torch.uint8, device='npu')

    # 默认输出 BF16
    z_default = decode_from_hifloat8(y)
    ok = z_default.dtype == torch.bfloat16
    record(ok)
    print(f"  {PASS if ok else FAIL} 默认输出 BF16: {z_default.dtype}")

    # 指定 FP16
    z_fp16 = decode_from_hifloat8(y, torch.float16)
    ok = z_fp16.dtype == torch.float16
    record(ok)
    print(f"  {PASS if ok else FAIL} 指定 FP16: {z_fp16.dtype}")

    # 指定 BF16
    z_bf16 = decode_from_hifloat8(y, torch.bfloat16)
    ok = z_bf16.dtype == torch.bfloat16
    record(ok)
    print(f"  {PASS if ok else FAIL} 指定 BF16: {z_bf16.dtype}")

    # 非 uint8 输入报错
    try:
        y_fp32 = torch.randn(1000, dtype=torch.float32, device='npu')
        decode_from_hifloat8(y_fp32)
        record(False)
        print(f"  {FAIL} 非 uint8 输入应报错但未报错")
    except (ValueError, RuntimeError):
        record(True)
        print(f"  {PASS} 非 uint8 输入正确报错")


# ── 测试 3: 编解码往返（基础值）───────────────────────────────────────────
def test_roundtrip_basic():
    print("\n[3] 编解码往返测试（基础值）")
    print("-" * 60)

    x_orig = torch.tensor([1.0, -1.0, 0.5, 2.0, 0.0], dtype=torch.float16, device='npu')
    y_encoded = encode_to_hifloat8(x_orig)
    z_decoded = decode_from_hifloat8(y_encoded, torch.float16)

    match_count = (x_orig == z_decoded).sum().item()
    ok = match_count == x_orig.numel()
    record(ok)
    print(f"  原始: {x_orig.cpu().tolist()}")
    print(f"  编码: {y_encoded.cpu().tolist()}")
    print(f"  解码: {z_decoded.cpu().tolist()}")
    print(f"  {PASS if ok else FAIL} 精确匹配率: {match_count}/{x_orig.numel()}")


# ── 测试 4: 编解码往返（随机数据）─────────────────────────────────────────
def test_roundtrip_random(dtype_name, dtype, num_elements=10000):
    print(f"\n[4] 编解码往返测试（随机 {dtype_name}，{num_elements} 元素）")
    print("-" * 60)

    x_orig = torch.randn(num_elements, dtype=dtype, device='npu')
    y_encoded = encode_to_hifloat8(x_orig)
    z_decoded = decode_from_hifloat8(y_encoded, dtype)

    x_cpu = x_orig.float().cpu()
    z_cpu = z_decoded.float().cpu()

    abs_diff = torch.abs(x_cpu - z_cpu)
    max_abs_diff = abs_diff.max().item()
    mean_abs_diff = abs_diff.mean().item()

    rel_diff = abs_diff / (torch.abs(x_cpu) + 1e-8)
    nonzero_mask = torch.abs(x_cpu) > 1e-4
    max_rel_diff_nonzero = rel_diff[nonzero_mask].max().item() if nonzero_mask.any() else 0.0

    ok = max_rel_diff_nonzero < 0.25
    record(ok)

    print(f"  最大绝对误差: {max_abs_diff:.6f}")
    print(f"  平均绝对误差: {mean_abs_diff:.6f}")
    print(f"  最大相对误差（排除极小值）: {max_rel_diff_nonzero:.6f}")
    print(f"  {PASS if ok else FAIL} 相对误差在 HiFloat8 精度范围内 (< 25%)")


# ── 测试 5: 边界值 ──────────────────────────────────────────────────────────
def test_boundary_values():
    print("\n[5] 边界值测试（FP16）")
    print("-" * 60)

    test_values = [
        0.0, -0.0,
        1.0, -1.0,
        0.5, -0.5,
        2.0, -2.0,
        0.25, 0.125,
        3.0, -3.0,
        0.0625, 0.03125,
        0.01, 0.001,
        10.0, 100.0,
    ]

    x_orig = torch.tensor(test_values, dtype=torch.float16, device='npu')
    y_encoded = encode_to_hifloat8(x_orig)
    z_decoded = decode_from_hifloat8(y_encoded, torch.float16)

    x_cpu = x_orig.float().cpu()
    z_cpu = z_decoded.float().cpu()
    abs_diff = torch.abs(x_cpu - z_cpu)

    n_ok = 0
    for i, val in enumerate(test_values):
        diff = abs_diff[i].item()
        is_ok = diff < 0.001
        if is_ok:
            n_ok += 1
        status = "OK" if is_ok else "WARN"
        print(f"  {val:8.4f} → {z_cpu[i].item():8.4f}, 误差={diff:.6f} {status}")

    ok = n_ok >= len(test_values) * 0.9
    record(ok)
    print(f"  {PASS if ok else WARN} 边界值精度: {n_ok}/{len(test_values)}")


# ── 测试 6: 特殊值（±0、±Inf、NaN）─────────────────────────────────────────
#
# HiFloat8 规范专用编码：
#   +0   → 0x00
#   -0   → 0x00（HiFloat8 不保留零的符号位）
#   +Inf → 0x6F
#   -Inf → 0xEF
#   NaN  → 0x80
#
# 校验三件事：
#   1. encode 输出严格等于规范编码
#   2. decode 后语义还原（±Inf 仍为 isinf、NaN 仍为 isnan）
#   3. FP16 与 BF16 在同一组特殊值上的编码必须完全一致

EXPECTED_ENCODE = {
    '+0': 0x00,
    '-0': 0x00,
    '+Inf': 0x6F,
    '-Inf': 0xEF,
    'NaN': 0x80,
}


def _check_decoded(name, decoded_val):
    """校验 decode 出的浮点值是否符合规范语义；返回 (ok, 描述)"""
    if name in ('+0', '-0'):
        return (decoded_val == 0.0, "应为 0.0")
    if name == '+Inf':
        return (decoded_val == float('inf'), "应为 +Inf")
    if name == '-Inf':
        return (decoded_val == float('-inf'), "应为 -Inf")
    if name == 'NaN':
        return (decoded_val != decoded_val, "应为 NaN")
    return (False, "未知")


def _test_special_one_dtype(dtype_label, dtype):
    test_names = ['+0', '-0', '+Inf', '-Inf', 'NaN']
    test_values = [0.0, -0.0, float('inf'), float('-inf'), float('nan')]

    x_orig = torch.tensor(test_values, dtype=dtype, device='npu')
    y_encoded = encode_to_hifloat8(x_orig)
    z_decoded = decode_from_hifloat8(y_encoded, dtype)

    enc_cpu = y_encoded.cpu()
    dec_cpu = z_decoded.float().cpu()

    n_ok = 0
    print(f"  [{dtype_label}] {'name':>5}  {'enc':>5}  {'expect':>6}  {'dec':>10}  status")
    for i, name in enumerate(test_names):
        enc = enc_cpu[i].item()
        exp = EXPECTED_ENCODE[name]
        dec = dec_cpu[i].item()

        enc_ok = (enc == exp)
        dec_ok, dec_msg = _check_decoded(name, dec)
        ok = enc_ok and dec_ok

        if ok:
            n_ok += 1
            status = "OK"
        else:
            parts = []
            if not enc_ok:
                parts.append(f"编码错(应为 0x{exp:02x})")
            if not dec_ok:
                parts.append(f"解码错({dec_msg})")
            status = "FAIL: " + ", ".join(parts)
        print(f"  [{dtype_label}] {name:>5}   0x{enc:02x}    0x{exp:02x}   {dec:>10}  {status}")

    return n_ok == len(test_names), enc_cpu


def test_special_values():
    print("\n[6] 特殊值测试（FP16 / BF16，按 HiFloat8 规范严格校验）")
    print("-" * 70)

    fp16_ok, fp16_enc = _test_special_one_dtype("FP16", torch.float16)
    bf16_ok, bf16_enc = _test_special_one_dtype("BF16", torch.bfloat16)

    cross_ok = bool((fp16_enc == bf16_enc).all().item())
    print(f"  FP16↔BF16 编码交叉一致: {'✓' if cross_ok else '✗'}")
    if not cross_ok:
        for i, n in enumerate(['+0', '-0', '+Inf', '-Inf', 'NaN']):
            if fp16_enc[i].item() != bf16_enc[i].item():
                print(f"    {n}: FP16=0x{fp16_enc[i].item():02x}  BF16=0x{bf16_enc[i].item():02x}")

    ok = fp16_ok and bf16_ok and cross_ok
    record(ok)
    print(f"  {PASS if ok else FAIL} 特殊值处理符合 HiFloat8 规范")


# ── 测试 7: 非规格化数（Denormal）─────────────────────────────────────────

def test_denormal_values():
    print("\n[7] 非规格化数测试（Denormal/Subnormal）")
    print("-" * 60)

    print("  说明: HiFloat8 可能对极小值应用 Flush-to-Zero (FTZ)")

    test_cases = [
        ('FP16 max denormal', torch.float16, 6.1035e-5),
        ('FP16 min denormal', torch.float16, 5.96e-8),
        ('BF16 max denormal', torch.bfloat16, 1.17549435e-38),
        ('BF16 min denormal', torch.bfloat16, 9.2e-41),
    ]

    n_ok = 0
    for name, dtype, value in test_cases:
        x_orig = torch.tensor([value], dtype=dtype, device='npu')
        y_encoded = encode_to_hifloat8(x_orig)
        z_decoded = decode_from_hifloat8(y_encoded, dtype)

        decoded_val = z_decoded[0].item()
        encoded_byte = y_encoded.cpu().item()

        if decoded_val == 0.0:
            status = "FTZ (预期行为)"
            is_ok = True
        else:
            rel_diff = abs(value - decoded_val) / abs(value)
            is_ok = rel_diff < 0.2
            status = "OK" if is_ok else "WARN"

        if is_ok:
            n_ok += 1
        print(f"  {name:20s}: 编码=0x{encoded_byte:02x}, 解码={decoded_val:.6e} {status}")

    ok = n_ok == len(test_cases)
    record(ok)
    print(f"  {PASS if ok else WARN} 非规格化数处理: {n_ok}/{len(test_cases)}")


# ── 测试 8: 全 256 个 HiFloat8 值 decode ─────────────────────────────────────

def test_full_256_decode():
    print("\n[8] 全 256 个 HiFloat8 值 decode 完备性")
    print("-" * 60)

    all_hif8 = torch.arange(256, dtype=torch.int32, device='npu').to(torch.uint8)

    z_fp16 = decode_from_hifloat8(all_hif8, torch.float16)
    z_bf16 = decode_from_hifloat8(all_hif8, torch.bfloat16)

    fp16_count = z_fp16.numel()
    fp16_valid_count = (~z_fp16.isnan()).sum().item()
    fp16_valid = fp16_valid_count == fp16_count - 1  # 0x80 是 NaN 编码
    record(fp16_valid)
    print(f"  {PASS if fp16_valid else FAIL} FP16 decode: {fp16_valid_count}/{fp16_count} 有效（0x80 为 NaN 编码）")

    bf16_count = z_bf16.numel()
    bf16_valid_count = (~z_bf16.isnan()).sum().item()
    bf16_valid = bf16_valid_count == bf16_count - 1
    record(bf16_valid)
    print(f"  {PASS if bf16_valid else FAIL} BF16 decode: {bf16_valid_count}/{bf16_count} 有效（0x80 为 NaN 编码）")

    print(f"  示例解码结果:")
    for i in [0x00, 0x01, 0x7F, 0x80, 0xFF]:
        fp16_val = z_fp16[i].item()
        bf16_val = z_bf16[i].item()
        fp16_str = f"{fp16_val:.4f}" if not (fp16_val != fp16_val) else "NaN"
        bf16_str = f"{bf16_val:.4f}" if not (bf16_val != bf16_val) else "NaN"
        print(f"    0x{i:02x} → FP16={fp16_str}, BF16={bf16_str}")


# ── 主程序 ──────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("HiFloat8 Cast 精度测试（仅 NPU）")
    print("=" * 60)

    test_type_inference()
    test_decode_output_type()
    test_roundtrip_basic()
    test_roundtrip_random("FP16", torch.float16, 10000)
    test_roundtrip_random("BF16", torch.bfloat16, 10000)
    test_boundary_values()
    test_special_values()
    test_denormal_values()
    test_full_256_decode()

    print()
    print("=" * 60)
    print(f"测试汇总: {total_pass}/{total_cases} 通过")
    if total_pass == total_cases:
        print("结论: 全部测试通过 ✓")
    else:
        print(f"结论: {total_cases - total_pass} 项未通过，请检查上方详情")
    print("=" * 60)

    return total_pass == total_cases


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
