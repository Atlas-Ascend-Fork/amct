# -*- coding: UTF-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# Apache License, 2.0
"""
HiFloat8 Cast 性能测试（仅 NPU）

测试覆盖：
  1. Encode 性能（FP16/BF16 → HiFloat8）
  2. Decode 性能（HiFloat8 → FP16/BF16）
  3. 编解码往返性能
  4. 多种数据规模（1K ~ 16M）

运行：
    python scripts/test_performance.py
"""

import time
import torch
import torch_npu

torch.npu.set_device(0)

from amct_ops.hifloat8_cast import encode_to_hifloat8, decode_from_hifloat8


# ── 配置 ──────────────────────────────────────────────────────────────────

WARMUP = 10
ITERS = 100

SIZES = [
    (1024,        "1K"),
    (4096,        "4K"),
    (16384,       "16K"),
    (65536,       "64K"),
    (262144,      "256K"),
    (1048576,     "1M"),
    (4194304,     "4M"),
    (16777216,    "16M"),
]


# ── 工具函数 ──────────────────────────────────────────────────────────────

def measure(func, warmup=WARMUP, repeat=ITERS):
    """测量函数执行时间（ms）"""
    for _ in range(warmup):
        func()
    torch.npu.synchronize()

    start = time.perf_counter()
    for _ in range(repeat):
        func()
    torch.npu.synchronize()
    elapsed = time.perf_counter() - start

    return elapsed / repeat * 1000


def throughput_bytes(n_elem, in_bytes, out_bytes, ms):
    """计算吞吐量 (MB/s)"""
    return (n_elem * (in_bytes + out_bytes)) / ms / 1000


def throughput_elems(n_elem, ms):
    """计算吞吐量 (elem/s)"""
    return n_elem / ms * 1000


# ── Encode 性能测试 ────────────────────────────────────────────────────────

def test_encode_perf():
    print("\n[1] Encode 性能测试")
    print("-" * 80)
    print(f"{'大小':>8}  {'FP16(ms)':>10}  {'FP16 MB/s':>12}  {'BF16(ms)':>10}  {'BF16 MB/s':>12}")
    print("-" * 80)

    for size, label in SIZES:
        x_fp16 = torch.randn(size, dtype=torch.float16, device='npu')
        x_bf16 = torch.randn(size, dtype=torch.bfloat16, device='npu')

        fp16_ms = measure(lambda: encode_to_hifloat8(x_fp16))
        fp16_tp = throughput_bytes(size, 2, 1, fp16_ms)

        bf16_ms = measure(lambda: encode_to_hifloat8(x_bf16))
        bf16_tp = throughput_bytes(size, 2, 1, bf16_ms)

        print(f"{label:>8}  {fp16_ms:>10.3f}  {fp16_tp:>12.1f}  {bf16_ms:>10.3f}  {bf16_tp:>12.1f}")


# ── Decode 性能测试 ────────────────────────────────────────────────────────

def test_decode_perf():
    print("\n[2] Decode 性能测试")
    print("-" * 80)
    print(f"{'大小':>8}  {'FP16(ms)':>10}  {'FP16 MB/s':>12}  {'BF16(ms)':>10}  {'BF16 MB/s':>12}")
    print("-" * 80)

    for size, label in SIZES:
        y_hif8 = torch.randint(0, 256, (size,), dtype=torch.uint8, device='npu')

        fp16_ms = measure(lambda: decode_from_hifloat8(y_hif8, torch.float16))
        fp16_tp = throughput_bytes(size, 1, 2, fp16_ms)

        bf16_ms = measure(lambda: decode_from_hifloat8(y_hif8, torch.bfloat16))
        bf16_tp = throughput_bytes(size, 1, 2, bf16_ms)

        print(f"{label:>8}  {fp16_ms:>10.3f}  {fp16_tp:>12.1f}  {bf16_ms:>10.3f}  {bf16_tp:>12.1f}")


# ── 编解码往返性能测试 ────────────────────────────────────────────────────

def test_roundtrip_perf():
    print("\n[3] 编解码往返性能测试")
    print("-" * 80)
    print(f"{'大小':>8}  {'FP16(ms)':>10}  {'FP16 MB/s':>12}  {'BF16(ms)':>10}  {'BF16 MB/s':>12}")
    print("-" * 80)

    for size, label in SIZES:
        x_fp16 = torch.randn(size, dtype=torch.float16, device='npu')
        x_bf16 = torch.randn(size, dtype=torch.bfloat16, device='npu')

        fp16_ms = measure(lambda: decode_from_hifloat8(encode_to_hifloat8(x_fp16), torch.float16))
        fp16_tp = throughput_bytes(size, 2, 2, fp16_ms)

        bf16_ms = measure(lambda: decode_from_hifloat8(encode_to_hifloat8(x_bf16), torch.bfloat16))
        bf16_tp = throughput_bytes(size, 2, 2, bf16_ms)

        print(f"{label:>8}  {fp16_ms:>10.3f}  {fp16_tp:>12.1f}  {bf16_ms:>10.3f}  {bf16_tp:>12.1f}")


# ── 主程序 ──────────────────────────────────────────────────────────────────

def main():
    print("=" * 80)
    print("HiFloat8 Cast 性能测试（仅 NPU）")
    print(f"预热迭代: {WARMUP}  测试迭代: {ITERS}")
    print("=" * 80)

    test_encode_perf()
    test_decode_perf()
    test_roundtrip_perf()

    print()
    print("=" * 80)
    print("说明:")
    print("  - 吞吐量 (MB/s) = (输入字节数 + 输出字节数) / 耗时")
    print("  - Encode: FP16/BF16 (2B) → HiFloat8 (1B)")
    print("  - Decode: HiFloat8 (1B) → FP16/BF16 (2B)")
    print("  - Roundtrip: FP16/BF16 (2B) → HiFloat8 (1B) → FP16/BF16 (2B)")
    print("=" * 80)


if __name__ == "__main__":
    main()