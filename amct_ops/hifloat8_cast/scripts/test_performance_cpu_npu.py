# -*- coding: UTF-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# Apache License, 2.0
"""
HiFloat8 Cast 性能对比：NPU vs CPU

测试覆盖：
  1. Encode 性能对比（FP16/BF16 → HiFloat8）
  2. Decode 性能对比（HiFloat8 → FP16/BF16）
  3. NPU 加速比统计

运行：
    LD_PRELOAD=/usr/lib/$(uname -m)-linux-gnu/libgomp.so.1 python scripts/test_performance_cpu_npu.py

依赖：
    - NPU 版本：amct_ops.hifloat8_cast
    - CPU 版本：amct_pytorch/experimental/hifloat8/hifloat8_cast.so
"""

import sys
import os
import time
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CPU_PKG_DIR = os.path.join(SCRIPT_DIR, '..', '..', '..', 'amct_pytorch', 'experimental', 'hifloat8')

# 加载 NPU 版本
npu_available = False
try:
    import torch_npu
    from amct_ops.hifloat8_cast import encode_to_hifloat8 as npu_encode, decode_from_hifloat8 as npu_decode
    torch.npu.set_device(0)
    npu_available = True
    print("[OK] NPU 版本加载成功")
except Exception as e:
    print(f"[SKIP] NPU: {e}")

# 加载 CPU 版本
cpu_available = False
try:
    sys.path.insert(0, CPU_PKG_DIR)
    import hifloat8_cast as cpu_hif8
    cpu_available = True
    print("[OK] CPU 版本加载成功")
except Exception as e:
    print(f"[SKIP] CPU: {e}")
    print("       提示: 需先构建 CPU 参考实现")

if not npu_available and not cpu_available:
    print("两个版本都不可用，退出")
    sys.exit(1)


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
]


# ── 工具函数 ──────────────────────────────────────────────────────────────

def measure_npu(func):
    for _ in range(WARMUP):
        func()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(ITERS):
        func()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / ITERS * 1000


def measure_cpu(func):
    for _ in range(WARMUP):
        func()
    t0 = time.perf_counter()
    for _ in range(ITERS):
        func()
    return (time.perf_counter() - t0) / ITERS * 1000


def throughput(n_elem, in_bytes, out_bytes, ms):
    return (n_elem * (in_bytes + out_bytes)) / ms / 1000


# ── 通用对比表 ────────────────────────────────────────────────────────────

def _print_header(title):
    print(f"\n[{title}]")
    print("-" * 90)
    hdr = f"{'大小':>8}"
    if npu_available:
        hdr += f"  {'NPU(ms)':>10}  {'NPU MB/s':>11}"
    if cpu_available:
        hdr += f"  {'CPU(ms)':>10}  {'CPU MB/s':>11}"
    if npu_available and cpu_available:
        hdr += f"  {'加速比':>8}"
    print(hdr)
    print("-" * 90)


def run_compare(title, in_bytes, out_bytes, make_npu_input, npu_call, make_cpu_input, cpu_call):
    """
    title       : 表头标题
    in_bytes    : 输入元素字节数（用于吞吐量计算）
    out_bytes   : 输出元素字节数
    make_npu_input(size) -> tensor on NPU；预先创建，避免计入计时
    npu_call(x)          -> 调用一次 NPU kernel
    make_cpu_input(size) -> tensor on CPU
    cpu_call(x)          -> 调用一次 CPU kernel
    """
    _print_header(title)

    for size, label in SIZES:
        row = f"{label:>8}"
        npu_ms = cpu_ms = None

        if npu_available:
            x_npu = make_npu_input(size)
            npu_ms = measure_npu(lambda: npu_call(x_npu))
            row += f"  {npu_ms:>10.3f}  {throughput(size, in_bytes, out_bytes, npu_ms):>11.1f}"

        if cpu_available:
            x_cpu = make_cpu_input(size)
            cpu_ms = measure_cpu(lambda: cpu_call(x_cpu))
            row += f"  {cpu_ms:>10.3f}  {throughput(size, in_bytes, out_bytes, cpu_ms):>11.1f}"

        if npu_ms and cpu_ms:
            row += f"  {cpu_ms / npu_ms:>7.2f}x"

        print(row)


# ── 测试用例 ──────────────────────────────────────────────────────────────

def test_encode(dtype_label, dtype):
    run_compare(
        title=f"{dtype_label} Encode 性能对比",
        in_bytes=2, out_bytes=1,
        make_npu_input=lambda n: torch.randn(n, dtype=dtype, device='npu'),
        npu_call=lambda x: npu_encode(x),
        make_cpu_input=lambda n: torch.randn(n, dtype=dtype),
        cpu_call=lambda x: cpu_hif8.float_to_hifloat8(x),
    )


def test_decode(dtype_label, dtype):
    # CPU 仅提供 BF16 decode 接口，FP16/BF16 共用
    run_compare(
        title=f"{dtype_label} Decode 性能对比",
        in_bytes=1, out_bytes=2,
        make_npu_input=lambda n: torch.randint(0, 256, (n,), dtype=torch.uint8, device='npu'),
        npu_call=lambda y: npu_decode(y, dtype),
        make_cpu_input=lambda n: torch.randint(0, 256, (n,), dtype=torch.uint8),
        cpu_call=lambda y: cpu_hif8.hifloat8_to_bfloat16(y),
    )


# ── 主程序 ──────────────────────────────────────────────────────────────────

def main():
    print("=" * 90)
    print("HiFloat8 Cast 性能对比: NPU vs CPU")
    print(f"预热迭代: {WARMUP}  测试迭代: {ITERS}")
    print("=" * 90)

    test_encode("FP16", torch.float16)
    test_decode("FP16", torch.float16)
    test_encode("BF16", torch.bfloat16)
    test_decode("BF16", torch.bfloat16)

    print()
    print("=" * 90)
    print("说明:")
    print("  - 吞吐量 (MB/s) = (输入字节数 + 输出字节数) / 耗时")
    print("  - 加速比 = CPU 时间 / NPU 时间（>1 表示 NPU 更快）")
    print("  - CPU decode 使用 BF16 接口（HiFloat8 无 FP16 decode 接口）")
    print("=" * 90)


if __name__ == "__main__":
    main()
