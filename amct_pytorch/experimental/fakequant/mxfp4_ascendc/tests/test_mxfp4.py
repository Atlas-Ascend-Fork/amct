#!/usr/bin/env python3
"""
Correctness + performance test for the MXFP4 Ascend-C kernel.

Compares against the pure-PyTorch reference in ../reference/mxfp4_ref.py.
"""

from __future__ import annotations

import importlib
import os
import sys
import time

import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PYTHON = os.path.join(_ROOT, "python")
_REF = os.path.join(_ROOT, "reference")
for _p in (_PYTHON, _REF):
    if _p not in sys.path:
        sys.path.insert(0, _p)

importlib.import_module("torch_npu")  # registers NPU device
ref_qdq = importlib.import_module("mxfp4_ref").quant_dequant_mxfp4
quant_dequant_mxfp4_npu = importlib.import_module("mxfp4").quant_dequant_mxfp4


def _banner(msg: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {msg}")
    print(f"{'=' * 60}")


def test_correctness(
    shapes: list[tuple[int, ...]] | None = None,
    seed: int = 42,
    atol: float = 1e-6,
) -> bool:
    if shapes is None:
        shapes = [
            (4, 128),
            (1, 32),
            (16, 1024),
            (64, 4096),
            # last_dim not a multiple of 32: catches per-row vs flatten-then-pad
            (2, 33),
            (3, 17),
            (4, 48, 33),
        ]
    _banner("Correctness")
    torch.manual_seed(seed)
    all_ok = True

    for shape in shapes:
        x_cpu = torch.randn(shape, dtype=torch.float32) * 3.0
        ref_cpu = ref_qdq(x_cpu)

        x_npu = x_cpu.clone().npu()
        out_npu = quant_dequant_mxfp4_npu(x_npu)
        out_cpu = out_npu.cpu()

        max_err = (out_cpu - ref_cpu).abs().max().item()
        ok = max_err <= atol
        tag = "PASS" if ok else "FAIL"
        print(f"  {tag}  shape={str(shape):>16s}  max_err={max_err:.2e}")
        if not ok:
            all_ok = False
            print(f"         ref[:4] = {ref_cpu.reshape(-1)[:4].tolist()}")
            print(f"         npu[:4] = {out_cpu.reshape(-1)[:4].tolist()}")

    return all_ok


def _bench_one(fn, x, warmup: int, iters: int) -> float:
    """Return average ms/call after warmup."""
    for _ in range(warmup):
        fn(x)
    if x.is_npu:
        torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn(x)
    if x.is_npu:
        torch.npu.synchronize()
    return (time.perf_counter() - t0) / iters * 1000.0


def benchmark(
    shapes: list[tuple[int, ...]] | None = None,
    warmup: int = 20,
    iters: int = 200,
) -> None:
    if shapes is None:
        shapes = [(1, 32), (4, 128), (16, 1024), (64, 4096), (256, 4096), (1024, 4096)]

    _banner(f"Benchmark  warmup={warmup}  iters={iters}")
    header = f"  {'shape':>14s}  {'elems':>8s}  {'torch_npu':>10s}  {'ascendc':>10s}  {'speedup':>8s}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for shape in shapes:
        x_npu = torch.randn(shape, dtype=torch.float32, device="npu:0") * 3.0

        ms_tnpu = _bench_one(ref_qdq, x_npu, warmup, iters)
        ms_ac = _bench_one(quant_dequant_mxfp4_npu, x_npu, warmup, iters)
        sp = ms_tnpu / ms_ac if ms_ac > 0 else float("inf")

        print(
            f"  {str(shape):>14s}  {x_npu.numel():>8d}  {ms_tnpu:>8.3f}ms  {ms_ac:>8.3f}ms  {sp:>7.2f}x"
        )


if __name__ == "__main__":
    ok = test_correctness()
    benchmark()
    sys.exit(0 if ok else 1)
