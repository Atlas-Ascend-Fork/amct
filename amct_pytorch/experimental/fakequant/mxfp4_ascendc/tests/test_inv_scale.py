#!/usr/bin/env python3
"""Verify the new inv_scale_factor_scale parameter of the NPU MXFP4 kernel.

Relationship being checked:
    kernel raw_scale = max_abs * (1/6.0) * s
                     = max_abs / (6.0 / s)
so the NPU kernel with inv_scale_factor_scale=s must match the pure-PyTorch
reference quant_dequant_mxfp4(..., scale_factor=6.0/s).
"""

from __future__ import annotations

import importlib
import os
import sys

import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PYTHON = os.path.join(_ROOT, "python")
_REF = os.path.join(_ROOT, "reference")
for _p in (_PYTHON, _REF):
    if _p not in sys.path:
        sys.path.insert(0, _p)

importlib.import_module("torch_npu")  # registers NPU device
ref_qdq = importlib.import_module("mxfp4_ref").quant_dequant_mxfp4
npu_qdq = importlib.import_module("mxfp4").quant_dequant_mxfp4


def _banner(msg: str) -> None:
    print(f"\n{'=' * 64}\n  {msg}\n{'=' * 64}")


def test_scales(
    shapes=((4, 128), (1, 32), (16, 1024), (64, 4096), (2, 33), (3, 17)),
    scales=(1.0, 2.0, 0.5, 1.5, 0.25),
    seed: int = 42,
    atol: float = 1e-6,
) -> bool:
    _banner("inv_scale_factor_scale correctness (NPU vs PyTorch reference)")
    torch.manual_seed(seed)
    all_ok = True

    for s in scales:
        ref_scale_factor = 6.0 / s
        print(
            f"\n-- inv_scale_factor_scale={s}  "
            f"(<=> reference scale_factor={ref_scale_factor:.4f}) --"
        )
        for shape in shapes:
            x_cpu = torch.randn(shape, dtype=torch.float32) * 3.0

            ref_cpu = ref_qdq(x_cpu, scale_factor=ref_scale_factor)

            x_npu = x_cpu.clone().npu()
            out_cpu = npu_qdq(x_npu, inv_scale_factor_scale=s).cpu()

            max_err = (out_cpu - ref_cpu).abs().max().item()
            ok = max_err <= atol
            all_ok &= ok
            print(
                f"   {'PASS' if ok else 'FAIL'}  shape={str(shape):>14s}  "
                f"max_err={max_err:.3e}"
            )
    return all_ok


def test_default_equivalence(seed: int = 0, atol: float = 1e-6) -> bool:
    """Default (no arg) must equal explicit s=1.0 and reference scale_factor=6."""
    _banner("default behaviour == inv_scale_factor_scale=1.0")
    torch.manual_seed(seed)
    x = torch.randn(16, 1024, dtype=torch.float32) * 3.0
    x_npu = x.npu()

    y_default = npu_qdq(x_npu).cpu()
    y_one = npu_qdq(x_npu, inv_scale_factor_scale=1.0).cpu()
    y_ref = ref_qdq(x, scale_factor=6.0)

    e_dvo = (y_default - y_one).abs().max().item()
    e_dvr = (y_default - y_ref).abs().max().item()
    ok = e_dvo <= atol and e_dvr <= atol
    print(f"   default vs s=1.0     max_err={e_dvo:.3e}")
    print(f"   default vs ref(6.0)  max_err={e_dvr:.3e}")
    print(f"   {'PASS' if ok else 'FAIL'}")
    return ok


def test_scale_effect(seed: int = 7) -> bool:
    """Sanity: different multipliers should actually change the output."""
    _banner("scaling actually changes the output")
    torch.manual_seed(seed)
    x_npu = (torch.randn(8, 256, dtype=torch.float32) * 3.0).npu()

    y1 = npu_qdq(x_npu, inv_scale_factor_scale=1.0).cpu()
    y2 = npu_qdq(x_npu, inv_scale_factor_scale=2.0).cpu()
    diff = (y1 - y2).abs().max().item()
    ok = diff > 0.0
    print(
        f"   max|y(s=1.0) - y(s=2.0)| = {diff:.4f}  "
        f"({'differs' if ok else 'IDENTICAL!'})"
    )
    print(f"   {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    r1 = test_default_equivalence()
    r2 = test_scales()
    r3 = test_scale_effect()
    _banner("SUMMARY")
    print(f"   default equivalence : {'PASS' if r1 else 'FAIL'}")
    print(f"   scale correctness   : {'PASS' if r2 else 'FAIL'}")
    print(f"   scale effect        : {'PASS' if r3 else 'FAIL'}")
    sys.exit(0 if (r1 and r2 and r3) else 1)
