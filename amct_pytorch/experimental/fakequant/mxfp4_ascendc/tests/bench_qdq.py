"""Benchmark Ascend-C MXFP4 QDQ kernel vs torch software fallback.

Both implementations preserve the exact reference semantics used by the patched
vLLM `_quant_dequant_mxfp4`:
  * pad last dim to multiple of 32
  * per-block (32 elem) absolute-max scale, divided by scale_factor=6.0
  * E8M0 scale = 2 ** round(log2(scale_raw))
  * 8-level FP4-e2m1 codebook on |x/scale|
  * dq = sign(x) * code * scale
"""

import argparse
import importlib
import os
import sys
import time

import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PYTHON = os.path.join(_ROOT, "python")
if _PYTHON not in sys.path:
    sys.path.insert(0, _PYTHON)

importlib.import_module("torch_npu")  # registers npu device on import


def torch_qdq(x: torch.Tensor) -> torch.Tensor:
    """Software fallback identical to the patched vllm path."""
    x_fp = x.to(torch.float32)
    block = 32
    last = x_fp.shape[-1]
    pad = (block - (last % block)) % block
    if pad:
        x_fp = torch.nn.functional.pad(x_fp, (0, pad))
    x_view = x_fp.view(*x_fp.shape[:-1], -1, block)
    max_abs = x_view.abs().amax(dim=-1, keepdim=True)
    raw_scale = torch.clamp(max_abs / 6.0, min=2.0**-30)
    log2_scale = torch.round(torch.log2(raw_scale))
    scale = torch.exp2(log2_scale)
    y = x_view / scale
    y_abs = y.abs()
    q_abs = torch.zeros_like(y_abs)
    q_abs = torch.where(y_abs >= 0.25, 0.5, q_abs)
    q_abs = torch.where(y_abs >= 0.75, 1.0, q_abs)
    q_abs = torch.where(y_abs >= 1.25, 1.5, q_abs)
    q_abs = torch.where(y_abs >= 1.75, 2.0, q_abs)
    q_abs = torch.where(y_abs >= 2.50, 3.0, q_abs)
    q_abs = torch.where(y_abs >= 3.50, 4.0, q_abs)
    q_abs = torch.where(y_abs >= 5.00, 6.0, q_abs)
    q = torch.sign(y) * q_abs
    dq = (q * scale).reshape(*x_fp.shape[:-1], -1)
    if pad:
        dq = dq[..., :last]
    return dq.to(x.dtype)


def ascendc_qdq(x: torch.Tensor, kernel) -> torch.Tensor:
    """Ascend-C path mirroring the patched vllm _npu_qdq_kernel branch."""
    x_fp = x.to(torch.float32)
    block = 32
    last = x_fp.shape[-1]
    pad = (block - (last % block)) % block
    if pad:
        x_fp = torch.nn.functional.pad(x_fp, (0, pad))
    x_2d = x_fp.reshape(-1, x_fp.shape[-1]).contiguous()
    dq = kernel(x_2d).reshape(*x_fp.shape[:-1], x_fp.shape[-1])
    if pad:
        dq = dq[..., :last]
    return dq.to(x.dtype)


def time_op(fn, iters: int) -> float:
    """Returns mean us/iter across `iters` runs after warmup, with sync."""
    for _ in range(5):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) * 1e6 / iters


def correctness(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float]:
    af = a.float()
    bf = b.float()
    abs_err = (af - bf).abs()
    return abs_err.max().item(), abs_err.mean().item()


def fmt(us: float) -> str:
    if us >= 1000:
        return f"{us / 1000:8.3f} ms"
    return f"{us:8.2f} us"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    args = parser.parse_args()

    dtype = getattr(torch, args.dtype)
    device = torch.device(args.device)
    if device.type != "npu":
        raise ValueError(f"--device must be an NPU device, got {args.device!r}")
    # Match current NPU (and thus getCurrentNPUStream / mxfp4 self-test)
    # to --device; hardcoding set_device(0) breaks multi-NPU runs.
    torch.npu.set_device(device.index if device.index is not None else 0)

    from mxfp4 import quant_dequant_mxfp4 as kernel

    hidden_size = 3072
    intermediate_size = 1536
    shapes = [
        ("decode    bs=1   x H", (1, hidden_size)),
        ("decode    bs=32  x H", (32, hidden_size)),
        ("decode    bs=64  x H", (64, hidden_size)),
        ("prefill   bs=1024 x H", (1024, hidden_size)),
        ("prefill   bs=4096 x H", (4096, hidden_size)),
        ("prefill   bs=8192 x H", (8192, hidden_size)),
        ("prefill   bs=32768 x H", (32768, hidden_size)),
        ("MoE expert 256 x I (n_tok=256)", (256, intermediate_size)),
        ("MoE expert 1024 x I", (1024, intermediate_size)),
        ("MoE expert 4096 x I", (4096, intermediate_size)),
        ("MoE expert 16384 x I", (16384, intermediate_size)),
        ("MoE 3D    8 x 4096 x I", (8, 4096, intermediate_size)),
    ]

    print(f"\n  device={device}  dtype={dtype}  iters={args.iters}")
    print(f"  H={hidden_size} (hidden_size)  I={intermediate_size} (intermediate_size)")
    print()
    print(
        f"  {'shape':40s} {'numel':>12s}  {'torch':>12s}  "
        f"{'ascendc':>12s}  {'speedup':>8s}  {'max_abs_err':>12s}"
    )
    print("  " + "-" * 110)

    for label, shape in shapes:
        torch.manual_seed(0)
        x = torch.randn(shape, dtype=dtype, device=device)

        y_torch = torch_qdq(x)
        y_ac = ascendc_qdq(x, kernel)
        torch.npu.synchronize()
        max_err, mean_err = correctness(y_torch, y_ac)

        t_torch = time_op(lambda: torch_qdq(x), args.iters)
        t_ac = time_op(lambda: ascendc_qdq(x, kernel), args.iters)
        speedup = t_torch / t_ac

        print(
            f"  {label:40s} {x.numel():12d}  {fmt(t_torch):>12s}  "
            f"{fmt(t_ac):>12s}  {speedup:7.2f}x  {max_err:12.3e}"
        )

    print()


if __name__ == "__main__":
    main()
