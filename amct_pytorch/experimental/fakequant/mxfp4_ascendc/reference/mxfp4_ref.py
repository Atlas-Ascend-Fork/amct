import os

import torch
import torch.nn.functional as F


def quant_dequant_mxfp4(
    x: torch.Tensor,
    scale_factor: float | None = None,
    block_size: int = 32,
) -> torch.Tensor:
    """Apply MXFP4 quantize + dequantize to a tensor.

    This mirrors the software reference path used in the vLLM patch:
    - per-element FP4 e2m1 quantization
    - per-block E8M0 power-of-two scales

    If ``scale_factor`` is not provided, it is read from
    ``VLLM_MXFP4_ACT_QDQ_SCALE_FACTOR`` and defaults to ``6.0``.
    """

    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")

    if x.ndim == 0:
        raise ValueError("x must have at least 1 dimension")

    if scale_factor is None:
        scale_factor = float(os.getenv("VLLM_MXFP4_ACT_QDQ_SCALE_FACTOR", "6.0"))

    if scale_factor <= 0:
        raise ValueError(f"scale_factor must be positive, got {scale_factor}")

    x_fp = x.to(torch.float32)
    last_dim = x_fp.shape[-1]
    pad = (block_size - (last_dim % block_size)) % block_size
    if pad:
        x_fp = F.pad(x_fp, (0, pad))

    x_view = x_fp.view(*x_fp.shape[:-1], -1, block_size)

    # Per-block E8M0 scale: nearest power-of-two around max_abs / scale_factor.
    max_abs = x_view.abs().amax(dim=-1, keepdim=True)
    raw_scale = torch.clamp(max_abs / scale_factor, min=2.0**-30)
    log2_scale = torch.round(torch.log2(raw_scale))
    scale = torch.exp2(log2_scale)

    y = x_view / scale
    y_abs = y.abs()

    # FP4 e2m1 positive codebook (nearest via midpoints): 0, 0.5, 1, 1.5, 2, 3, 4, 6
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
        dq = dq[..., :last_dim]

    return dq.to(x.dtype)


def _print_metric_group(label: str, recon: torch.Tensor, ref: torch.Tensor) -> None:
    diff = recon - ref
    mse = (diff * diff).mean().item()
    mae = diff.abs().mean().item()
    max_abs_err = diff.abs().max().item()
    rel_l2 = (torch.norm(diff) / torch.norm(ref)).item()

    print(f"{label}:")
    print(f"  mse: {mse:.8f}")
    print(f"  mae: {mae:.8f}")
    print(f"  max_abs_err: {max_abs_err:.8f}")
    print(f"  relative_l2: {rel_l2:.8f}")


def demo_quant_dequant_loss(
    shape: tuple[int, ...] = (4, 128),
    scale: float = 3.0,
    seed: int = 0,
    dtype: torch.dtype = torch.float32,
) -> None:
    """Run a small random-tensor demo: two QDQ passes and per-stage metrics."""

    torch.manual_seed(seed)
    x = torch.randn(shape, dtype=dtype) * scale
    dq1 = quant_dequant_mxfp4(x)
    dq2 = quant_dequant_mxfp4(dq1)

    print(f"input shape: {tuple(x.shape)}, dtype: {x.dtype}")
    print(
        "qdq config: "
        f"scale_factor={os.getenv('VLLM_MXFP4_ACT_QDQ_SCALE_FACTOR', '6.0')}, "
        "block_size=32"
    )
    _print_metric_group("first qdq (x -> dq1)", dq1, x)
    _print_metric_group("second qdq (dq1 -> dq2)", dq2, dq1)
    _print_metric_group("total drift (x -> dq2)", dq2, x)

    print(f"dq1 == dq2 (elementwise): {torch.equal(dq1, dq2)}")
    flat = x.reshape(-1)
    print("sample input[:8]:", flat[:8])
    print("sample dq1[:8]:", dq1.reshape(-1)[:8])
    print("sample dq2[:8]:", dq2.reshape(-1)[:8])


if __name__ == "__main__":
    demo_quant_dequant_loss()
