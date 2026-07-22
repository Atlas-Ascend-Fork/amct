"""
mxfp4_npu — MXFP4 quant-dequant accelerated by Ascend-C on 910B3.

Usage:
    from mxfp4_npu import quant_dequant_mxfp4

    x_npu = torch.randn(4, 128, device="npu:0")
    y_npu = quant_dequant_mxfp4(x_npu)

Requirements:
    - Ascend 910B3 + CANN 8.2.RC1
    - torch 2.6.0 + torch_npu 2.6.0.post4
    - Python 3.10 (aarch64)
"""

from __future__ import annotations

import ctypes
import importlib
import os
import sys

import torch
import torch.nn.functional as F

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
_LIB_DIR = os.path.join(_PKG_DIR, "_lib")
_EXT_LOADED = False
_ext = None

_SELF_TEST_DONE = False
_SELF_TEST_OK = False
_SELF_TEST_ERR: str | None = None


def _load_ext() -> bool:
    global _EXT_LOADED, _ext
    if _EXT_LOADED:
        return _ext is not None

    try:
        import torch_npu as _tnpu

        _tnpu_lib = os.path.join(
            os.path.dirname(_tnpu.__file__), "lib", "libtorch_npu.so"
        )
        if os.path.isfile(_tnpu_lib):
            ctypes.CDLL(_tnpu_lib, mode=ctypes.RTLD_GLOBAL)
    except Exception:
        pass

    kernel_lib = os.path.join(_LIB_DIR, "libascendc_kernels_npu.so")
    if os.path.isfile(kernel_lib):
        try:
            ctypes.CDLL(kernel_lib, mode=ctypes.RTLD_GLOBAL)
        except OSError:
            pass

    if _LIB_DIR not in sys.path:
        sys.path.insert(0, _LIB_DIR)

    try:
        _ext = importlib.import_module("mxfp4_ascendc_ext")
    except ImportError as e:
        print(f"[mxfp4_npu] Failed to load extension: {e}")

    _EXT_LOADED = True
    if _ext is not None:
        # Validates the kernel actually executes on this SoC. The auto-gen
        # host stub returns ret=0 even when registration silently fails
        # (e.g. SoC mismatch), which would otherwise propagate uninitialised
        # device memory as the "QDQ output" and corrupt every downstream call.
        _self_test_or_raise()
    return _ext is not None


def _self_test_or_raise() -> None:
    """Run a fixed-input MXFP4 QDQ on the kernel and verify the output.

    Input ones(32) is chosen because the FP4 e2m1 codebook + scale_factor
    derivation produces an analytically known result for scale=6.0:
        max_abs    = 1.0
        raw_scale  = 1.0 / scale_factor   (~0.1667)
        log2_round = -3
        scale      = 2^-3 = 0.125
        x / scale  = 8.0
        q_abs      = 6.0  (max FP4 e2m1 bucket)
        dq         = 6.0 * 0.125 = 0.75
    """
    global _SELF_TEST_DONE, _SELF_TEST_OK, _SELF_TEST_ERR

    if _SELF_TEST_DONE:
        if not _SELF_TEST_OK:
            raise RuntimeError(
                f"mxfp4_npu kernel previously failed self-test: {_SELF_TEST_ERR}"
            )
        return
    _SELF_TEST_DONE = True

    if _ext is None:
        _SELF_TEST_OK = False
        _SELF_TEST_ERR = "extension not loaded"
        raise RuntimeError("mxfp4_npu extension not loaded.")

    try:
        is_npu_available = bool(
            getattr(torch, "npu", None) and torch.npu.is_available()
        )
    except Exception:
        is_npu_available = False
    if not is_npu_available:
        # Cannot self-test without an NPU; defer validation to first real call.
        _SELF_TEST_OK = True
        return

    try:
        x = torch.ones(32, dtype=torch.float32, device="npu")
        y = _ext.mxfp4_quant_dequant_npu(x, 20)
        torch.npu.synchronize()
        diff = (y.float() - 0.75).abs().max().item()
    except Exception as e:
        _SELF_TEST_OK = False
        _SELF_TEST_ERR = f"kernel raised during self-test: {e}"
        raise RuntimeError(
            f"mxfp4_npu kernel self-test FAILED while invoking the kernel: "
            f"{e}. Check that the compiled kernel SoC matches the runtime "
            f"hardware (`npu-smi info -t board -i 0 -c 0 | "
            f"grep 'NPU Name'`)."
        ) from e

    if not (diff < 1e-5):
        _SELF_TEST_OK = False
        _SELF_TEST_ERR = (
            f"input=ones(32), expected output all 0.75, max_abs_err={diff:.4e}"
        )
        raise RuntimeError(
            "mxfp4_npu kernel self-test FAILED: input=ones(32), "
            f"expected output all 0.75, max_abs_err={diff:.4e}. This usually "
            "means the compiled kernel SoC does not match the runtime "
            "hardware. Check SoC with "
            "`npu-smi info -t board -i 0 -c 0 | grep 'NPU Name'` and rebuild "
            "with the matching SOC_VERSION (e.g. "
            "SOC_VERSION=Ascend910_9392 bash build.sh)."
        )

    _SELF_TEST_OK = True


def quant_dequant_mxfp4(
    x: torch.Tensor,
    block_size: int = 32,
    block_dim: int = 20,
    inv_scale_factor_scale: float = 1.0,
) -> torch.Tensor:
    """MXFP4 quant-dequant on Ascend NPU.

    Drop-in replacement for the PyTorch reference ``quant_dequant_mxfp4()``.
    Input tensor must be on an NPU device.

    Parameters
    ----------
    x : torch.Tensor
        Any shape, float32 (or auto-cast), on ``npu`` device.
    block_size : int
        Quantisation block width. Must be 32.
    block_dim : int
        Number of AI Cores to launch (default 20 for 910B3).
    inv_scale_factor_scale : float
        Runtime multiplier applied on top of the kernel's built-in
        ``INV_SCALE_FACTOR`` (1/6.0). The effective inverse scale factor
        becomes ``(1/6.0) * inv_scale_factor_scale``, i.e. the per-block
        ``raw_scale = max_abs / 6.0`` is multiplied by this value before the
        power-of-two rounding. Default ``1.0`` reproduces the standard MXFP4
        behaviour.

    Returns
    -------
    torch.Tensor
        Same shape / dtype / device as *x*.
    """
    if not _load_ext():
        raise RuntimeError(
            "mxfp4_npu: failed to load compiled extension. "
            "Ensure this package was built for your platform "
            "(Ascend 910B3, CANN 8.2, Python 3.10 aarch64)."
        )

    if block_size != 32:
        raise ValueError(
            f"only block_size=32 supported by the NPU kernel, got {block_size}"
        )

    if block_dim <= 0:
        raise ValueError(f"block_dim must be positive, got {block_dim}")

    if inv_scale_factor_scale <= 0:
        raise ValueError(
            f"inv_scale_factor_scale must be positive, got {inv_scale_factor_scale}"
        )

    orig_dtype = x.dtype
    if not x.is_npu:
        raise ValueError(
            f"input must be on an NPU device, got device={x.device}"
        )
    # Pad along the last dim first (per-row blocks), matching the PyTorch
    # reference. Flattening before pad would merge adjacent rows into the
    # same MXFP4 block when last_dim is not a multiple of block_size.
    x_fp = x.to(torch.float32)
    last_dim = x_fp.shape[-1]
    pad = (block_size - last_dim % block_size) % block_size
    if pad:
        x_fp = F.pad(x_fp, (0, pad))
    padded_shape = x_fp.shape
    x_flat = x_fp.reshape(-1).contiguous()

    y_flat = _ext.mxfp4_quant_dequant_npu(
        x_flat, block_dim, float(inv_scale_factor_scale)
    )

    y_fp = y_flat.reshape(padded_shape)
    if pad:
        y_fp = y_fp[..., :last_dim]

    return y_fp.to(orig_dtype)
