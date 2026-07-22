"""
Python wrapper for the MXFP4 Ascend-C quant-dequant kernel.

Usage:
    from mxfp4_npu import quant_dequant_mxfp4_npu

    x_npu = torch.randn(4, 128, device="npu:0")
    y_npu = quant_dequant_mxfp4_npu(x_npu)
"""

from __future__ import annotations

import ctypes
import os
import sys

import torch
import torch.nn.functional as F

_DIR = os.path.dirname(os.path.abspath(__file__))
_BUILD_DIR = os.path.join(_DIR, "build")
_EXT_LOADED = False
_ext = None


def _try_load() -> bool:
    global _EXT_LOADED, _ext
    if _EXT_LOADED:
        return _ext is not None

    if not os.path.isdir(_BUILD_DIR):
        _EXT_LOADED = True
        return False

    # pre-load libtorch_npu.so so that c10_npu symbols are globally visible
    try:
        import torch_npu as _tnpu

        _tnpu_dir = os.path.dirname(_tnpu.__file__)
        _tnpu_lib = os.path.join(_tnpu_dir, "lib", "libtorch_npu.so")
        if os.path.isfile(_tnpu_lib):
            ctypes.CDLL(_tnpu_lib, mode=ctypes.RTLD_GLOBAL)
    except Exception:
        pass

    # pre-load the device kernel shared lib
    lib_dir = os.path.join(_BUILD_DIR, "lib")
    for name in ("libascendc_kernels_npu.so",):
        for d in (lib_dir, _BUILD_DIR):
            p = os.path.join(d, name)
            if os.path.isfile(p):
                try:
                    ctypes.CDLL(p, mode=ctypes.RTLD_GLOBAL)
                except OSError:
                    pass
                break

    if _BUILD_DIR not in sys.path:
        sys.path.insert(0, _BUILD_DIR)

    try:
        import mxfp4_ascendc_ext  # type: ignore[import-not-found]

        _ext = mxfp4_ascendc_ext
    except ImportError as e:
        print(f"[mxfp4_npu] Failed to load extension: {e}")

    _EXT_LOADED = True
    return _ext is not None


def quant_dequant_mxfp4_npu(
    x: torch.Tensor,
    block_size: int = 32,
    block_dim: int = 20,
    inv_scale_factor_scale: float = 1.0,
) -> torch.Tensor:
    """
    MXFP4 quant-dequant on Ascend NPU.

    Parameters
    ----------
    x : torch.Tensor   (any shape, float32, on NPU device)
    block_size : int    quantisation block width (must be 32)
    block_dim  : int    number of AI Cores to launch
    inv_scale_factor_scale : float
        Runtime multiplier applied on top of the kernel's built-in
        INV_SCALE_FACTOR (1/6.0). Default 1.0 keeps standard MXFP4 behaviour.

    Returns
    -------
    torch.Tensor with same shape / dtype / device as *x*.
    """
    if not _try_load():
        raise RuntimeError(
            "mxfp4_ascendc_ext not built.  Run:  bash mxfp4_ascendc/build.sh"
        )

    assert block_size == 32, "only block_size=32 supported by the NPU kernel"

    orig_shape = x.shape
    orig_dtype = x.dtype
    x_fp = x.to(torch.float32).reshape(-1).contiguous()

    # pad to multiple of 32
    n = x_fp.numel()
    pad = (block_size - n % block_size) % block_size
    if pad:
        x_fp = F.pad(x_fp, (0, pad))

    y_fp = _ext.mxfp4_quant_dequant_npu(x_fp, block_dim, float(inv_scale_factor_scale))

    if pad:
        y_fp = y_fp[:n]

    return y_fp.reshape(orig_shape).to(orig_dtype)
