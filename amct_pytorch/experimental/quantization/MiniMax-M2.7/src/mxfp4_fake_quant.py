"""MXFP4 fake quantization for OS+ migration search (NPU-compatible).

用于 OSPlus SmoothQuant 校准搜索阶段的 MXFP4 伪量化算子：对沿 reduction 轴的张量
先执行 MXFP4 量化再反量化，返回与原 dtype 一致的高精度张量，用于在 W4A4 MXFP4
配置下评估 SmoothQuant 等价缩放的输出重建误差。底层 ``mxfp4_quantize`` /
``mxfp4_dequantize`` 均为 device-agnostic 的纯 torch 实现，会在输入张量所在设备
（CPU / NPU / CUDA）上运行，且与昇腾 MXFP4 GEMM 反量化路径的舍入规则严格一致。
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import torch


def _ensure_mxfp4_quantizer_on_path() -> None:
    """Make ``mxfp4_quantizer`` importable in a machine-independent way.

    Resolution order (first hit wins):
      1. Already importable from the current ``sys.path`` / installed env.
      2. ``MXFP4_QUANTIZER_PATH`` env var — points to the directory
         **containing** the ``mxfp4_quantizer/`` package, or the package
         directory itself.
      3. The vendored copy shipped with this sample at
         ``<sample_root>/mxfp4_quantizer/`` — discovered by walking up from
         this file (``src/`` 的上一级即样例根目录），so copying the whole
         sample directory to a new machine just works.
    """

    try:
        importlib.import_module("mxfp4_quantizer")
        return
    except ImportError:
        pass

    candidates: list[Path] = []

    env_path = os.environ.get("MXFP4_QUANTIZER_PATH")
    if env_path:
        p = Path(env_path).expanduser().resolve()
        if p.name == "mxfp4_quantizer":
            candidates.append(p.parent)
        else:
            candidates.append(p)

    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "mxfp4_quantizer" / "__init__.py").is_file():
            candidates.append(parent)
            break

    for cand in candidates:
        if (cand / "mxfp4_quantizer" / "__init__.py").is_file():
            sys.path.insert(0, str(cand))
            return

    raise ImportError(
        "Could not locate the `mxfp4_quantizer` package. Set the "
        "MXFP4_QUANTIZER_PATH env var to the directory containing it, or "
        "make sure `<sample_root>/mxfp4_quantizer/__init__.py` exists."
    )


def _load_mxfp4_ops():
    """Resolve path then load mxfp4 quantize/dequantize via importlib."""
    _ensure_mxfp4_quantizer_on_path()
    mod = importlib.import_module("mxfp4_quantizer")
    return mod.mxfp4_quantize, mod.mxfp4_dequantize


mxfp4_quantize, mxfp4_dequantize = _load_mxfp4_ops()


def fake_quantize_mxfp4(tensor: torch.Tensor, axis: int = -1) -> torch.Tensor:
    orig_dtype = tensor.dtype
    t_float = tensor.float()
    packed, scale_e8m0 = mxfp4_quantize(t_float, axis=axis)
    dequant = mxfp4_dequantize(
        packed, scale_e8m0, target_dtype=torch.float32, axis=axis
    )
    return dequant.to(orig_dtype)


def fake_quantize_mxfp4_weight(weight: torch.Tensor) -> torch.Tensor:
    return fake_quantize_mxfp4(weight, axis=-1)


def fake_quantize_mxfp4_activation(activation: torch.Tensor) -> torch.Tensor:
    return fake_quantize_mxfp4(activation, axis=-1)
