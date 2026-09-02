# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------
"""MXFP4 fake quantisation with a straight-through estimator (STE).

MXFP4 = per-element FP4 E2M1 mantissa plus a per-block E8M0 (power-of-two)
shared scale, with ``block_size`` consecutive elements of the last dimension
sharing one scale.

Quantisation is not differentiable (it is piecewise constant, so its true
derivative is zero almost everywhere). QAT therefore uses the straight-through
estimator: the forward pass sees quantised values while the backward pass
treats the quantiser as an identity, letting the gradient reach the underlying
high-precision weights and activations.

This module is intentionally self-contained (torch only) so it can be copied
into a third-party training framework as-is. The Ascend-C kernel under
``../mxfp4_ascendc`` is used automatically when it is available and the tensor
lives on an NPU; otherwise a pure-PyTorch path runs on any device.
"""

from __future__ import annotations

__all__ = [
    "BACKEND_AUTO",
    "BACKEND_NPU",
    "BACKEND_TORCH",
    "BLOCK_SIZE",
    "MXFP4_E2M1_MAX",
    "MXFP4FakeQuantizer",
    "MXFP4QATConfig",
    "SCALE_FACTOR",
    "mxfp4_fake_quant",
    "mxfp4_quant_dequant",
    "mxfp4_saturation_mask",
]

import dataclasses
import os
import sys
from typing import Callable

import torch
import torch.nn.functional as F
from torch import nn

BLOCK_SIZE = 32
SCALE_FACTOR = 6.0
MXFP4_E2M1_MAX = 6.0

# Smallest representable block scale, kept in sync with MXFP4_MIN_SCALE_RAW in
# ../mxfp4_ascendc/op_kernel/mxfp4_tiling.h.
_MIN_SCALE_RAW = 2.0**-30

# Positive FP4 E2M1 codebook and the midpoints between its adjacent entries.
_E2M1_CODES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
_E2M1_MIDPOINTS = (0.25, 0.75, 1.25, 1.75, 2.50, 3.50, 5.00)
_codebook_cache: dict[tuple[torch.device, torch.dtype], tuple[torch.Tensor, ...]] = {}

BACKEND_AUTO = "auto"
BACKEND_TORCH = "torch"
BACKEND_NPU = "npu"
_BACKENDS = (BACKEND_AUTO, BACKEND_TORCH, BACKEND_NPU)

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
# Ascend-C kernel shipped next to this package; overridable via env var so a
# framework can point at an out-of-tree build.
_VENDORED_KERNEL_PATH = os.path.join(_PKG_DIR, os.pardir, "mxfp4_ascendc", "python")

_npu_qdq: Callable[..., torch.Tensor] | None = None
_npu_load_error: str | None = None


def _validate(block_size: int, scale_factor: float) -> None:
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")
    if scale_factor <= 0:
        raise ValueError(f"scale_factor must be positive, got {scale_factor}")


def _block_view(x: torch.Tensor, block_size: int) -> tuple[torch.Tensor, int]:
    """Reshape *x* to ``(..., n_block, block_size)``, zero-padding the last dim.

    Padding is applied on the last dimension rather than on a flattened view so
    that adjacent rows never end up sharing an MXFP4 block.
    """
    if x.ndim == 0:
        raise ValueError("x must have at least 1 dimension")
    last_dim = x.shape[-1]
    pad = (block_size - last_dim % block_size) % block_size
    x_fp = x.to(torch.float32)
    if pad:
        x_fp = F.pad(x_fp, (0, pad))
    return x_fp.reshape(*x_fp.shape[:-1], -1, block_size), pad


def _unpad(blocked: torch.Tensor, last_dim: int, pad: int) -> torch.Tensor:
    flat = blocked.reshape(*blocked.shape[:-2], -1)
    return flat[..., :last_dim] if pad else flat


def _block_scale(x_blocked: torch.Tensor, scale_factor: float) -> torch.Tensor:
    """Per-block E8M0 scale: the power of two nearest to ``max_abs / scale_factor``."""
    max_abs = x_blocked.abs().amax(dim=-1, keepdim=True)
    raw_scale = torch.clamp(max_abs / scale_factor, min=_MIN_SCALE_RAW)
    return torch.exp2(torch.round(torch.log2(raw_scale)))


def _codebook(like: torch.Tensor) -> tuple[torch.Tensor, ...]:
    key = (like.device, like.dtype)
    if key not in _codebook_cache:
        _codebook_cache[key] = tuple(
            torch.tensor(values, device=like.device, dtype=like.dtype)
            for values in (_E2M1_MIDPOINTS, _E2M1_CODES)
        )
    return _codebook_cache[key]


def _round_to_e2m1(y_abs: torch.Tensor) -> torch.Tensor:
    """Round each magnitude to the nearest FP4 E2M1 code.

    Since the thresholds are the codebook midpoints, the number of thresholds a
    value exceeds is exactly the index of its nearest code.
    """
    midpoints, codes = _codebook(y_abs)
    return codes[torch.bucketize(y_abs, midpoints, right=True)]


def _quant_dequant_torch(
    x: torch.Tensor, block_size: int, scale_factor: float
) -> torch.Tensor:
    x_blocked, pad = _block_view(x, block_size)
    scale = _block_scale(x_blocked, scale_factor)
    y = x_blocked / scale
    q = torch.sign(y) * _round_to_e2m1(y.abs())
    return _unpad(q * scale, x.shape[-1], pad).to(x.dtype)


def _load_npu_qdq() -> Callable[..., torch.Tensor]:
    """Import the Ascend-C MXFP4 kernel wrapper, caching success and failure."""
    global _npu_qdq, _npu_load_error

    if _npu_qdq is not None:
        return _npu_qdq
    if _npu_load_error is not None:
        raise RuntimeError(_npu_load_error)

    search_path = os.environ.get("MXFP4_ASCENDC_PATH") or os.path.realpath(
        _VENDORED_KERNEL_PATH
    )
    if search_path not in sys.path:
        sys.path.insert(0, search_path)

    try:
        import mxfp4

        _npu_qdq = mxfp4.quant_dequant_mxfp4
    except Exception as e:
        _npu_load_error = (
            f"cannot load the Ascend-C MXFP4 kernel from '{search_path}': {e}. "
            "Build it with `bash build.sh` in mxfp4_ascendc/, or point "
            "MXFP4_ASCENDC_PATH at a directory containing the built `mxfp4` "
            f"package, or use backend='{BACKEND_TORCH}'."
        )
        raise RuntimeError(_npu_load_error) from e

    return _npu_qdq


def _npu_qdq_available() -> bool:
    try:
        _load_npu_qdq()
    except RuntimeError:
        return False
    return True


def mxfp4_quant_dequant(
    x: torch.Tensor,
    block_size: int = BLOCK_SIZE,
    scale_factor: float = SCALE_FACTOR,
    backend: str = BACKEND_AUTO,
) -> torch.Tensor:
    """Quantise *x* to MXFP4 and dequantise it back, without any autograd hook.

    Bit-exactly matches ``mxfp4_ascendc/reference/mxfp4_ref.quant_dequant_mxfp4``.
    Use :func:`mxfp4_fake_quant` instead when training.

    Args:
        x: Tensor with at least one dimension, any dtype/device.
        block_size: Elements per shared scale. The NPU kernel only supports 32.
        scale_factor: Divisor applied to each block maximum before rounding the
            scale to a power of two. ``6.0`` maps the block maximum onto the
            largest E2M1 code.
        backend: ``"auto"`` picks the Ascend-C kernel for NPU tensors when it is
            available and falls back to PyTorch; ``"torch"`` / ``"npu"`` force
            one path.

    Returns:
        Tensor with the same shape, dtype and device as *x*.

    Raises:
        ValueError: On a non-positive ``block_size`` / ``scale_factor``, a 0-dim
            input, or an unknown ``backend``.
        RuntimeError: If ``backend="npu"`` but the kernel cannot be loaded.
    """
    _validate(block_size, scale_factor)
    if backend not in _BACKENDS:
        raise ValueError(f"backend must be one of {_BACKENDS}, got {backend!r}")

    use_npu = backend == BACKEND_NPU or (
        backend == BACKEND_AUTO and x.device.type == "npu" and _npu_qdq_available()
    )
    if not use_npu:
        return _quant_dequant_torch(x, block_size, scale_factor)

    # The kernel hard-codes 1 / 6.0 and multiplies it by inv_scale_factor_scale,
    # so scale_factor == SCALE_FACTOR / inv_scale_factor_scale.
    return _load_npu_qdq()(
        x,
        block_size=block_size,
        inv_scale_factor_scale=SCALE_FACTOR / scale_factor,
    )


def mxfp4_saturation_mask(
    x: torch.Tensor,
    block_size: int = BLOCK_SIZE,
    scale_factor: float = SCALE_FACTOR,
) -> torch.Tensor:
    """Return a bool mask that is ``True`` where MXFP4 quantisation clips *x*.

    An element saturates when its magnitude exceeds ``6 * block_scale``, which
    happens because the block scale is rounded to a power of two and may round
    down. Gradients of clipped elements are misleading under a plain STE, so
    :class:`MXFP4QATConfig` can mask them out (clipped STE).
    """
    _validate(block_size, scale_factor)
    x_blocked, pad = _block_view(x, block_size)
    limit = MXFP4_E2M1_MAX * _block_scale(x_blocked, scale_factor)
    return _unpad(x_blocked.abs() > limit, x.shape[-1], pad)


class _MXFP4FakeQuantSTE(torch.autograd.Function):
    """Fake-quantise in forward; pass the gradient through unchanged in backward."""

    @staticmethod
    def forward(ctx, x, block_size, scale_factor, clip_grad, backend):
        ctx.clip_grad = clip_grad
        if clip_grad:
            ctx.save_for_backward(mxfp4_saturation_mask(x, block_size, scale_factor))
        return mxfp4_quant_dequant(x, block_size, scale_factor, backend)

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.clip_grad:
            (saturated,) = ctx.saved_tensors
            grad_output = grad_output.masked_fill(saturated, 0.0)
        return grad_output, None, None, None, None


def mxfp4_fake_quant(
    x: torch.Tensor,
    block_size: int = BLOCK_SIZE,
    scale_factor: float = SCALE_FACTOR,
    clip_grad: bool = False,
    backend: str = BACKEND_AUTO,
) -> torch.Tensor:
    """Differentiable MXFP4 fake quantisation (STE).

    Args:
        x: Tensor to fake-quantise, typically a weight or an activation.
        block_size: Elements per shared scale.
        scale_factor: See :func:`mxfp4_quant_dequant`.
        clip_grad: If ``True``, zero the gradient of elements that saturated
            (clipped STE) instead of passing everything through.
        backend: See :func:`mxfp4_quant_dequant`.

    Returns:
        Fake-quantised tensor, same shape/dtype/device as *x*, whose gradient
        flows back to *x* unchanged (or masked when ``clip_grad``).
    """
    return _MXFP4FakeQuantSTE.apply(x, block_size, scale_factor, clip_grad, backend)


class MXFP4FakeQuantizer(nn.Module):
    """Stateless module wrapper around :func:`mxfp4_fake_quant`.

    Holds no parameters or buffers, so inserting it into a model leaves the
    ``state_dict`` untouched and checkpoints stay interchangeable with the
    float model.
    """

    def __init__(
        self,
        block_size: int = BLOCK_SIZE,
        scale_factor: float = SCALE_FACTOR,
        clip_grad: bool = False,
        backend: str = BACKEND_AUTO,
    ) -> None:
        super().__init__()
        _validate(block_size, scale_factor)
        if backend not in _BACKENDS:
            raise ValueError(f"backend must be one of {_BACKENDS}, got {backend!r}")
        self.block_size = block_size
        self.scale_factor = scale_factor
        self.clip_grad = clip_grad
        self.backend = backend

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return mxfp4_fake_quant(
            x, self.block_size, self.scale_factor, self.clip_grad, self.backend
        )

    def extra_repr(self) -> str:
        return (
            f"block_size={self.block_size}, scale_factor={self.scale_factor}, "
            f"clip_grad={self.clip_grad}, backend={self.backend!r}"
        )


@dataclasses.dataclass
class MXFP4QATConfig:
    """MXFP4 QAT settings shared by every converted layer.

    Attributes:
        quantize_weight: Fake-quantise the weight. Disabling it turns the layer
            into an activation-only experiment.
        quantize_input: Fake-quantise the layer input as well. ``False`` gives
            W4A16 (the recommended starting point); ``True`` gives W4A4.
        block_size: Elements per shared scale. The Ascend-C kernel needs 32.
        scale_factor: See :func:`mxfp4_quant_dequant`. Raising it shrinks the
            block scale, resolving inliers more finely but clipping outliers
            harder; lowering it does the opposite.
        clip_grad: Use a clipped STE instead of a plain one.
        backend: ``"auto"`` / ``"torch"`` / ``"npu"``.
    """

    quantize_weight: bool = True
    quantize_input: bool = False
    block_size: int = BLOCK_SIZE
    scale_factor: float = SCALE_FACTOR
    clip_grad: bool = False
    backend: str = BACKEND_AUTO

    def __post_init__(self) -> None:
        _validate(self.block_size, self.scale_factor)
        if self.backend not in _BACKENDS:
            raise ValueError(
                f"backend must be one of {_BACKENDS}, got {self.backend!r}"
            )

    def make_quantizer(self) -> MXFP4FakeQuantizer:
        """Build a fresh quantizer module carrying this configuration."""
        return MXFP4FakeQuantizer(
            block_size=self.block_size,
            scale_factor=self.scale_factor,
            clip_grad=self.clip_grad,
            backend=self.backend,
        )
