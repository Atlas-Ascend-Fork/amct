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
"""Diagnostics / validation layer — post-conversion quality scoring.

Provides error statistics for both HiF4 modes:
  - ``compute_error_stats`` — generic error metrics (works for e6m2 and e6m0).
  - ``compute_hif4_saturation`` — saturation / zero rates (e6m0 only;
    requires debug fields from the local encoder).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class Hif4Stats:
    """Container for HiF4 conversion quality metrics."""

    numel: int
    max_abs_x: float
    max_abs_recon: float
    mean_abs_err: float
    max_abs_err: float
    rel_l2_err: float
    sat_rate: Optional[float] = None       # only available for e6m0
    zero_nonzero_rate: Optional[float] = None  # only available for e6m0


def compute_error_stats(
    x: torch.Tensor, recon: torch.Tensor
) -> dict:
    """Compute generic reconstruction error metrics.

    These metrics are computable for **both** e6m2 and e6m0 modes because
    they only need the original tensor *x* and the reconstructed tensor *recon*.

    Args:
        x: Original tensor (any float dtype).
        recon: Reconstructed tensor (same shape as *x*).

    Returns:
        Dict with keys: ``numel``, ``max_abs_x``, ``max_abs_recon``,
        ``mean_abs_err``, ``max_abs_err``, ``rel_l2_err``.
    """
    x32 = x.float()
    r32 = recon.float()
    err = (x32 - r32).abs()

    denom = torch.linalg.vector_norm(x32.reshape(-1)).clamp_min(1e-30)
    rel_l2 = (
        torch.linalg.vector_norm((x32 - r32).reshape(-1)) / denom
    )

    return {
        "numel": int(x32.numel()),
        "max_abs_x": float(x32.abs().max().item()),
        "max_abs_recon": float(r32.abs().max().item()),
        "mean_abs_err": float(err.mean().item()),
        "max_abs_err": float(err.max().item()),
        "rel_l2_err": float(rel_l2.item()),
    }


def compute_hif4_saturation(
    qmag: torch.Tensor, y_scaled: torch.Tensor
) -> dict:
    """Compute HiF4-specific saturation and false-zero rates (e6m0 only).

    These metrics require the debug fields ``qmag`` and ``y_scaled`` that
    are only produced by :func:`fp4_formats.bf16_to_hif4_e6m0`. The official
    e6m2 encoder does **not** return these fields, so this function is not
    usable with e6m2 output.

    Args:
        qmag: Per-element quantized magnitude indices (``[..., 64]``, uint8).
        y_scaled: Per-element scaled values before rounding (``[..., 64]``, float32).

    Returns:
        Dict with keys: ``sat_rate``, ``zero_nonzero_rate``.
    """
    sat_rate = (
        ((qmag == 7) & (y_scaled.abs() > 1.75)).float().mean()
    )
    zero_nonzero_rate = (
        ((qmag == 0) & (y_scaled.abs() > 0)).float().mean()
    )
    return {
        "sat_rate": float(sat_rate.item()),
        "zero_nonzero_rate": float(zero_nonzero_rate.item()),
    }


def make_stats(
    x: torch.Tensor,
    recon: torch.Tensor,
    qmag: Optional[torch.Tensor] = None,
    y_scaled: Optional[torch.Tensor] = None,
) -> Hif4Stats:
    """Convenience: build a full ``Hif4Stats`` from available data.

    *qmag* and *y_scaled* are optional — if omitted, saturation metrics
    will be ``None`` (this is the normal case for e6m2 mode).
    """
    err = compute_error_stats(x, recon)
    sat: dict = {}
    if qmag is not None and y_scaled is not None:
        sat = compute_hif4_saturation(qmag, y_scaled)
    return Hif4Stats(
        numel=err["numel"],
        max_abs_x=err["max_abs_x"],
        max_abs_recon=err["max_abs_recon"],
        mean_abs_err=err["mean_abs_err"],
        max_abs_err=err["max_abs_err"],
        rel_l2_err=err["rel_l2_err"],
        sat_rate=sat.get("sat_rate"),
        zero_nonzero_rate=sat.get("zero_nonzero_rate"),
    )
