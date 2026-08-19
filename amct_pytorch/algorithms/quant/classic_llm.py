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

"""Scalable LLM-new-path variants of the Classic MinMax, AWQ and GPTQ ideas.

The eager Classic implementations operate on a fully resident model.  The LLM
workflow instead loads one decoder unit at a time, so these algorithms expose a
small finalized state that can be persisted and reapplied to source weights.
"""

from __future__ import annotations

import torch

from amct_pytorch.algorithms.quant.base import QuantAlgorithmBase
from amct_pytorch.algorithms.registry_factory import ALGO_REGISTRY
from amct_pytorch.quantization.dtypes.int_impl import dense_linear_w4a8_quant


_EPS = 1e-9
_SEARCH_STEPS = 10
_MIN_CLIP = 0.5


def _validate_int_args(args, bits):
    if args.quant_dtype != "int":
        raise ValueError(
            "LLM MinMax/AWQ/GPTQ adapters currently support quant_dtype='int' only."
        )
    if bits not in (4, 8, 16):
        raise ValueError(f"LLM MinMax/AWQ/GPTQ require 4, 8 or 16 bits, got {bits}.")


def _symmetric_scale(weight: torch.Tensor, bits: int) -> torch.Tensor:
    qmax = 2 ** (bits - 1) - 1
    return (
        weight.detach()
        .float()
        .abs()
        .amax(dim=1, keepdim=True)
        .div(qmax)
        .clamp_min(_EPS)
    )


def _quant_dequant_with_scale(
    weight: torch.Tensor, scale: torch.Tensor, bits: int
) -> torch.Tensor:
    if bits == 16:
        return weight
    qmax = 2 ** (bits - 1) - 1
    qmin = -qmax
    scale = scale.to(device=weight.device, dtype=weight.dtype)
    quantized = torch.round(weight / scale).clamp(qmin, qmax)
    return (quantized * scale).to(weight.dtype)


class _StatisticalWeightAlgorithm(QuantAlgorithmBase):
    """Shared lifecycle for non-gradient, calibration-statistics algorithms."""

    def __init__(self, args, bits):
        super().__init__()
        _validate_int_args(args, bits)
        self.bits = bits
        self.rows, self.columns = tuple(args.w_size)
        self._observed_weight = None

    def forward(self, weight):
        return weight

    def trainable_params(self):
        return []

    def _remember_weight(self, weight):
        if tuple(weight.shape) != (self.rows, self.columns):
            raise ValueError(
                f"weight shape changed from {(self.rows, self.columns)} to "
                f"{tuple(weight.shape)} during calibration."
            )
        self._observed_weight = weight.detach()

    def _require_observed_weight(self):
        if self._observed_weight is None:
            raise RuntimeError(
                f"{type(self).__name__} has no observed weight; run calibration first."
            )
        return self._observed_weight

    @staticmethod
    def _load_buffer(target, source, name):
        if not isinstance(source, torch.Tensor) or source.shape != target.shape:
            shape = tuple(source.shape) if isinstance(source, torch.Tensor) else None
            raise ValueError(
                f"invalid {name}: expected tensor shape {tuple(target.shape)}, got {shape}."
            )
        if not torch.isfinite(source).all():
            raise ValueError(f"invalid {name}: values must be finite.")
        target.copy_(source.to(device=target.device, dtype=target.dtype))


@ALGO_REGISTRY.register(
    name="minmax",
    description="Per-output-channel symmetric MinMax weight quantization",
    targets=("weight",),
    calibration="statistics",
)
class LlmMinMax(_StatisticalWeightAlgorithm):
    """Persist the exact per-output-channel range used to quantize each weight."""

    def __init__(self, args, bits):
        super().__init__(args, bits)
        self.register_buffer("weight_scale", torch.ones((self.rows, 1)))
        self.register_buffer("ready", torch.tensor(False, dtype=torch.bool))

    def calib_forward(self, weight, *args, **kwargs):
        self._remember_weight(weight)
        self.weight_scale.copy_(_symmetric_scale(weight, self.bits))
        self.ready.fill_(True)
        return weight

    def _ensure_ready(self, weight):
        if not self.ready.item():
            self.calib_forward(weight)

    def quantize(self, weight, quant_obj):
        del quant_obj
        self._ensure_ready(weight)
        return _quant_dequant_with_scale(weight, self.weight_scale, self.bits)

    def export_ptq_params(self):
        if not self.ready.item():
            self.calib_forward(self._require_observed_weight())
        return {"weight_scale": self.weight_scale.detach().cpu()}

    def load_ptq_params(self, params):
        if not isinstance(params, dict) or set(params) != {"weight_scale"}:
            raise ValueError("MinMax parameters must contain only 'weight_scale'.")
        self._load_buffer(self.weight_scale, params["weight_scale"], "weight_scale")
        if not torch.all(self.weight_scale > 0):
            raise ValueError("MinMax weight_scale values must be positive.")
        self.ready.fill_(True)

    def export_deploy(self, weight, quant_obj):
        self._ensure_ready(weight)
        if self.bits == 4:
            return quant_obj.export_deploy(weight)
        if self.bits == 8:
            scale = self.weight_scale.to(device=weight.device, dtype=weight.dtype)
            qweight = torch.round(weight / scale).clamp(-127, 127).to(torch.int8)
            return {
                "qweight": qweight.detach().cpu(),
                "weight_scale": scale.float().detach().cpu(),
                "weight_bias": None,
            }
        raise NotImplementedError("MinMax deploy export supports W8 and W4 only.")

    def export_dense_linear_deploy(self, weight, quant_obj):
        del quant_obj
        self._ensure_ready(weight)
        return dense_linear_w4a8_quant(weight, self.weight_scale)


@ALGO_REGISTRY.register(
    name="awq",
    description="Activation-aware channel equalization for LLM weight quantization",
    targets=("weight",),
    calibration="statistics",
)
class LlmAwq(_StatisticalWeightAlgorithm):
    """Search an input-channel equalization scale using activation statistics."""

    def __init__(self, args, bits):
        super().__init__(args, bits)
        self.register_buffer("sum_abs", torch.zeros(self.columns, dtype=torch.float32))
        self.register_buffer("sum_sq", torch.zeros(self.columns, dtype=torch.float32))
        self.register_buffer("sample_count", torch.zeros((), dtype=torch.float64))
        self.register_buffer("channel_scale", torch.ones((1, self.columns)))
        self.register_buffer("selected_alpha", torch.zeros((), dtype=torch.float32))
        self.register_buffer("ready", torch.tensor(False, dtype=torch.bool))

    def observe_input(self, inputs, weight=None):
        if weight is not None:
            self._remember_weight(weight)
        flat = inputs.detach().reshape(-1, inputs.shape[-1]).float()
        if flat.shape[-1] != self.columns:
            raise ValueError(
                f"AWQ input width must be {self.columns}, got {flat.shape[-1]}."
            )
        self.sum_abs.add_(flat.abs().sum(dim=0).to(self.sum_abs))
        self.sum_sq.add_(flat.square().sum(dim=0).to(self.sum_sq))
        self.sample_count.add_(flat.shape[0])

    @torch.no_grad()
    def _finalize(self):
        weight = self._require_observed_weight()
        if self.sample_count.item() <= 0:
            raise RuntimeError("AWQ requires at least one observed activation token.")
        act_mean = (self.sum_abs / self.sample_count).clamp_min(1e-6)
        importance = (self.sum_sq / self.sample_count).clamp_min(1e-8)
        weight_mean = weight.detach().float().abs().mean(dim=0).clamp_min(1e-6)

        best_loss = None
        best_scale = None
        best_alpha = None
        for index in range(_SEARCH_STEPS + 1):
            alpha = index / _SEARCH_STEPS
            scale = act_mean.pow(alpha) / weight_mean.pow(1.0 - alpha)
            scale = scale / torch.sqrt(scale.max() * scale.min()).clamp_min(1e-6)
            scale = scale.clamp(1e-4, 1e4)
            scaled_weight = weight * scale.to(weight)
            dequant = _quant_dequant_with_scale(
                scaled_weight, _symmetric_scale(scaled_weight, self.bits), self.bits
            )
            reconstructed = dequant.float() / scale.unsqueeze(0)
            loss = ((weight.float() - reconstructed).square() * importance).mean()
            if best_loss is None or loss < best_loss:
                best_loss = loss
                best_scale = scale
                best_alpha = alpha

        self.channel_scale.copy_(best_scale.reshape(1, -1).to(self.channel_scale))
        self.selected_alpha.fill_(best_alpha)
        self.ready.fill_(True)

    def transform_input(self, inputs):
        if not self.ready.item():
            return inputs
        return inputs / self.channel_scale.to(device=inputs.device, dtype=inputs.dtype)

    def quantize(self, weight, quant_obj):
        if not self.ready.item():
            raise RuntimeError("AWQ parameters are not finalized or loaded.")
        scale = self.channel_scale.to(device=weight.device, dtype=weight.dtype)
        return quant_obj(weight * scale)

    def export_ptq_params(self):
        if not self.ready.item():
            self._finalize()
        return {
            "channel_scale": self.channel_scale.detach().cpu(),
            "selected_alpha": self.selected_alpha.detach().cpu(),
        }

    def load_ptq_params(self, params):
        expected = {"channel_scale", "selected_alpha"}
        if not isinstance(params, dict) or set(params) != expected:
            raise ValueError(f"AWQ parameters must contain exactly {sorted(expected)}.")
        self._load_buffer(self.channel_scale, params["channel_scale"], "channel_scale")
        self._load_buffer(
            self.selected_alpha, params["selected_alpha"], "selected_alpha"
        )
        if not torch.all(self.channel_scale > 0):
            raise ValueError("AWQ channel_scale values must be positive.")
        if not 0 <= self.selected_alpha.item() <= 1:
            raise ValueError("AWQ selected_alpha must be in [0, 1].")
        self.ready.fill_(True)

    def export_deploy(self, weight, quant_obj):
        del weight, quant_obj
        raise NotImplementedError(
            "AWQ deploy requires a runtime input-channel transformation; "
            "compressed-tensors dense Linear has no field for channel_scale."
        )

    def export_dense_linear_deploy(self, weight, quant_obj):
        return self.export_deploy(weight, quant_obj)


@ALGO_REGISTRY.register(
    name="gptq_diag",
    description="Diagonal-Hessian GPTQ clipping approximation for LLM weights",
    targets=("weight",),
    calibration="statistics",
    approximation="diagonal_hessian",
)
class LlmDiagonalGptq(_StatisticalWeightAlgorithm):
    """Use diagonal second-order input statistics to choose row clipping ranges."""

    def __init__(self, args, bits):
        super().__init__(args, bits)
        self.register_buffer("hessian_diag", torch.zeros(self.columns))
        self.register_buffer("sample_count", torch.zeros((), dtype=torch.float64))
        self.register_buffer("clip_factor", torch.ones((self.rows, 1)))
        self.register_buffer("ready", torch.tensor(False, dtype=torch.bool))

    def observe_input(self, inputs, weight=None):
        if weight is not None:
            self._remember_weight(weight)
        flat = inputs.detach().reshape(-1, inputs.shape[-1]).float()
        if flat.shape[-1] != self.columns:
            raise ValueError(
                f"GPTQ input width must be {self.columns}, got {flat.shape[-1]}."
            )
        self.hessian_diag.add_(flat.square().sum(dim=0).to(self.hessian_diag))
        self.sample_count.add_(flat.shape[0])

    @torch.no_grad()
    def _finalize(self):
        weight = self._require_observed_weight()
        if self.sample_count.item() <= 0:
            raise RuntimeError("GPTQ requires at least one observed activation token.")
        importance = (self.hessian_diag / self.sample_count).clamp_min(1e-8)
        row_absmax = weight.detach().float().abs().amax(dim=1, keepdim=True)
        best_loss = torch.full(
            (self.rows,), float("inf"), device=weight.device, dtype=torch.float32
        )
        best_factor = torch.ones_like(best_loss)

        for index in range(_SEARCH_STEPS + 1):
            factor = 1.0 - (1.0 - _MIN_CLIP) * index / _SEARCH_STEPS
            bound = (row_absmax * factor).to(weight)
            clipped = weight.clamp(-bound, bound)
            dequant = _quant_dequant_with_scale(
                clipped, _symmetric_scale(clipped, self.bits), self.bits
            )
            loss = ((weight.float() - dequant.float()).square() * importance).mean(
                dim=1
            )
            update = loss < best_loss
            best_loss = torch.where(update, loss, best_loss)
            best_factor = torch.where(
                update, torch.full_like(best_factor, factor), best_factor
            )

        self.clip_factor.copy_(best_factor.reshape(-1, 1).to(self.clip_factor))
        self.ready.fill_(True)

    def quantize(self, weight, quant_obj):
        if not self.ready.item():
            raise RuntimeError("GPTQ parameters are not finalized or loaded.")
        row_absmax = weight.detach().abs().amax(dim=1, keepdim=True)
        bound = row_absmax * self.clip_factor.to(weight)
        return quant_obj(weight.clamp(-bound, bound))

    def export_ptq_params(self):
        if not self.ready.item():
            self._finalize()
        return {"clip_factor": self.clip_factor.detach().cpu()}

    def load_ptq_params(self, params):
        if not isinstance(params, dict) or set(params) != {"clip_factor"}:
            raise ValueError("GPTQ parameters must contain only 'clip_factor'.")
        self._load_buffer(self.clip_factor, params["clip_factor"], "clip_factor")
        if not torch.all((self.clip_factor >= _MIN_CLIP) & (self.clip_factor <= 1)):
            raise ValueError(f"GPTQ clip_factor values must be in [{_MIN_CLIP}, 1].")
        self.ready.fill_(True)

    def _clipped_weight(self, weight):
        if not self.ready.item():
            raise RuntimeError("GPTQ parameters are not finalized or loaded.")
        row_absmax = weight.detach().abs().amax(dim=1, keepdim=True)
        bound = row_absmax * self.clip_factor.to(weight)
        return weight.clamp(-bound, bound)

    def export_deploy(self, weight, quant_obj):
        if self.bits not in (4, 8):
            raise NotImplementedError("GPTQ deploy export supports W8 and W4 only.")
        clipped = self._clipped_weight(weight)
        return quant_obj.export_deploy(clipped)

    def export_dense_linear_deploy(self, weight, quant_obj):
        del quant_obj
        if self.bits != 4:
            raise NotImplementedError("dense GPTQ export is only needed for W4.")
        return dense_linear_w4a8_quant(self._clipped_weight(weight))
