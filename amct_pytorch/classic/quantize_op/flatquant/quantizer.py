# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------
#
# The code snippet comes from FlatQuant (https://github.com/ruikangliu/FlatQuant).
#
# Copyright (c) 2024 ruikangliu. All rights reserved.
#
# This file is part of FlatQuant, which is released under the MIT License.
# See the LICENSE file in the root directory of this source tree
# or at https://opensource.org/licenses/MIT for details.
#
# --------------------------------------------------------------------------------
"""Fake-quant primitives for FlatQuant: STE rounding, activation/weight quantizers."""

import torch


def round_ste(x: torch.Tensor):
    """Straight-Through Estimator for the rounding op."""
    return (x.round() - x).detach() + x


def get_qmin_qmax(bits, sym):
    if sym:
        q_max = torch.tensor(2 ** (bits - 1) - 1)
        q_min = -q_max - 1
    else:
        q_max, q_min = torch.tensor(2**bits - 1), 0
    return q_max, q_min


def sym_quant(x, scale, maxq):
    scale = scale.to(x.device)
    q = torch.clamp(round_ste(x / scale), -(maxq + 1), maxq)
    return q, scale


def sym_dequant(q, scale):
    return scale * q


def sym_quant_dequant(x, scale, maxq):
    return sym_dequant(*sym_quant(x, scale, maxq))


class ActivationQuantizer(torch.nn.Module):
    """Per-token activation fake quantizer, with optional learnable clip (lac).

    act_dtype selects the carrier for the quantize/dequantize math: 'fp32'
    (default, historical behaviour) or 'fp16' to mirror the
    npu_kronecker_quant fp16 rounding — the per-token max, scale division
    and rounding then happen on the same carrier as the deploy operator,
    so calibration noise matches on-device noise."""

    def __init__(self, bits, sym=True, lac=False, act_dtype='fp32'):
        super().__init__()
        self.bits = bits
        self.q_max, self.q_min = get_qmin_qmax(bits, sym)
        self.sym = sym
        self.clip_factor_a = torch.nn.Parameter(torch.ones((1,)), requires_grad=True)
        self.lac = lac
        self.enable = True
        self.act_dtype = act_dtype

    def forward(self, x):
        if self.bits == 16 or not self.enable:
            return x
        return self.fake_quant(x)

    def fake_quant(self, x):
        x_dtype = x.dtype
        if self.act_dtype == 'fp16' and x_dtype != torch.float16:
            # deploy-faithful carrier: mirror npu_kronecker_quant — the
            # ACTIVATION passes through an fp16 interface (rounding the
            # input representation to fp16), while the scale and quantize
            # math stay in high precision (the operator returns fp32
            # per-token scales; the deploy path consumes them as fp32).
            # Doing the division in fp16 instead would overflow x/scale
            # to Inf whenever one token's outlier exceeds ~516x another
            # element (bits=8), and round_ste(Inf) = Inf - Inf = NaN.
            x = x.to(torch.float16).to(x_dtype)
        scale, zero = self.get_scale_zero(x)
        if self.sym:
            return sym_quant_dequant(x, scale, self.q_max.to(x)).to(x_dtype)
        # asym branch kept for parity with experimental; flatquant is sym-only
        xq, scale, _ = self._asym_quant(x, scale, zero)
        return xq.to(x_dtype)

    def _asym_quant(self, x, scale, zero):
        scale = scale.to(x.device)
        zero = zero.to(x.device)
        q = torch.clamp(round_ste(x / scale) + zero, 0, self.q_max.to(x))
        return scale * (q - zero), scale, zero

    def get_scale_zero(self, x):
        q_max = self.q_max.to(x)
        init_shape = x.shape
        reshaped_x = x.reshape((-1, x.shape[-1]))
        xmax = reshaped_x.amax(1, keepdim=True)
        xmin = reshaped_x.amin(1, keepdim=True)
        tmp = torch.zeros_like(xmax)
        xmax = torch.maximum(xmax, tmp)
        xmin = torch.minimum(xmin, tmp)

        if self.lac:
            xmax = xmax * self.clip_factor_a
            xmin = xmin * self.clip_factor_a
        # symmetric: one scale per token, signed range from the wider side
        xmax = torch.maximum(torch.abs(xmin), xmax)
        tmp = xmax == 0
        scale = xmax / q_max
        scale[tmp] = 1
        scale = scale.repeat(1, reshaped_x.shape[-1]).reshape(init_shape)
        zero = torch.zeros_like(scale)
        return scale, zero


class WeightQuantizer(torch.nn.Module):
    """Per-channel (or per-tensor) weight fake quantizer."""

    def __init__(self, shape=(4096, 1)):
        super().__init__()
        self.register_buffer('maxq', torch.tensor(0))
        self.register_buffer('scale', torch.zeros(*shape))
        self.register_buffer('zero', torch.zeros(*shape))
        self.enable = True

    def configure(self, bits, perchannel=False, sym=True):
        self.bits = bits
        self.perchannel = perchannel
        self.sym = sym
        if sym:
            self.maxq = torch.tensor(2 ** (bits - 1) - 1)
        else:
            self.maxq = torch.tensor(2**bits - 1)

    def find_params(self, x):
        if self.bits == 16 or not self.enable:
            return
        dev = x.device
        self.maxq = self.maxq.to(dev)
        shape = x.shape
        if self.perchannel:
            x = x.flatten(1)
        else:
            x = x.flatten().unsqueeze(0)

        tmp = torch.zeros(x.shape[0], device=dev)
        xmin = torch.minimum(x.min(1)[0], tmp)
        xmax = torch.maximum(x.max(1)[0], tmp)

        if self.sym:
            xmax = torch.maximum(torch.abs(xmin), xmax).clamp(min=1e-5)
            self.scale = xmax / self.maxq
            self.zero = torch.zeros_like(self.scale)
        else:
            tmp = (xmin == 0) & (xmax == 0)
            xmin[tmp] = -1
            xmax[tmp] = +1
            self.scale = (xmax - xmin).clamp(min=1e-5) / self.maxq
            self.zero = torch.round(-xmin / self.scale)

        if self.perchannel:
            shape = [-1] + [1] * (len(shape) - 1)
            self.scale = self.scale.reshape(shape)
            self.zero = self.zero.reshape(shape)
        # per-tensor: keep the single scale as-is ((1,1) buffer); the div in
        # sym_quant broadcasts it across the whole weight, and the deploy-side
        # pack hands npu_quant_matmul exactly one scale value.

    def quantize(self, x, quantonly=False):
        x_dtype = x.dtype
        if self.enable and self.ready() and self.bits < 16:
            if self.sym:
                if quantonly:
                    return sym_quant(x, self.scale, self.maxq)
                return sym_quant_dequant(x, self.scale, self.maxq).to(x_dtype)
            xq, _, _ = self._asym(x)
            return xq.to(x_dtype)
        return x

    def _asym(self, x):
        scale = self.scale.to(x.device)
        zero = self.zero.to(x.device)
        q = torch.clamp(round_ste(x / scale) + zero, 0, self.maxq)
        return scale * (q - zero), scale, zero

    def forward(self, x):
        return self.quantize(x)

    def enabled(self):
        return self.maxq > 0

    def ready(self):
        return torch.all(self.scale != 0)


def set_quantizer_state(model, enable=True):
    """Toggle all flatquant quantizers (diagnostics/tests)."""
    for m in model.modules():
        if isinstance(m, (WeightQuantizer, ActivationQuantizer)):
            m.enable = enable
    return model
