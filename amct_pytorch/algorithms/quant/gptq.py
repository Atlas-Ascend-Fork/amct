# -*- coding: UTF-8 -*-
# Copyright (c) 2022 IST-DASLab. All rights reserved.
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
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

import math

import torch
import torch.nn as nn

from amct_pytorch.algorithms.registry_factory import ALGO_REGISTRY
from amct_pytorch.quantization.dtypes import DTYPE_REGISTRY


@ALGO_REGISTRY.register(
    name="gptq",
    description="Minimal GPTQ-style Hessian-aware weight quantization",
    targets=("weight",),
)
class GPTQ(nn.Module):
    """
    Minimal GPTQ core adapted to the current framework.

    This implementation keeps the essential GPTQ logic:
    1. collect input Hessian statistics during an observe pass
    2. quantize weights column-by-column with second-order error compensation

    It does not port the original repository's full model wrappers or custom
    quantizer classes. Instead, it reuses the current framework's downstream
    `quant_obj` callable for the actual quantization step.
    """

    def __init__(self, args, w_bits):
        super().__init__()
        self.args = args
        self.blocksize = int(getattr(args, "gptq_blocksize", 128))
        self.percdamp = float(getattr(args, "gptq_percdamp", 0.01))
        self.actorder = bool(getattr(args, "gptq_actorder", False))
        self.is_observe = False

        weight_shape = tuple(args.w_size)
        if len(weight_shape) != 2:
            raise ValueError(f"GPTQ currently expects 2D weight tensors, got shape {weight_shape}")
        self.rows = weight_shape[0]
        self.columns = weight_shape[1]

        self.register_buffer("H", torch.zeros((self.columns, self.columns), dtype=torch.float32))
        self.register_buffer("nsamples", torch.tensor(0, dtype=torch.long))
        self.register_buffer("_cached_qweight", torch.empty(0), persistent=False)
        self.quant_obj = DTYPE_REGISTRY.get(args.quant_dtype)(bits=w_bits)
        if hasattr(self.quant_obj, "group_size"):
            self.quant_obj.group_size = int(getattr(args, "group_size", -1))
        self._observed_weight = None

    def observe_input(self, inp: torch.Tensor, weight: torch.Tensor | None = None):
        if not self.is_observe:
            return
        if weight is not None:
            self._observed_weight = weight.detach()
        if inp.ndim == 2:
            inp = inp.unsqueeze(0)

        tmp = inp.shape[0]
        if inp.ndim == 3:
            inp = inp.reshape(-1, inp.shape[-1])

        inp = inp.t().float()
        current_nsamples = int(self.nsamples.item())
        total_nsamples = current_nsamples + tmp
        if total_nsamples <= 0:
            return

        if current_nsamples > 0:
            self.H.mul_(current_nsamples / total_nsamples)
        self.nsamples.fill_(total_nsamples)
        inp = math.sqrt(2.0 / total_nsamples) * inp
        self.H.add_(inp.matmul(inp.t()))
        self._cached_qweight = torch.empty(0, device=self.H.device)

    def trainable_params(self):
        return []

    def _group_scale_for(self, weight, quant_obj):
        """Per-group symmetric scale (mirrors quantize); (None, None) if not group-int4."""
        bits = getattr(quant_obj, "bits", None)
        gsize = getattr(quant_obj, "group_size", -1)
        if bits is None or bits >= 16 or gsize is None or gsize <= 0 or self.columns % gsize != 0:
            return None, None
        qmax = 2 ** (bits - 1) - 1
        n_groups = self.columns // gsize
        scale = (weight.detach().float().abs()
                 .reshape(self.rows, n_groups, gsize).amax(dim=2) / qmax).clamp(min=1e-9)
        return scale, gsize

    def export_ptq_params(self):
        # Only the GPTQ-corrected quantized weight is needed downstream; the
        # Hessian H is a calibration-time intermediate (eval/deploy reuse the
        # cached qweight and never recompute Cholesky), so it is NOT persisted.
        # For group-wise int4 the weight is stored as int8 levels + per-group
        # scale (q_int * scale == the fp32 dequant, bit-exact, ~4x smaller).
        if int(self.nsamples.item()) == 0 or self._observed_weight is None:
            return {}
        qweight = self.quantize(self._observed_weight, self.quant_obj)
        scale, gsize = self._group_scale_for(self._observed_weight, self.quant_obj)
        if scale is not None:
            n_groups = self.columns // gsize
            q_int = torch.round(qweight.detach().float()
                                .reshape(self.rows, n_groups, gsize) / scale.unsqueeze(-1))
            return {
                "qweight_int": q_int.to(torch.int8).cpu(),
                "qweight_scale": scale.detach().cpu(),
                "group_size": int(gsize),
            }
        return {"qweight": qweight.detach().cpu()}

    def load_ptq_params(self, params):
        # Reconstruct the cached fp32 dequant weight; quantize() then short-circuits.
        if "qweight_int" in params:
            q = params["qweight_int"].to(device=self.H.device).float()
            scale = params["qweight_scale"].to(device=self.H.device)
            gsize = int(params["group_size"])
            n_groups = self.columns // gsize
            self._cached_qweight = (q.reshape(self.rows, n_groups, gsize)
                                    * scale.unsqueeze(-1)).reshape(self.rows, self.columns)
        elif "qweight" in params:
            self._cached_qweight = params["qweight"].to(device=self.H.device)
        else:
            self._cached_qweight = torch.empty(0, device=self.H.device)

    @torch.no_grad()
    def quantize(self, weight: torch.Tensor, quant_obj):
        if self._cached_qweight.numel() != 0 and tuple(self._cached_qweight.shape) == tuple(weight.shape):
            return self._cached_qweight.to(device=weight.device, dtype=weight.dtype)

        if int(self.nsamples.item()) == 0:
            return quant_obj(weight)

        # GPTQ freezes a fixed per-group/block quantizer from the original
        # weight (computed once) and applies it column by column; the Hessian
        # error compensation below is dtype-agnostic. The grid itself (int
        # symmetric / mxfp shared-exponent) is owned by the dtype, which
        # exposes freeze_column_quantizer + quant_column.
        bits = getattr(quant_obj, "bits", None)
        real_quant = bits is not None and bits < 16
        has_col_api = (hasattr(quant_obj, "freeze_column_quantizer")
                       and hasattr(quant_obj, "quant_column"))
        if real_quant and not has_col_api:
            raise NotImplementedError(
                f"GPTQ needs the dtype to implement freeze_column_quantizer/"
                f"quant_column for faithful column quantization; "
                f"{type(quant_obj).__name__} does not."
            )
        qstate = quant_obj.freeze_column_quantizer(weight) if has_col_api else None

        W = weight.detach().float().clone()
        H = self.H.to(device=weight.device, dtype=torch.float32).clone()

        dead = torch.diag(H) == 0
        if torch.any(dead):
            H[dead, dead] = 1
            W[:, dead] = 0

        perm = None
        invperm = None
        if self.actorder:
            perm = torch.argsort(torch.diag(H), descending=True)
            W = W[:, perm]
            H = H[perm][:, perm]
            invperm = torch.argsort(perm)

        damp = self.percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=weight.device)
        H[diag, diag] += damp
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H

        Q = torch.zeros_like(W)

        for i1 in range(0, self.columns, self.blocksize):
            i2 = min(i1 + self.blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Err1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]
                if qstate is not None:
                    q = quant_obj.quant_column(w, i1 + i, qstate).to(W1)
                else:
                    q = quant_obj(w.unsqueeze(1)).flatten().to(W1)
                Q[:, i1 + i] = q

                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            if i2 < self.columns:
                W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

        if self.actorder and invperm is not None:
            Q = Q[:, invperm]

        Q = Q.reshape(weight.shape).to(dtype=weight.dtype)
        self._cached_qweight = Q.detach().to(device=self.H.device, dtype=weight.dtype)
        return Q
