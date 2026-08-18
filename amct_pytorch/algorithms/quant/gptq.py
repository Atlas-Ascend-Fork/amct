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
import math

import torch

from amct_pytorch.algorithms.quant.base import QuantAlgorithmBase
from amct_pytorch.algorithms.registry_factory import ALGO_REGISTRY
from amct_pytorch.common.utils.data_utils import check_linear_input_dim
from amct_pytorch.classic.quantize_op.utils import (
    calculate_scale_offset,
    get_weight_min_max_by_granularity,
)
from amct_pytorch.common.utils.quant_util import quant_dequant_tensor
from amct_pytorch.quantization.dtypes.hifp_impl import hifloat4_fake_quant


# int (4/8-bit) and HiFloat4 are supported for now
_INT_QUANT_TYPE_BY_BITS = {4: 'int4', 8: 'int8'}


@ALGO_REGISTRY.register(
    name="gptq",
    description="GPTQ Hessian-based single-pass weight quantizer; supports int4/int8 and "
    "HiFloat4 (mxfp/hifloat8 not yet supported). Composed into WeightQuantizer like any "
    "other targets=(\"weight\",) algorithm and driven by BlockwiseSolver.",
    targets=("weight",),
)
class GPTQ(QuantAlgorithmBase):
    """
    GPTQ: Hessian-based closed-form weight quantizer (int4/int8, HiFloat4)
    """

    HIFX4_GROUP_SIZE = 64

    def __init__(self, args, w_bits):
        super().__init__()
        if args.quant_dtype == 'int':
            quant_type = _INT_QUANT_TYPE_BY_BITS[w_bits]
        elif args.quant_dtype == 'hifp':
            if w_bits != 4:
                raise ValueError(
                    f"gptq: only HiFloat4 (w_bits=4) is currently supported for "
                    f"quant_dtype='hifp', got w_bits={w_bits!r}"
                )
            quant_type = 'hifloat4'
        else:
            raise ValueError(
                f"gptq: unsupported quant_dtype={args.quant_dtype!r} (only 'int' and 'hifp' "
                "(HiFloat4) are currently supported)"
            )
        weight_shape = tuple(args.w_size)
        if len(weight_shape) != 2:
            raise ValueError(
                f"GPTQ currently expects 2D weight tensors, got shape {weight_shape}"
            )
        self.rows, self.columns = weight_shape

        self.perc_damp = 0.01
        self.block_size = 128
        self.wts_type = quant_type
        self.is_hif4 = self.wts_type == 'hifloat4'
        if self.is_hif4 and self.columns % self.HIFX4_GROUP_SIZE != 0:
            raise ValueError(
                "GPTQ: HiFloat4 requires the weight's column count to be a multiple "
                "of {}, got {}.".format(self.HIFX4_GROUP_SIZE, self.columns)
            )

        self.register_buffer(
            "hessian", torch.zeros((self.columns, self.columns), dtype=torch.float32)
        )
        self.register_buffer("nsamples", torch.tensor(0, dtype=torch.long))

    def forward(self, x):
        # Required by QuantAlgorithmBase (abstract); not actually used since GPTQ is
        # dispatched via its quantize() hook instead.
        return x

    def trainable_params(self):
        return []

    def observe_input(self, inp: torch.Tensor, weight: torch.Tensor = None):
        """
        Accumulate the input Hessian. Called unconditionally on every QuantLinear forward
        (see WeightQuantizer.observe_input) -- self.is_observe (toggled in lockstep with the
        rest of the module tree by set_model_to_observe) is what actually gates real work.
        """
        if not self.is_observe:
            return
        check_linear_input_dim(inp)

        if inp.ndim == 2:
            inp = inp.unsqueeze(0)
        batch_size = inp.shape[0]
        inp = inp.reshape(-1, inp.shape[-1]).t()

        self.hessian = self.hessian.to(inp.device)
        current_nsamples = int(self.nsamples.item())
        total_nsamples = current_nsamples + batch_size
        if total_nsamples <= 0:
            return
        if current_nsamples > 0:
            self.hessian.mul_(current_nsamples / total_nsamples)
        self.nsamples.fill_(total_nsamples)

        inp = math.sqrt(2.0 / total_nsamples) * inp.float()
        self.hessian.add_(inp.matmul(inp.t()))

        if not torch.isfinite(self.hessian).all():
            raise RuntimeError("GPTQ's Hessian matrix has invalid value, inf or nan.")

    def export_ptq_params(self):
        if int(self.nsamples.item()) == 0:
            return {}
        return {
            "H": self.hessian.detach().cpu(),
            "nsamples": int(self.nsamples.item()),
        }

    def load_ptq_params(self, params):
        if "H" not in params:
            return
        self.hessian.copy_(
            params["H"].to(device=self.hessian.device, dtype=self.hessian.dtype)
        )
        self.nsamples.fill_(int(params.get("nsamples", 0)))

    @torch.no_grad()
    def quantize(self, weight: torch.Tensor, quant_obj):
        """
        Function: run the GPTQ closed-form solve (column-by-column, or HIFX4_GROUP_SIZE-group-
        by-group for hif4, with second-order error compensation) using the Hessian accumulated
        via observe_input. Falls back to a single whole-tensor quant_obj(weight) call (matching
        AutoRound's usage of quant_obj, amct_pytorch/algorithms/quant/auto_round.py) when no
        calibration data has been observed yet -- this is safe for int/hif4 since it's a
        single call over the full (already 64-column-validated for hif4) tensor, unlike an
        incremental per-column call.
        """
        if int(self.nsamples.item()) == 0:
            return quant_obj(weight)

        columns = self.columns
        weight = weight.clone().float()
        hessian = self.hessian.to(weight.device).clone()

        dead = torch.diag(hessian) == 0
        hessian[dead, dead] = 1
        weight[:, dead] = 0

        perm = torch.argsort(torch.diag(hessian), descending=True)
        invperm = torch.argsort(perm)

        scale_w, offset_w = self._cal_scale_offset_static(weight)

        weight_column_sorted = weight[:, perm]
        hessian_sorted = hessian[perm][:, perm]

        hessian_inverse = self._cal_hessian_inverse(hessian_sorted, columns)
        del hessian_sorted, hessian

        for i1 in range(0, columns, self.block_size):
            i2 = min(i1 + self.block_size, columns)
            count = i2 - i1

            weight_block = weight_column_sorted[:, i1:i2]
            err_block = torch.zeros_like(weight_block)
            hessian_inverse_block = hessian_inverse[i1:i2, i1:i2]

            if self.is_hif4:
                for g in range(0, count, self.HIFX4_GROUP_SIZE):
                    g2 = min(g + self.HIFX4_GROUP_SIZE, count)
                    w = weight_block[:, g:g2]
                    d = torch.diagonal(hessian_inverse_block[g:g2, g:g2])
                    w_q = self._cal_quant_weight_group(w)
                    err = (w - w_q) / d.unsqueeze(0)
                    weight_block[:, g:g2] = w_q
                    weight_block[:, g2:] -= err.matmul(hessian_inverse_block[g:g2, g2:])
                    err_block[:, g:g2] = err
            else:
                for i in range(count):
                    w = weight_block[:, i]
                    d = hessian_inverse_block[i, i]
                    w_q = self._cal_quant_weight(w, scale_w, offset_w)
                    err = (w - w_q) / d
                    weight_block[:, i:] -= err.unsqueeze(1).matmul(
                        hessian_inverse_block[i, i:].unsqueeze(0)
                    )
                    err_block[:, i] = err

            weight_column_sorted[:, i2:] -= err_block.matmul(
                hessian_inverse[i1:i2, i2:]
            )

        return weight_column_sorted[:, invperm].to(weight.dtype)

    def _cal_quant_weight(self, w, scale_w, offset_w):
        return quant_dequant_tensor(
            w.reshape(-1, 1), self.wts_type, scale_w, offset_w
        ).reshape(-1)

    def _cal_quant_weight_group(self, w_group):
        """Quantize-dequantize a contiguous <=HIFX4_GROUP_SIZE-column group as one HiFloat4
        block. Only called when self.is_hif4."""
        return hifloat4_fake_quant(w_group, qdim=-1)

    def _cal_hessian_inverse(self, hessian_sorted, columns):
        damp = self.perc_damp * torch.mean(torch.diag(hessian_sorted))
        diag = torch.arange(columns, device=hessian_sorted.device)
        hessian_sorted[diag, diag] += damp
        device = hessian_sorted.device
        # npu unsupported torch.linalg
        hessian_sorted = hessian_sorted.cpu()
        hessian_sorted = torch.linalg.cholesky(hessian_sorted)
        hessian_sorted = torch.cholesky_inverse(hessian_sorted)
        hessian_inverse = torch.linalg.cholesky(hessian_sorted, upper=True)
        return hessian_inverse.to(device)

    def _cal_scale_offset_static(self, weight):
        # HiFloat4 self-scales internally; no external scale/offset needed.
        if self.is_hif4:
            return None, None

        # get_weight_min_max_by_granularity/calculate_scale_offset only read strategy and
        # symmetric off this dict -- both are fixed (int is always symmetric/per-channel here),
        # so build it inline rather than storing it as an instance attribute.
        weight_min, weight_max = get_weight_min_max_by_granularity(
            weight, {'weights_cfg': {'strategy': 'channel'}}
        )

        scale_w, offset_w = calculate_scale_offset(
            weight_max,
            weight_min,
            True,
            self.wts_type,
        )
        return scale_w, offset_w
