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
from amct_pytorch.common.utils.vars import MXFP4_E2M1, FLOAT8_E4M3FN, HIFLOAT8
from amct_pytorch.quantization.dtypes.hifp_impl import hifloat4_fake_quant


# Dtype/bit-width -> weights_cfg mapping, defined ahead of GPTQ since __init__ uses it.
_INT_QUANT_TYPE_BY_BITS = {4: 'int4', 8: 'int8'}
_MXFP_QUANT_TYPE_BY_BITS = {4: 'mxfp4_e2m1', 8: 'mxfp8_e4m3fn'}
_HIF_QUANT_TYPE_BY_BITS = {4: 'hifloat4', 8: 'hifloat8'}
_MXFP_GROUP_SIZE = 32  # vars.py::GROUP_SIZE_SUPPORTED_MAP[MXFP4_E2M1 / MXFP8_E4M3FN] == [32]


def _build_gptq_weights_cfg(quant_dtype, w_bits):
    """Build the per-layer weights_cfg GPTQ uses for its own scale/offset computation.
    Mirrors the dtype string conventions in amct_pytorch/common/utils/vars.py and the
    group_size precedent in AutoRound._get_group_size (amct_pytorch/algorithms/quant/
    auto_round.py)."""
    if quant_dtype == 'int':
        quant_type = _INT_QUANT_TYPE_BY_BITS.get(w_bits)
        if quant_type is None:
            raise ValueError(f"gptq: unsupported int w_bits={w_bits!r} (expected 4 or 8)")
        return {'quant_type': quant_type, 'symmetric': True, 'strategy': 'channel'}
    if quant_dtype == 'mxfp':
        quant_type = _MXFP_QUANT_TYPE_BY_BITS.get(w_bits)
        if quant_type is None:
            raise ValueError(f"gptq: unsupported mxfp w_bits={w_bits!r} (expected 4 or 8)")
        return {
            'quant_type': quant_type, 'symmetric': True,
            'strategy': 'group', 'group_size': _MXFP_GROUP_SIZE,
        }
    # hif family: bit width picks hifloat4 (joint-64-column path, see is_hif4) or hifloat8
    # (a plain per-column dtype like float8_e4m3fn). Neither uses group_size.
    quant_type = _HIF_QUANT_TYPE_BY_BITS.get(w_bits)
    if quant_type is None:
        raise ValueError(f"gptq: unsupported hif w_bits={w_bits!r} (expected 4 or 8)")
    return {'quant_type': quant_type, 'symmetric': True, 'strategy': 'channel'}


@ALGO_REGISTRY.register(
    name="gptq",
    description="GPTQ Hessian-based single-pass weight quantizer; handles both the standard "
    "per-column int/mxfp dtypes and the HiFloat4 (hifloat4/hif4/hifp) block format in one "
    "class. Composed into WeightQuantizer like any other targets=(\"weight\",) algorithm and "
    "driven by BlockwiseSolver.",
    targets=("weight",),
    solver="gradient",
)
@ALGO_REGISTRY.register(
    name="higptq",
    description="Deprecated alias for 'gptq'.",
    targets=("weight",),
    solver="gradient",
)
class GPTQ(QuantAlgorithmBase):
    """
    GPTQ engine for both standard per-column dtypes (int8/int4/mxfp4_e2m1/mxfp8_e4m3fn/
    hifloat8/float8_e4m3fn) and HiFloat4, branching on self.is_hif4 wherever they diverge.
    HiFloat4 can't be quantized one column at a time -- its 3-level block scale is only
    meaningful over a real contiguous 64-element run -- so _get_opt_weight's inner loop
    processes HIFX4_GROUP_SIZE-wide groups jointly via hifloat4_fake_quant instead.

    Unlike WeightQuantizer's generic quant_obj (a single per-layer callable built from
    DTYPE_REGISTRY), GPTQ derives its own scale/offset from weights_cfg up front and reuses
    it across the per-column loop: calling the generic quant_obj once per column is not
    equivalent (int's scale is recomputed per call and degenerates to a no-op on a lone
    column; mxfp/hifloat4 need fixed-width groups a single column can't provide). quantize()'s
    quant_obj argument is therefore only used as a fallback when no Hessian has been built yet.

    No trainable parameters -- this is a closed-form (Cholesky/OBS) solve, not a
    gradient-trained transform, even though it is driven by BlockwiseSolver like the
    gradient-trained weight algorithms (AutoRound/LWC).
    """

    HIFX4_GROUP_SIZE = 64

    def __init__(self, args, w_bits):
        super().__init__()
        weights_cfg = _build_gptq_weights_cfg(args.quant_dtype, w_bits)
        self.quant_config = {'weights_cfg': weights_cfg}

        weight_shape = tuple(args.w_size)
        if len(weight_shape) != 2:
            raise ValueError(f"GPTQ currently expects 2D weight tensors, got shape {weight_shape}")
        self.rows, self.columns = weight_shape

        self.perc_damp = 0.01
        self.block_size = 128
        self.group_size = weights_cfg.get('group_size')
        self.wts_type = weights_cfg.get('quant_type')
        self.is_hif4 = self.wts_type == 'hifloat4'
        if self.is_hif4 and self.columns % self.HIFX4_GROUP_SIZE != 0:
            raise ValueError(
                "GPTQ: HiFloat4 requires the weight's column count to be a multiple "
                "of {}, got {}.".format(self.HIFX4_GROUP_SIZE, self.columns)
            )

        self.register_buffer("H", torch.zeros((self.columns, self.columns), dtype=torch.float32))
        self.register_buffer("nsamples", torch.tensor(0, dtype=torch.long))

    def forward(self, x):
        # Never actually reached: WeightQuantizer.algo_forward always routes an algo
        # exposing .quantize to the quantize_algo branch instead of calling forward().
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

        self.H = self.H.to(inp.device)
        current_nsamples = int(self.nsamples.item())
        total_nsamples = current_nsamples + batch_size
        if total_nsamples <= 0:
            return
        if current_nsamples > 0:
            self.H.mul_(current_nsamples / total_nsamples)
        self.nsamples.fill_(total_nsamples)

        inp = math.sqrt(2.0 / total_nsamples) * inp.float()
        self.H.add_(inp.matmul(inp.t()))

        if not torch.isfinite(self.H).all():
            raise RuntimeError("GPTQ's Hessian matrix has invalid value, inf or nan.")

    def export_ptq_params(self):
        if int(self.nsamples.item()) == 0:
            return {}
        return {
            "H": self.H.detach().cpu(),
            "nsamples": int(self.nsamples.item()),
        }

    def load_ptq_params(self, params):
        if "H" not in params:
            return
        self.H.copy_(params["H"].to(device=self.H.device, dtype=self.H.dtype))
        self.nsamples.fill_(int(params.get("nsamples", 0)))

    @torch.no_grad()
    def quantize(self, weight: torch.Tensor, quant_obj):
        """
        Function: run the GPTQ closed-form solve using the Hessian accumulated via
        observe_input. Falls back to a single whole-tensor quant_obj(weight) call (matching
        AutoRound's usage of quant_obj, amct_pytorch/algorithms/quant/auto_round.py) when no
        calibration data has been observed yet -- this is safe for int/mxfp/hif4/hif8 since
        it's a single call over the full (already 64-column-validated for hif4) tensor, unlike
        an incremental per-column call.
        """
        if int(self.nsamples.item()) == 0:
            return quant_obj(weight)
        return self._get_opt_weight(weight)

    def _get_opt_weight(self, weight):
        """
        Function: quantize weights column-by-column (or HIFX4_GROUP_SIZE-group-by-group for
        hif4) with second-order error compensation, following the GPTQ algorithm.
        """
        columns = self.columns
        weight = weight.clone().float()
        hessian = self.H.to(weight.device).clone()

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
                    column_index = perm[i1 + i]
                    w_q = self._cal_quant_weight(w, column_index, scale_w, offset_w)
                    err = (w - w_q) / d
                    weight_block[:, i:] -= err.unsqueeze(1).matmul(
                        hessian_inverse_block[i, i:].unsqueeze(0)
                    )
                    err_block[:, i] = err

            weight_column_sorted[:, i2:] -= err_block.matmul(hessian_inverse[i1:i2, i2:])

        return weight_column_sorted[:, invperm].to(weight.dtype)

    def _cal_quant_weight(self, w, column_index, scale_w, offset_w):
        if self.group_size:
            scale = (
                None
                if scale_w is None
                else scale_w[:, column_index // self.group_size].unsqueeze(-1)
            )
            offset = (
                None
                if offset_w is None
                else offset_w[:, column_index // self.group_size].unsqueeze(-1)
            )
            w_q = quant_dequant_tensor(
                w.unsqueeze(-1).repeat(1, 2),
                self.wts_type,
                scale,
                offset,
                self.group_size,
            )[:, 0]
        else:
            if self.wts_type in (FLOAT8_E4M3FN, HIFLOAT8):
                w_q = quant_dequant_tensor(
                    w.reshape(1, -1), self.wts_type, scale_w.reshape(-1), offset_w
                ).reshape(-1)
            else:
                w_q = quant_dequant_tensor(
                    w.reshape(-1, 1), self.wts_type, scale_w, offset_w
                ).reshape(-1)

        return w_q

    def _cal_quant_weight_group(self, w_group):
        """Quantize-dequantize a contiguous <=HIFX4_GROUP_SIZE-column group as one HiFloat4
        block. Only called when self.is_hif4."""
        return hifloat4_fake_quant(w_group, qdim=-1)

    def _cal_hessian_inverse(self, hessian_sorted, columns):
        damp = self.perc_damp * torch.mean(torch.diag(hessian_sorted))
        diag = torch.arange(columns, device=hessian_sorted.device)
        hessian_sorted[diag, diag] += damp
        device = hessian_sorted.device
        # npu unsupport torch.linalg
        hessian_sorted = hessian_sorted.cpu()
        hessian_sorted = torch.linalg.cholesky(hessian_sorted)
        hessian_sorted = torch.cholesky_inverse(hessian_sorted)
        hessian_inverse = torch.linalg.cholesky(hessian_sorted, upper=True)
        return hessian_inverse.to(device)

    def _cal_scale_offset_static(self, weight):
        # HiFloat4 and MXFP4 self-scale internally; no external scale/offset needed.
        if self.is_hif4 or self.wts_type in (MXFP4_E2M1,):
            return None, None

        weight_min, weight_max = get_weight_min_max_by_granularity(
            weight, self.quant_config
        )

        scale_w, offset_w = calculate_scale_offset(
            weight_max,
            weight_min,
            self.quant_config.get('weights_cfg').get('symmetric'),
            self.wts_type,
        )
        return scale_w, offset_w
