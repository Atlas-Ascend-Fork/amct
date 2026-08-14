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
import torch.nn.functional as F

from amct_pytorch.algorithms.registry_factory import ALGO_REGISTRY
from amct_pytorch.common.optimization import CalibrateBuildContext
from amct_pytorch.quantize_op.base_quant_module import BaseQuantizeModule
from amct_pytorch.common.utils.data_utils import check_linear_input_dim
from amct_pytorch.classic.quantize_op.utils import (
    calculate_scale_offset,
    get_weight_min_max_by_granularity,
)
from amct_pytorch.common.utils.quant_util import (
    quant_dequant_tensor,
    quant_dequant_weight,
)
from amct_pytorch.common.utils.vars import MXFP4_E2M1, FLOAT8_E4M3FN, HIFLOAT8
from amct_pytorch.common.utils.log import LOGGER
from amct_pytorch.quantization.dtypes.hifp_impl import hifloat4_fake_quant


class GPTQuant(BaseQuantizeModule):
    """
    Function: Customized torch.nn.Module of the GPTQuant class.
    APIs: forward.

    This is a standalone copy of amct_pytorch/classic/quantize_op/gptq_module.py's
    GPTQuant, duplicated here (not imported/subclassed) so that amct_pytorch/algorithms/
    (the new ALGO_REGISTRY/args.algos-driven pipeline) does not depend on amct_pytorch/classic/
    (the older config-dict-driven pipeline). Any bugfix to the Hessian/OBS math below should
    be applied to both copies.
    """

    def __init__(self, ori_module, layer_name, quant_config):
        """
        Function: init objective.
        Args:
        ori_module: torch module. Linear.
        layer_name: ori_module's name.
        quant_config: quantization parameters.
        """
        super().__init__(ori_module, layer_name, quant_config)
        self.ori_module_type = type(ori_module).__name__
        self.weight = ori_module.weight
        if not torch.isfinite(self.weight.data).all():
            raise ValueError(
                "{}'s weight has invalid value, inf or nan.".format(layer_name)
            )
        self.bias = ori_module.bias

        self.layer_name = layer_name
        self.quant_config = quant_config

        self.cur_batch = 0
        self.nsamples = 0
        self.perc_damp = 0.01
        self.block_size = 128
        self.hessian = None
        self.group_size = self.quant_config.get('weights_cfg').get('group_size')
        self.wts_type = self.quant_config.get('weights_cfg').get('quant_type')
        enable_quant = quant_config.get('inputs_cfg').get('enable_quant')
        if enable_quant is None or enable_quant:
            self.act_granularity = quant_config.get('inputs_cfg').get('strategy')

    @torch.no_grad()
    def forward(self, inputs):
        """
        Function: GPTQuant forward function.
        Args:
        inputs: data used for calibration in torch.tensor.
        """
        check_linear_input_dim(inputs)
        inputs = inputs.to(self.weight.device)

        # When batch_num is reached, h stops updating
        if self.cur_batch < self.quant_config.get('batch_num'):
            self.update_hessian(inputs)
            self.cur_batch += 1

        # do gptq
        if self.cur_batch == self.quant_config.get('batch_num'):
            self.cur_batch += 1
            optimized_weight, self.scale_w, self.offset_w = (
                self.get_opt_weight_and_quant_factor()
            )
            self.weight.data = optimized_weight
            LOGGER.logd(
                "Calculate gptq quant params of layer '{}' success!".format(
                    self.layer_name
                ),
                type(self).__name__,
            )
        if self.cur_batch > self.quant_config.get('batch_num'):
            return self.fake_quant_forward(inputs)
        return F.linear(inputs, self.weight, self.bias)

    @torch.no_grad()
    def fake_quant_forward(self, inputs):
        if not getattr(self, 'fake_quant_cache_ready', False):
            self.cached_dq_w = quant_dequant_weight(
                self.weight.data,
                self.wts_type,
                self.scale_w,
                self.offset_w,
                self.group_size,
            )
            self.fake_quant_cache_ready = True
        return F.linear(inputs, self.cached_dq_w, self.bias)

    def get_opt_weight_and_quant_factor(self):
        """
        Get optimized weights and quantization factors.

        Returns:
        tuple: Optimized weights, scale and offset for quantization.
        """
        columns = self.weight.shape[1]
        # Do not alter the original weights
        weight = self.weight.clone().float()

        self.hessian = self.hessian.to(weight.device)
        dead = torch.diag(self.hessian) == 0
        self.hessian[dead, dead] = 1
        weight[:, dead] = 0

        perm = torch.argsort(torch.diag(self.hessian), descending=True)
        invperm = torch.argsort(perm)

        # 0 calculate quantization factors
        scale_w, offset_w = self.cal_scale_offset_static(weight)

        # 1 Sort based on the size of the diagonal elements of the Hessian matrix.
        weight_column_sorted = weight[:, perm]
        hessian_sorted = self.hessian[perm][:, perm]

        # 2 Calculate the inverse of the Hessian matrix
        hessian_inverse = self.cal_hessian_inverse(hessian_sorted, columns)
        del hessian_sorted, self.hessian
        import torch_npu

        if torch_npu.npu.is_available():
            torch_npu.npu.empty_cache()

        # 3 Optimize weights according to the granularity of blocks.
        for i1 in range(0, columns, self.block_size):
            i2 = min(i1 + self.block_size, columns)
            count = i2 - i1

            weight_block = weight_column_sorted[:, i1:i2]
            err_block = torch.zeros_like(weight_block)
            hessian_inverse_block = hessian_inverse[i1:i2, i1:i2]

            for i in range(count):
                # Each column shares the value of a Hessian matrix.
                w = weight_block[:, i]
                d = hessian_inverse_block[i, i]
                column_index = perm[i1 + i]
                w_q = self.cal_quant_weight(w, column_index, scale_w, offset_w)
                err = (w - w_q) / d
                weight_block[:, i:] -= err.unsqueeze(1).matmul(
                    hessian_inverse_block[i, i:].unsqueeze(0)
                )
                err_block[:, i] = err

            weight_column_sorted[:, i2:] -= err_block.matmul(
                hessian_inverse[i1:i2, i2:]
            )

        # 4 Restore the sorting of the weight column
        weight = weight_column_sorted[:, invperm].to(self.weight.dtype)

        return weight, scale_w, offset_w

    def cal_quant_weight(self, w, column_index, scale_w, offset_w):
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

    def cal_hessian_inverse(self, hessian_sorted, columns):
        damp = self.perc_damp * torch.mean(torch.diag(hessian_sorted))
        diag = torch.arange(columns, device=self.weight.device)
        hessian_sorted[diag, diag] += damp
        # npu unspport torch.linalg
        hessian_sorted = hessian_sorted.cpu()
        hessian_sorted = torch.linalg.cholesky(hessian_sorted)
        hessian_sorted = torch.cholesky_inverse(hessian_sorted)
        hessian_inverse = torch.linalg.cholesky(hessian_sorted, upper=True)
        hessian_inverse = hessian_inverse.to(self.weight.device)
        return hessian_inverse

    def cal_scale_offset_static(self, weight):
        if self.wts_type in (MXFP4_E2M1,):
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

    def update_hessian(self, input_data):
        """
        Update the Hessian matrix.

        Args:
            input_data (Tensor): Input data, data used for calibration.
        """
        if len(input_data.shape) == 2:
            input_data = input_data.unsqueeze(0)
        batch_size = input_data.shape[0]

        # input_data shape is (-1, cin)
        input_data = input_data.reshape((-1, input_data.shape[-1]))
        input_data = input_data.t()
        if self.hessian is None:
            self.hessian = torch.zeros((self.weight.shape[1], self.weight.shape[1]))
        self.hessian = self.hessian.to(input_data.device)
        self.hessian *= self.nsamples / (self.nsamples + batch_size)
        self.nsamples += batch_size

        input_data = math.sqrt(2 / self.nsamples) * input_data.float()
        self.hessian += input_data.matmul(input_data.t())

        if not torch.isfinite(self.hessian).all():
            raise RuntimeError(
                "{}'s hessian matrix has invalid value, inf or nan.".format(
                    self.layer_name
                )
            )
        self.hessian = self.hessian.cpu()


class HiGPTQuant(GPTQuant):
    """
    Function: GPTQ variant for the HiFloat4 (hifx4) block floating-point format.
    APIs: forward (inherited from GPTQuant).

    HiFloat4 differs from GPTQuant's other supported dtypes in that it cannot be quantized
    one column at a time: its 3-level hierarchical scale (L1 E6M2 over 64 elements, L2/L3
    micro-exponents over 8/4) is only meaningful over a real contiguous 64-element run, and
    is computed internally by ``hifloat4_fake_quant`` -- there is no external scale/offset to
    precompute the way GPTQuant does for int/hifloat8/mxfp4. So this class overrides:
      1. The Cholesky/OBS loop to process columns in groups of 64 (HIFX4_GROUP_SIZE) instead
         of one at a time, quantizing each group jointly.
      2. The quant-dequant call itself, to use ``hifloat4_fake_quant`` instead of
         ``quant_dequant_tensor``.
    Everything else (Hessian accumulation, Cholesky-inverse computation, the calibrate/optimize/
    fake-quant state machine in ``forward``, the outer block_size=128 chunking and lazy
    cross-block error propagation) is inherited unchanged from GPTQuant.
    """

    HIFX4_GROUP_SIZE = 64

    def cal_scale_offset_static(self, _weight):
        """HiFloat4 self-scales per 64-element block inside hifloat4_fake_quant; no external
        scale/offset is computed (same precedent as GPTQuant's MXFP4_E2M1 special case)."""
        return None, None

    def get_opt_weight_and_quant_factor(self):
        """
        Get optimized weights and quantization factors.

        Same Hessian-diagonal sort, Cholesky-inverse, and block/lazy-batch-update structure as
        GPTQuant.get_opt_weight_and_quant_factor, but the per-column loop is replaced with a
        per-group-of-64 loop (see cal_quant_weight_group).

        Returns:
        tuple: Optimized weights, scale and offset for quantization (scale/offset are None).
        """
        columns = self.weight.shape[1]
        # Do not alter the original weights
        weight = self.weight.clone().float()

        self.hessian = self.hessian.to(weight.device)
        dead = torch.diag(self.hessian) == 0
        self.hessian[dead, dead] = 1
        weight[:, dead] = 0

        perm = torch.argsort(torch.diag(self.hessian), descending=True)
        invperm = torch.argsort(perm)

        # 0 HiFloat4 has no external quantization factors.
        scale_w, offset_w = self.cal_scale_offset_static(weight)

        # 1 Sort based on the size of the diagonal elements of the Hessian matrix.
        weight_column_sorted = weight[:, perm]
        hessian_sorted = self.hessian[perm][:, perm]

        # 2 Calculate the inverse of the Hessian matrix
        hessian_inverse = self.cal_hessian_inverse(hessian_sorted, columns)
        del hessian_sorted, self.hessian
        import torch_npu
        if torch_npu.npu.is_available():
            torch_npu.npu.empty_cache()

        # 3 Optimize weights according to the granularity of blocks.
        for i1 in range(0, columns, self.block_size):
            i2 = min(i1 + self.block_size, columns)
            count = i2 - i1

            weight_block = weight_column_sorted[:, i1:i2]
            err_block = torch.zeros_like(weight_block)
            hessian_inverse_block = hessian_inverse[i1:i2, i1:i2]

            for g in range(0, count, self.HIFX4_GROUP_SIZE):
                # Each group of (up to) 64 columns shares one HiF4 block scale, so it must be
                # quantized jointly -- quantizing one column at a time would not reproduce the
                # format's real hierarchical block scaling.
                g2 = min(g + self.HIFX4_GROUP_SIZE, count)
                w = weight_block[:, g:g2]
                # Each column's residual is still normalized by its own Hessian-inverse
                # diagonal entry, same as stock GPTQ -- only the quantization step is joint.
                d = torch.diagonal(hessian_inverse_block[g:g2, g:g2])
                w_q = self.cal_quant_weight_group(w)
                err = (w - w_q) / d.unsqueeze(0)
                # Stock GPTQ's per-column update is `weight_block[:, i:] -= err ⊗
                # hessian_inverse_block[i, i:]`, an *inclusive* slice starting at the just-
                # quantized column -- since hessian_inverse_block[i, i] == d, that update sets
                # column i to exactly w_q as a side effect. A joint group can't be quantized
                # sequentially column-by-column (that would break HiF4's shared block scale),
                # so the group's own columns are set to w_q directly here, and only the
                # remaining not-yet-processed columns receive the propagated residual. This
                # reduces exactly to the stock per-column formula when HIFX4_GROUP_SIZE == 1.
                weight_block[:, g:g2] = w_q
                weight_block[:, g2:] -= err.matmul(hessian_inverse_block[g:g2, g2:])
                err_block[:, g:g2] = err

            weight_column_sorted[:, i2:] -= err_block.matmul(hessian_inverse[i1:i2, i2:])

        # 4 Restore the sorting of the weight column
        weight = weight_column_sorted[:, invperm].to(self.weight.dtype)

        return weight, scale_w, offset_w

    def cal_quant_weight_group(self, w_group):
        """
        Function: quantize-dequantize a contiguous group of (up to) 64 columns as one HiFloat4
        block.
        Args:
        w_group: torch.tensor, shape [Cout, gsize]. A contiguous run of columns (gsize <= 64).
        Returns:
        torch.tensor, same shape as w_group.
        """
        return hifloat4_fake_quant(w_group, qdim=-1)

    @torch.no_grad()
    def fake_quant_forward(self, inputs):
        if not getattr(self, 'fake_quant_cache_ready', False):
            # Re-quantize the final, unpermuted [Cout, Cin] weight so every HiF4 block spans a
            # real contiguous run of original input channels (the in-loop grouping above runs in
            # Hessian-sorted order, which is only used to drive OBS error compensation).
            self.cached_dq_w = hifloat4_fake_quant(self.weight.data, qdim=-1)
            self.fake_quant_cache_ready = True
        return F.linear(inputs, self.cached_dq_w, self.bias)


# Dtype/bit-width -> weights_cfg mapping, moved here from CalibrationSolver so the solver
# doesn't need to know GPTQ's config shape (weights_cfg: quant_type/symmetric/strategy/
# group_size) -- see CalibrateBuildContext (amct_pytorch/common/optimization/__init__.py)
# for the generic build contract this module conforms to.
_INT_QUANT_TYPE_BY_BITS = {4: 'int4', 8: 'int8'}
_MXFP_QUANT_TYPE_BY_BITS = {4: 'mxfp4_e2m1', 8: 'mxfp8_e4m3fn'}
_MXFP_GROUP_SIZE = 32  # vars.py::GROUP_SIZE_SUPPORTED_MAP[MXFP4_E2M1 / MXFP8_E4M3FN] == [32]


def _build_gptq_weights_cfg(quant_dtype, w_bits):
    """Build the per-layer weights_cfg passed into GPTQuant/HiGPTQuant. Mirrors the dtype
    string conventions in amct_pytorch/common/utils/vars.py and the group_size precedent in
    AutoRound._get_group_size (amct_pytorch/algorithms/quant/auto_round.py)."""
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
    # hif family ("hifp", or any future "hif*" quant_dtype): hifloat4 is the only supported
    # width today regardless of w_bits; HiGPTQuant.cal_scale_offset_static ignores group_size.
    return {'quant_type': 'hifloat4', 'symmetric': True, 'strategy': 'channel'}


def build_gptq(quant_linear, layer_name, ctx: CalibrateBuildContext):
    """
    Function: dtype-dispatching factory registered under ALGO_REGISTRY["gptq"]
    (and its backward-compat alias ALGO_REGISTRY["higptq"]), conforming to
    CalibrationSolver's generic calibrate-algorithm build contract:
    build(quant_linear, layer_name, ctx) -> object exposing .forward(inputs).
    Args:
    quant_linear: QuantLinear -- the wrapped nn.Linear (.linear) plus its already-resolved
        weight bit width (.w_bits).
    layer_name: quant_linear's name.
    ctx: CalibrateBuildContext(args, num_batches).
    Returns:
    HiGPTQuant instance if the resolved quant_type is a hif-family dtype (e.g. "hifloat4",
    matched by a "hif" prefix), else a plain GPTQuant instance (int4/int8/mxfp4_e2m1/
    mxfp8_e4m3fn).
    """
    weights_cfg = _build_gptq_weights_cfg(ctx.args.quant_dtype, quant_linear.w_bits)
    quant_config = {
        'batch_num': ctx.num_batches,
        'weights_cfg': weights_cfg,
        'inputs_cfg': {'enable_quant': False},
    }
    cls = HiGPTQuant if weights_cfg['quant_type'].startswith('hif') else GPTQuant
    return cls(quant_linear.linear, layer_name, quant_config)


ALGO_REGISTRY.register(
    build_gptq,
    name="gptq",
    description="GPTQ Hessian-based single-pass calibrator; dispatches to HiGPTQuant for "
    "hif-family weight dtypes (hifloat4/hif4/hifp) or plain GPTQuant for int/mxfp.",
    solver="calibrate",
)
ALGO_REGISTRY.register(
    build_gptq,
    name="higptq",
    description="Deprecated alias for 'gptq'.",
    solver="calibrate",
)
