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

from amct_pytorch.algorithms.quant.base import QuantAlgorithmBase
from amct_pytorch.algorithms.registry_factory import ALGO_REGISTRY
from amct_pytorch.common.optimization import CalibrateBuildContext
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


# Dtype/bit-width -> weights_cfg mapping, defined ahead of GPTQuant since __init__ uses it.
_INT_QUANT_TYPE_BY_BITS = {4: 'int4', 8: 'int8'}
_MXFP_QUANT_TYPE_BY_BITS = {4: 'mxfp4_e2m1', 8: 'mxfp8_e4m3fn'}
_HIF_QUANT_TYPE_BY_BITS = {4: 'hifloat4', 8: 'hifloat8'}
_MXFP_GROUP_SIZE = 32  # vars.py::GROUP_SIZE_SUPPORTED_MAP[MXFP4_E2M1 / MXFP8_E4M3FN] == [32]


def _build_gptq_weights_cfg(quant_dtype, w_bits):
    """Build the per-layer weights_cfg passed into GPTQuant. Mirrors the dtype string
    conventions in amct_pytorch/common/utils/vars.py and the group_size precedent in
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
    # hif family: bit width picks hifloat4 (joint-64-column path, see is_hif4) or hifloat8
    # (a plain per-column dtype like float8_e4m3fn). Neither uses group_size.
    quant_type = _HIF_QUANT_TYPE_BY_BITS.get(w_bits)
    if quant_type is None:
        raise ValueError(f"gptq: unsupported hif w_bits={w_bits!r} (expected 4 or 8)")
    return {'quant_type': quant_type, 'symmetric': True, 'strategy': 'channel'}


@ALGO_REGISTRY.register(
    name="gptq",
    description="GPTQ Hessian-based single-pass calibrator; handles both the standard "
    "per-column int/mxfp dtypes and the HiFloat4 (hifloat4/hif4/hifp) block format in one "
    "class.",
    solver="calibrate",
)
@ALGO_REGISTRY.register(
    name="higptq",
    description="Deprecated alias for 'gptq'.",
    solver="calibrate",
)
class GPTQuant(QuantAlgorithmBase):
    """
    GPTQ engine for both standard per-column dtypes (int8/int4/mxfp4_e2m1/mxfp8_e4m3fn/
    hifloat8/float8_e4m3fn) and HiFloat4, branching on self.is_hif4 wherever they diverge.
    HiFloat4 can't be quantized one column at a time -- its 3-level block scale is only
    meaningful over a real contiguous 64-element run -- so get_opt_weight_and_quant_factor's
    inner loop processes HIFX4_GROUP_SIZE-wide groups jointly via hifloat4_fake_quant instead.

    __init__(quant_linear, layer_name, ctx) is itself the ALGO_REGISTRY["gptq"]/["higptq"]
    build contract CalibrationSolver expects, same as AutoRound/OmniQuant/FlatQuant register
    directly for their own (different) contract.
    """

    HIFX4_GROUP_SIZE = 64

    def __init__(self, quant_linear, layer_name, ctx: CalibrateBuildContext):
        """
        Function: init objective.
        Args:
        quant_linear: QuantLinear -- the wrapped nn.Linear (.linear) plus its already-resolved
            weight bit width (.w_bits).
        layer_name: quant_linear's name.
        ctx: CalibrateBuildContext(args, num_batches).
        """
        weights_cfg = _build_gptq_weights_cfg(ctx.args.quant_dtype, quant_linear.w_bits)
        quant_config = {
            'batch_num': ctx.num_batches,
            'weights_cfg': weights_cfg,
            'inputs_cfg': {'enable_quant': False},
        }
        ori_module = quant_linear.linear

        super().__init__()
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
        self.group_size = weights_cfg.get('group_size')
        self.wts_type = weights_cfg.get('quant_type')
        self.is_hif4 = self.wts_type == 'hifloat4'
        if self.is_hif4 and self.weight.shape[1] % self.HIFX4_GROUP_SIZE != 0:
            raise ValueError(
                "{}: HiFloat4 GPTQ requires the weight's column count to be a multiple "
                "of {}, got {}.".format(
                    layer_name, self.HIFX4_GROUP_SIZE, self.weight.shape[1]
                )
            )
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
            if self.is_hif4:
                # Re-quantize in original column order (not the Hessian-sorted order used
                # during OBS) so each HiF4 block spans real contiguous columns.
                self.cached_dq_w = hifloat4_fake_quant(self.weight.data, qdim=-1)
            else:
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

        # 0 calculate quantization factors (HiFloat4 has none -- see cal_scale_offset_static)
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

            if self.is_hif4:
                for g in range(0, count, self.HIFX4_GROUP_SIZE):
                    g2 = min(g + self.HIFX4_GROUP_SIZE, count)
                    w = weight_block[:, g:g2]
                    d = torch.diagonal(hessian_inverse_block[g:g2, g:g2])
                    w_q = self.cal_quant_weight_group(w)
                    err = (w - w_q) / d.unsqueeze(0)
                    weight_block[:, g:g2] = w_q
                    weight_block[:, g2:] -= err.matmul(hessian_inverse_block[g:g2, g2:])
                    err_block[:, g:g2] = err
            else:
                for i in range(count):
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

    def cal_quant_weight_group(self, w_group):
        """Quantize-dequantize a contiguous <=HIFX4_GROUP_SIZE-column group as one HiFloat4
        block. Only called when self.is_hif4."""
        return hifloat4_fake_quant(w_group, qdim=-1)

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
