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
import torch
import torch.nn.functional as F

from amct_pytorch.algorithms.registry_factory import ALGO_REGISTRY
from amct_pytorch.classic.quantize_op.gptq_module import GPTQuant
from amct_pytorch.quantization.dtypes.hifp_impl import hifloat4_fake_quant


@ALGO_REGISTRY.register(name="higptq", description="HiGPTQ Hessian-based single-pass calibrator (HiFloat4)",
                         solver="calibrate")
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
