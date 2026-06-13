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
from torch import Tensor
from amct_pytorch.quantization.dtypes import DTYPE_REGISTRY
from amct_pytorch.quantization.dtypes.mxfp_impl import (
    f32_to_f4_unpacked, pack_uint4, quantize_elewise, shared_exponents,
)


@DTYPE_REGISTRY.register(name="mxfp", description="quant dequant for mxfp")
class QuantDequantMx(torch.nn.Module):
    def __init__(self, bits=8, is_act=False):
        super(QuantDequantMx, self).__init__()
        self.bits = bits
        self.is_act = is_act
        self._get_format_params()
        self.block_size = 32
        
    def deploy(self, x: Tensor, qdim: int = -1):
        e8m0, ex_mx = self.quant(x, qdim)
        e8m0 = (e8m0 + 127).to(torch.uint8).squeeze(-1).cpu()
        ex_mx = ex_mx.flatten(qdim - 1, qdim).cpu()
        if self.bits == 8:
            ex_mx = ex_mx.to(torch.float8_e4m3fn)
        else:
            ex_mx = f32_to_f4_unpacked(ex_mx.float().cpu())
            ex_mx = pack_uint4(ex_mx)
        return ex_mx, e8m0

    def export_deploy(self, x: Tensor):
        qx, scale = self.deploy(x)
        return {
            "qweight": qx.detach().cpu(),
            "weight_scale": scale.detach().cpu(),
        }

    def fake_quant(self, x: Tensor, qdim: int = -1, v: Tensor = 0.0):
        e8m0, ex_mx = self.quant(x, qdim, v=v)
        dx = (2 ** e8m0) * ex_mx
        dx = dx.flatten(qdim - 1, qdim)
        return dx

    def forward(self, x: Tensor, v: Tensor = 0.0) -> Tensor:
        if self.bits == 16:
            return x
        return self.fake_quant(x, v=v)

    def quant(self, x: Tensor, qdim: int = -1, v: Tensor = 0.0):
        x = x.unflatten(qdim, (-1, self.block_size))
        if isinstance(v, torch.Tensor):
            v = v.to(device=x.device, dtype=x.dtype).unflatten(qdim, (-1, self.block_size))
        e8m0 = shared_exponents(x, self.emax)
        x = x / (2 ** e8m0)
        ex_mx = quantize_elewise(x, self.min_exp, self.max_norm, self.shift_val, v=v)
        return e8m0, ex_mx

    def freeze_column_quantizer(self, weight: Tensor):
        """GPTQ helper: precompute the per-block shared exponent (E8M0) from the
        original weight [rows, cols]; MX groups block_size along input channels."""
        if self.bits is None or self.bits >= 16:
            return None
        rows, cols = weight.shape
        bs = self.block_size
        if cols % bs != 0:
            raise ValueError(
                f"mxfp-GPTQ needs in_features {cols} divisible by block_size {bs}"
            )
        x = weight.detach().float().reshape(rows, cols // bs, bs)
        e8m0 = shared_exponents(x, self.emax).squeeze(-1)  # [rows, n_blocks]
        return {"e8m0": e8m0, "block_size": bs}

    def quant_column(self, w_col: Tensor, col_idx: int, state) -> Tensor:
        """GPTQ helper: quantize one input-channel column [rows] with the frozen
        per-block shared exponent (matches fake_quant applied block-wise)."""
        if state is None:
            return w_col
        e8m0 = state["e8m0"][:, col_idx // state["block_size"]]
        scale = 2 ** e8m0
        qn = quantize_elewise(w_col / scale, self.min_exp, self.max_norm, self.shift_val)
        return scale * qn

    def _get_format_params(self):
        if self.bits == 8:
            self.ebits, self.mbits, self.emax, self.max_norm, self.shift_val, self.min_exp = 4, 5, 8, 448.0, 8, -6
        else:
            self.ebits, self.mbits, self.emax, self.max_norm, self.shift_val, self.min_exp = 2, 3, 2, 6.0, 2, 0
