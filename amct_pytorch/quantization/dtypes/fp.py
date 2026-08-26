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
from __future__ import annotations

import torch
from torch import Tensor
from amct_pytorch.quantization.dtypes import DTYPE_REGISTRY
from amct_pytorch.quantization.dtypes.fp_impl import (
    f32_to_f4_unpacked,
    pack_uint4,
    quantize_elewise,
    quantize_fp_block,
    dequantize_fp_block
)


@DTYPE_REGISTRY.register(name="fp", description="quant dequant for block fp")
class QuantDequantFP(torch.nn.Module):
    def __init__(self, bits=8, is_act=False, block_size=128):
        super(QuantDequantFP, self).__init__()
        self.bits = bits
        if self.bits != 8 and self.bits != 16:
            raise ValueError(f"Block fp quant currently supports only 8-bits quantization, but got bits={self.bits}")
        self.is_act = is_act
        self.is_observe = False
        self._get_format_params()
        self.block_size = block_size

    def deploy(self, x: Tensor, qdim: int = -1, v: Tensor | float = 0.0):
        scale, ex_mx = self.quant(x, qdim, v=v)
        ex_mx = ex_mx.flatten(qdim - 1, qdim).cpu()
        if self.bits == 8:
            ex_mx = ex_mx.to(torch.float8_e4m3fn)
        else:
            ex_mx = f32_to_f4_unpacked(ex_mx.float().cpu())
            ex_mx = pack_uint4(ex_mx)
        return ex_mx, scale

    def export_deploy(self, x: Tensor, v: Tensor | float = 0.0):
        qx, scale = self.deploy(x, v=v)
        return {
            "qweight": qx.detach().cpu(),
            "weight_scale": scale.detach().cpu(),
        }

    def fake_quant(self, x: Tensor, qdim: int = -1, v: Tensor = 0.0):
        scale, ex_mx = self.quant(x, qdim, v=v)
        dx = dequantize_fp_block(ex_mx, scale, self.min_exp, self.max_norm, self.block_size)
        return dx

    def forward(self, x: Tensor, v: Tensor = 0.0) -> Tensor:
        if self.is_observe or self.bits == 16:
            return x
        return self.fake_quant(x, v=v)

    def quant(self, x: Tensor, qdim: int = -1, v: Tensor = 0.0):
        if isinstance(v, torch.Tensor):
            v = v.to(device=x.device, dtype=x.dtype)
        x_scaled, scale = quantize_fp_block(x, self.min_exp, self.max_norm, block_size=self.block_size)
        ex_mx = quantize_elewise(x_scaled, self.min_exp, self.max_norm, self.shift_val, v=v)
        return scale, ex_mx

    def _get_format_params(self):
        if self.bits == 8:
            (
                self.ebits,
                self.mbits,
                self.emax,
                self.max_norm,
                self.shift_val,
                self.min_exp,
            ) = 4, 5, 8, 448.0, 8, -6
        elif self.bits == 4:
            (
                self.ebits,
                self.mbits,
                self.emax,
                self.max_norm,
                self.shift_val,
                self.min_exp,
            ) = 2, 3, 2, 6.0, 2, 0
