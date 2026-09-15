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
import torch.nn.functional as F

from amct_pytorch.quantization.dtypes import DTYPE_REGISTRY
from amct_pytorch.quantization.dtypes.fp_impl import (
    f32_to_f4_unpacked,
    pack_uint4,
    quantize_elewise,
    weight_dequant,
    round_to_decimal,
)

BLOCK_SIZE_ROW = 128


@DTYPE_REGISTRY.register(name="fp", description="quant dequant for block fp")
class QuantDequantFP(torch.nn.Module):
    extra_config = ("block_size_col", "scale_dtype")

    def __init__(
        self, bits=8, is_act=False, block_size_col=128, scale_dtype="fp32", *args
    ):
        super(QuantDequantFP, self).__init__()
        self.bits = bits
        if self.bits != 8 and self.bits != 16:
            if scale_dtype != "e8m0":
                raise ValueError(
                    f"Block fp quant currently supports only 8-bits quantization, but got bits={self.bits}"
                )
        self.is_act = is_act
        self.is_observe = False
        self._get_format_params()
        self.block_size_col = block_size_col
        self.scale_dtype = scale_dtype

    def reshape_block(self, x):
        M, N = x.shape

        block_m = self.block_size_col
        block_n = BLOCK_SIZE_ROW  # amct only support [1x128, 128x128] two quantization block shape

        pad_m = (block_n - M % block_n) % block_n
        pad_n = (block_m - N % block_m) % block_m

        W_pad = F.pad(x, (0, pad_n, 0, pad_m))
        pad_shape = W_pad.shape

        Mp, Np = W_pad.shape
        W_blocks = W_pad.reshape(
            Mp // block_n, block_n, Np // block_m, block_m
        ).permute(0, 2, 1, 3)

        return W_blocks, pad_shape

    def quantize_fp_block(self, x):
        ori_shape = x.shape
        N = x.shape[-1]
        x = x.reshape(-1, N)
        M, N = x.shape

        W_blocks, pad_shape = self.reshape_block(x)

        amax = torch.amax(torch.abs(W_blocks), dim=(-1, -2), keepdim=True)

        if self.scale_dtype == "e8m0":
            scale = round_to_decimal(amax) - self.emax
            scale = scale.clamp(min=-127, max=1e10)
            # e8m0 scale is exported as the float32 multiplier 2**e, not the
            # OCP E8M0 uint8 exponent bytes; deploy/runtime must read it as
            # float32 (weight_dequant passes float scales through unchanged).
            scale = 2**scale
        else:
            scale = amax / self.max_norm
            scale = torch.clamp(scale, min=1e-8)

        W_scaled = W_blocks / scale
        W_scaled = torch.clamp(W_scaled, -self.max_norm, self.max_norm)

        W_scaled = W_scaled.permute(0, 2, 1, 3).reshape(pad_shape)[:M, :N]
        W_fp8 = W_scaled.reshape(ori_shape)

        return W_fp8, scale

    def dequantize_fp_block(self, x, scale):
        ori_shape = x.shape
        x = x.reshape(-1, ori_shape[-1])
        scale = scale.reshape(scale.shape[0], scale.shape[1])
        if self.block_size_col == BLOCK_SIZE_ROW:
            out = weight_dequant(x, scale, block_size=BLOCK_SIZE_ROW)
        else:
            # block_size_col == 1 -> 128x1 blocks; transpose so weight_dequant's
            # is_mx path repeats the 128-grouping along the columns.
            out = weight_dequant(
                x.t(), scale.t(), block_size=BLOCK_SIZE_ROW, is_mx=True
            ).t()
        return out.reshape(ori_shape)

    def deploy(self, x: Tensor, qdim: int = -1, v: Tensor | float = 0.0):
        scale, ex_mx = self.quant(x, qdim, v=v)
        ex_mx = ex_mx.cpu()
        if self.bits == 8:
            ex_mx = ex_mx.to(torch.float8_e4m3fn)
        else:
            ex_mx = f32_to_f4_unpacked(ex_mx.float().cpu())
            ex_mx = pack_uint4(ex_mx)
        # Drop the trailing (1, 1) block dims for every scale_dtype so the
        # exported weight_scale rank stays uniform (fp32 and e8m0 both 2-D).
        scale = scale.squeeze(-1).squeeze(-1)
        return ex_mx, scale

    def export_deploy(self, x: Tensor, v: Tensor | float = 0.0):
        qx, scale = self.deploy(x, v=v)
        return {
            "qweight": qx.detach().cpu(),
            "weight_scale": scale.detach().cpu(),
        }

    def fake_quant(self, x: Tensor, qdim: int = -1, v: Tensor = 0.0):
        x_dtype = x.dtype
        scale, ex_mx = self.quant(x, qdim, v=v)
        dx = self.dequantize_fp_block(ex_mx, scale)
        dx = dx.to(x_dtype)
        return dx

    def forward(self, x: Tensor, v: Tensor = 0.0) -> Tensor:
        if self.is_observe or self.bits == 16:
            return x
        return self.fake_quant(x, v=v)

    def quant(self, x: Tensor, qdim: int = -1, v: Tensor = 0.0):
        if isinstance(v, torch.Tensor):
            v = v.to(device=x.device, dtype=x.dtype)
        x_scaled, scale = self.quantize_fp_block(x)
        ex_mx = quantize_elewise(
            x_scaled, self.min_exp, self.max_norm, self.shift_val, v=v
        )
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
