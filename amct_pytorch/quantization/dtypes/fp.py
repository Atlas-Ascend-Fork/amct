# -*- coding: UTF-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import torch
from torch import Tensor

from amct_pytorch.quantization.dtypes import DTYPE_REGISTRY
from amct_pytorch.quantization.dtypes.mxfp_impl import (
    f32_to_f4_unpacked,
    pack_uint4,
    quantize_elewise,
)


@DTYPE_REGISTRY.register(name="fp", description="per-tensor quant dequant for fp")
class QuantDequantFp(torch.nn.Module):
    """E4M3 FP8 / E2M1 FP4 with one FP32 scale per input tensor.

    Weights and activations both use current-tensor amax scaling. Rounding
    follows the existing MXFP implementation, including STE and offset ``v``.
    ``bits=16`` bypasses quantization in forward; it is not an export format.
    """

    def __init__(self, bits=8, is_act=False):
        super().__init__()
        if bits not in (4, 8, 16):
            raise ValueError(f"FP bits must be 4, 8 or 16, got {bits}.")
        self.bits = bits
        self.is_act = is_act
        if bits == 8:
            self.min_exp, self.max_norm, self.shift_val = -6, 448.0, 8
        else:
            self.min_exp, self.max_norm, self.shift_val = 0, 6.0, 2

    def quant(self, x: Tensor, v: Tensor | float = 0.0):
        """Return scalar dequantization scale and unpacked quantized values."""
        if self.bits == 16:
            raise ValueError("bits=16 is only supported as a forward passthrough.")
        if not x.is_floating_point():
            raise ValueError("FP quantization requires a floating-point tensor.")
        values = x.float()
        if values.numel() == 0:
            return values.new_ones(()), values
        amax = values.detach().abs().amax()
        # Keep the division safe for zero and subnormal inputs in FP32.
        fp32 = torch.finfo(torch.float32)
        scale = (amax / self.max_norm).clamp_min(fp32.tiny * fp32.eps)
        scale = torch.where(amax == 0, torch.ones_like(scale), scale)
        q = quantize_elewise(
            values / scale, self.min_exp, self.max_norm, self.shift_val, v=v
        )
        return scale, q

    def fake_quant(self, x: Tensor, v: Tensor | float = 0.0) -> Tensor:
        scale, q = self.quant(x, v=v)
        return (q * scale).to(x.dtype)

    def forward(self, x: Tensor, v: Tensor | float = 0.0) -> Tensor:
        if self.bits == 16:
            return x
        return self.fake_quant(x, v=v)

    def deploy(self, x: Tensor, v: Tensor | float = 0.0):
        if self.bits == 4 and (x.ndim == 0 or x.shape[-1] % 2):
            raise ValueError("FP4 export requires an even last dimension for packing.")
        scale, q = self.quant(x, v=v)
        q = q.detach().cpu()
        if self.bits == 8:
            q = q.to(torch.float8_e4m3fn)
        else:
            q = pack_uint4(f32_to_f4_unpacked(q))
        return q, scale.detach().cpu()

    def export_deploy(self, x: Tensor, v: Tensor | float = 0.0):
        q, scale = self.deploy(x, v=v)
        return {"qweight": q, "weight_scale": scale}
