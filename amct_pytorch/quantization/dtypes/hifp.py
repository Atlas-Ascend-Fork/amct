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
from amct_pytorch.quantization.dtypes.hifp_impl import hifloat8_fake_quant
from amct_pytorch.common.utils.hifloat4_quant import hifloat4_fake_quant, hif4_cast


@DTYPE_REGISTRY.register(name="hifp", description="quant dequant for hifloat")
class QuantDequantHifp(torch.nn.Module):
    """HiFloat (hifp) fake-quant / deploy dtype for the LLM engine.

    One registered dtype for both HiFloat widths, selected by ``bits``:
      - ``bits == 8`` -> HiFloat8 fake-quant (``hifp_impl.hifloat8_fake_quant``);
      - ``bits == 4`` -> HiFloat4 (hifx4) fake-quant + real pack (``common/utils/hifloat4_quant``);
        the hierarchical 64-block scale (L1 E6M2 + L2/L3 micro-exponents) is computed inside
        ``hifloat4_fake_quant`` / ``hif4_cast``, so this class only routes the tensor through it.
      - ``bits == 16`` -> pass-through.
    Deploy/pack is currently implemented for HiFloat4 only (HiFloat8 pack raises). The learnable
    rounding offset ``v`` used by int/mxfp does not apply to HiFloat's fixed rounding -> ignored.
    """

    def __init__(self, bits=8, is_act=False):
        super(QuantDequantHifp, self).__init__()
        self.bits = bits
        self.is_act = is_act
        self._deploy_mod = False

    def fake_quant(self, x: Tensor, v: Tensor = 0.0) -> Tensor:
        if self.bits == 8:
            dq = hifloat8_fake_quant(x)
            return dq.detach() + (x - x.detach())
        elif self.bits == 4:
            return hifloat4_fake_quant(x, qdim=-1)
        else:
            raise ValueError(f"HiFloat only supports 8-bit or 4-bit fake-quant, got {self.bits}.")

    def forward(self, x: Tensor, v: Tensor = 0.0) -> Tensor:
        if self.bits == 16:
            return x
        elif self.bits in (8, 4):
            return self.fake_quant(x, v=v)
        else:
            raise ValueError(f"HiFloat only supports 16/8/4-bit quantization, got {self.bits}.")

    def export_deploy(self, x: Tensor, v: Tensor | float = 0.0):
        if self.bits == 4:
            value, scale = self.deploy(x)
            return {"qweight": value, "weight_scale": scale}
        raise NotImplementedError(f"HiFloat-{self.bits} export_deploy is not supported yet.")

    def deploy(self, x: Tensor, qdim: int = -1, v: Tensor | float = 0.0):
        if self.bits == 4:
            # Real HiF4 pack: weight -> (value uint8 [M, N//2], scale uint8 [M, N//64, 4]).
            # hif4_cast groups along the last dim in 64-blocks, so N must be a multiple of 64.
            mat = x.reshape(-1, x.shape[-1]).to(torch.float32)
            scale, value = hif4_cast(mat, output="packed")
            return value.contiguous().cpu(), scale.contiguous().cpu()
        raise NotImplementedError(f"HiFloat-{self.bits} deploy is not supported yet.")
