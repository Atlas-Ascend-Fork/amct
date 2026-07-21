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
from amct_pytorch.common.utils.hifloat4_quant import hifloat4_fake_quant, hif4_cast


@DTYPE_REGISTRY.register(name="hifloat4", description="quant dequant for hifloat4")
class QuantDequantHiF4(torch.nn.Module):
    """HiFloat4 (hifx4) fake-quant dtype for the LLM-PTQ engine.

    A thin wrapper over ``hifloat4_fake_quant``. Two things differ from
    ``QuantDequantMx`` and are deliberate:

    1. No explicit block-scale / shared-exponent step here. HiFloat4's hierarchical
       64-element block scale (L1 E6M2 + L2/L3 micro-exponents) is computed *inside*
       ``hifloat4_fake_quant`` (see common/utils/hifloat4_quant), so this class only routes
       the tensor through it.
    2. ``bits`` selects only whether the format is applied, not which variant. HiFloat4
       is intrinsically a 4-bit (S1P2) element format, so any ``bits < 16`` maps to the
       same fake-quant; ``bits == 16`` is a pass-through (mirrors QuantDequantMx / QuantDequantInt).

    The learnable rounding offset ``v`` used by int/mxfp is not supported by HiFloat4's
    fixed round-half-up-to-quarter rule, so it is accepted for interface compatibility
    and ignored.

    ``deploy`` / ``export_deploy`` produce the real HiF4 packed bytes (value + scale) via
    ``hif4_cast(output="packed")`` -- the same packer the offline cast uses -- so PPL/accuracy
    eval (``forward``) and deploy share one quantization implementation.
    """

    def __init__(self, bits=4, is_act=False):
        super(QuantDequantHiF4, self).__init__()
        self.bits = bits
        self.is_act = is_act

    def fake_quant(self, x: Tensor, qdim: int = -1, v: Tensor = 0.0) -> Tensor:
        return hifloat4_fake_quant(x, qdim=qdim)

    def forward(self, x: Tensor, v: Tensor = 0.0) -> Tensor:
        if self.bits == 16:
            return x
        return self.fake_quant(x, v=v)

    def deploy(self, x: Tensor, qdim: int = -1):
        """Real HiF4 pack: weight -> (value, scale) uint8 byte tensors.

        Groups along the last dim in blocks of 64 (``hif4_cast`` contract), so the last dim
        must be a multiple of 64. Returns ``(value, scale)``:
          value: uint8 ``[M, N//2]``     -- two 4-bit S1P2 codes per byte
          scale: uint8 ``[M, N//64, 4]`` -- per-64-group E6M2 + hierarchical micro-exponents
        """
        mat = x.reshape(-1, x.shape[-1]).to(torch.float32)
        scale, value = hif4_cast(mat, output="packed")
        return value.contiguous().cpu(), scale.contiguous().cpu()

    def export_deploy(self, x: Tensor):
        value, scale = self.deploy(x)
        return {"qweight": value, "weight_scale": scale}
