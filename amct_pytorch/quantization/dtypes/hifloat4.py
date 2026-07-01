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
from amct_pytorch.common.utils.quant_util import hifloat4_fake_quant


@DTYPE_REGISTRY.register(name="hifloat4", description="quant dequant for hifloat4")
class QuantDequantHiF4(torch.nn.Module):
    """HiFloat4 (hifx4) fake-quant dtype for the LLM-PTQ engine.

    A thin wrapper over the vendored ``hifloat4_fake_quant``. Two things differ from
    ``QuantDequantMx`` and are deliberate:

    1. No explicit block-scale / shared-exponent step here. HiFloat4's hierarchical
       64-element block scale (L1 E6M2 + L2/L3 micro-exponents) is computed *inside*
       ``hifloat4_fake_quant`` (see common/utils/hifx4_quant), so this class only routes
       the tensor through it.
    2. ``bits`` selects only whether the format is applied, not which variant. HiFloat4
       is intrinsically a 4-bit (S1P2) element format, so any ``bits < 16`` maps to the
       same fake-quant; ``bits == 16`` is a pass-through (mirrors QuantDequantMx / QuantDequantInt).

    The learnable rounding offset ``v`` used by int/mxfp is not supported by HiFloat4's
    fixed round-half-up-to-quarter rule, so it is accepted for interface compatibility
    and ignored.

    Real 4-bit packing (``deploy`` / ``export_deploy``) is future work (stage 2C); only
    FP->HiF4->FP simulation is available today, which is what PPL/accuracy evaluation needs.
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
        raise NotImplementedError(
            "HiFloat4 real 4-bit packing (deploy) is future work (stage 2C); "
            "only the fake-quant simulation path is available."
        )

    def export_deploy(self, x: Tensor):
        raise NotImplementedError(
            "HiFloat4 export_deploy (qweight/weight_scale packing) is future work (stage 2C)."
        )
