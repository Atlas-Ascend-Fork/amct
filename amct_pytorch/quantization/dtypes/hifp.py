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



@DTYPE_REGISTRY.register(name="hifp", description="quant dequant for hifloat")
class QuantDequantHifp(torch.nn.Module):
    def __init__(self, bits=8, is_act=False):
        super(QuantDequantHifp, self).__init__()
        self.bits = bits
        self.is_act = is_act
        self._deploy_mod = False

    def fake_quant(self, x: Tensor, v: Tensor = 0.0) -> Tensor:
        if self.bits == 8:
            dq = hifloat8_fake_quant(x)
            return dq.detach() + (x - x.detach())
        else:
            raise ValueError(f"HiFloat 4-bit are not implemented yet.")

    def forward(self, x: Tensor, v: Tensor = 0.0) -> Tensor:
        if self.bits == 16:
            return x
        elif self.bits == 8:
            return self.fake_quant(x, v=v)
        else:
            raise ValueError(f"HiFloat only supports 8-bit quantization, got {self.bits}.")

    def export_deploy(self, x: Tensor, v: Tensor | float = 0.0):
        raise NotImplementedError(f"HiFloat-{self.bits} export_deploy is not supported yet.")

    def deploy(self, x: Tensor, qdim: int = -1, v: Tensor | float = 0.0):
        raise NotImplementedError(f"HiFloat-{self.bits} deploy is not supported yet.")
