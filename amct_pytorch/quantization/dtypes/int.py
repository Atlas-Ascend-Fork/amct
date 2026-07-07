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
from loguru import logger
from torch import Tensor
from amct_pytorch.quantization.dtypes import DTYPE_REGISTRY
from amct_pytorch.quantization.dtypes.int_impl import weight_quant, dynamic_per_token_quant


@DTYPE_REGISTRY.register(name="int", description="quant dequant for int")
class QuantDequantInt(torch.nn.Module):
    def __init__(self, bits=8, is_act=False):
        super(QuantDequantInt, self).__init__()
        self.bits = bits
        self.is_act = is_act
        self._deploy_mod = False

    def fake_quant(self, x: Tensor, v: Tensor = 0.0) -> Tensor:
        if self.is_act:
            x = dynamic_per_token_quant(x, self.bits)
        else:
            x = weight_quant(x, self.bits, v=v)
        return x

    def forward(self, x: Tensor, v: Tensor = 0.0) -> Tensor:
        if self.bits == 16:
            return x
        return self.fake_quant(x, v=v)

    def export_deploy(self, x: Tensor, v: Tensor | float = 0.0):
        qx, scale, bias = weight_quant(x, self.bits, real_quant=True, v=v)
        return {
            "qweight": qx.detach().cpu(),
            "weight_scale": scale.detach().cpu(),
            "weight_bias": bias.detach().cpu() if bias is not None else None,
        }
