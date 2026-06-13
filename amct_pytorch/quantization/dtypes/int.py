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
        self.group_size = -1

    def fake_quant(self, x: Tensor, v: Tensor = 0.0) -> Tensor:
        if self.is_act:
            x = dynamic_per_token_quant(x, self.bits)
        else:
            x = weight_quant(x, self.bits, v=v, group_size=self.group_size)
        return x

    def forward(self, x: Tensor, v: Tensor = 0.0) -> Tensor:
        if self.bits == 16:
            return x
        return self.fake_quant(x, v=v)

    def export_deploy(self, x: Tensor):
        qx, scale, bias = weight_quant(x, self.bits, real_quant=True,
                                       group_size=self.group_size)
        payload = {
            "qweight": qx.detach().cpu(),
            "weight_scale": scale.detach().cpu(),
        }
        if bias is not None:
            payload["weight_bias"] = bias.detach().cpu()
        elif self.group_size is not None and self.group_size > 0:
            # symmetric group-wise (W4A16): zero antiquant offset
            payload["weight_offset"] = torch.zeros_like(scale).detach().cpu()
        return payload

    def freeze_column_quantizer(self, weight: Tensor):
        """GPTQ helper: precompute a fixed per-group symmetric scale from the
        original weight [rows, cols]. Returns None for non-quant (bits>=16)."""
        if self.bits is None or self.bits >= 16:
            return None
        qmax = 2 ** (self.bits - 1) - 1
        rows, cols = weight.shape
        gsize = self.group_size
        if gsize is not None and gsize > 0 and cols % gsize == 0:
            eff = gsize
        else:
            eff = cols
        n_groups = cols // eff
        scale = (
            weight.detach().float().abs()
            .reshape(rows, n_groups, eff).amax(dim=2) / qmax
        ).clamp(min=1e-9)
        return {"qmax": qmax, "scale": scale, "eff_group": eff}

    def quant_column(self, w_col: Tensor, col_idx: int, state) -> Tensor:
        """GPTQ helper: quantize one input-channel column [rows] with the frozen
        per-group symmetric scale."""
        if state is None:
            return w_col
        s = state["scale"][:, col_idx // state["eff_group"]]
        qmax = state["qmax"]
        return torch.clamp(torch.round(w_col / s), -qmax, qmax) * s
