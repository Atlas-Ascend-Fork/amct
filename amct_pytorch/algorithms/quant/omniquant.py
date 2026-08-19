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

from amct_pytorch.algorithms.quant.base import QuantAlgorithmBase
from amct_pytorch.algorithms.registry_factory import ALGO_REGISTRY


@ALGO_REGISTRY.register(
    name="omniquant",
    description="omniquant",
    targets=("structure",),
)
class OmniQuant(QuantAlgorithmBase):
    def __init__(self, args, ctx):
        super().__init__()
        self.args = args
        self.dim = ctx.dim_size
        self.log_scale = torch.nn.Parameter(
            torch.zeros((1, self.dim)), requires_grad=True
        )

    def transform(self):
        pass

    def forward(
        self, x: torch.Tensor, inv_t: bool = False, name: str = None
    ) -> torch.Tensor:
        if self.is_observe:
            return self.calib_forward(x, inv_t=inv_t, name=name)

        dtype = x.dtype
        scale = self._get_scale(dtype=x.dtype, device=x.device)
        if not inv_t:
            x = x / scale
        else:
            x = x * scale
        return x.to(dtype)

    def calib_forward(
        self, x: torch.Tensor, inv_t: bool = False, name: str = None
    ) -> torch.Tensor:
        if inv_t:
            return x

        hidden_dim = x.shape[-1]
        tensor = x.view(-1, hidden_dim).abs().detach()
        comming_max = torch.max(tensor, dim=0)[0].float().clamp(min=1e-4)
        self.log_scale.data.copy_(
            torch.max(self.log_scale.data, comming_max.log().to(self.log_scale.device))
        )
        return x

    def _get_scale(self, dtype, device):
        scale = torch.exp(self.log_scale)
        scale = scale.clamp(min=1e-4, max=1e4)
        return scale.to(device=device, dtype=dtype)
