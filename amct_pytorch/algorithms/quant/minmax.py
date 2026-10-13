# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
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
    name="minmax",
    description="Statistic-based min/max calibration",
    targets=("weight", "activation"),
    solver="calibrate",
)
class MinMax(QuantAlgorithmBase):
    """Collect min/max statistics without changing the observed tensor."""

    def __init__(self, args, *ctor_args):
        super().__init__()
        self.args = args
        self.register_buffer("min_value", torch.tensor(float("inf")))
        self.register_buffer("max_value", torch.tensor(float("-inf")))

    @torch.no_grad()
    def calib_forward(self, x, *args, **kwargs):
        self.min_value = torch.minimum(self.min_value.to(x.device), x.min())
        self.max_value = torch.maximum(self.max_value.to(x.device), x.max())
        return x

    def forward(self, x, *args, **kwargs):
        return x

    def export_ptq_params(self):
        if not torch.isfinite(self.min_value) or not torch.isfinite(self.max_value):
            return {}
        return {
            "min_value": self.min_value.detach().cpu(),
            "max_value": self.max_value.detach().cpu(),
        }
