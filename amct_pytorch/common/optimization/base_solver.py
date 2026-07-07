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

from abc import ABC, abstractmethod
from typing import Any, Optional

from amct_pytorch.common.datasets.ptq_io import load_ptq_inps
from amct_pytorch.common.optimization.factory import build_lr_scheduler, build_optimizer


class BaseSolver(ABC):
    granularity = "block"

    def __init__(
            self,
            args: Any,
            layer_idx: int,
            model: Any,
            optimizer: Optional[Any] = None,
            lr_scheduler: Optional[Any] = None,
            max_iters: int = 100,
    ):
        self.args = args
        self.quant_target = args.quant_target
        self.layer_idx = layer_idx
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.max_iters = max_iters
        self.current_iter = 0

    @abstractmethod
    def solve(self, calibration_data: Any) -> Any:
        pass

    def finalize(self) -> Any:
        if hasattr(self.model, "export_ptq_params"):
            return self.model.export_ptq_params()
        params = {}
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            params[name] = param.detach().cpu()
        return params

    def forward_once(self, *args, **kwargs):
        pass

    def origin_forward(self, *args, **kwargs):
        pass

    def quant_forward(self, *args, **kwargs):
        pass

    def step(self,):
        if self.optimizer:
            self.optimizer.step()
            self.optimizer.zero_grad()

        if self.lr_scheduler:
            self.lr_scheduler.step()

        self.current_iter += 1
