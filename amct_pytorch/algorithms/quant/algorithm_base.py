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

import torch.nn as nn


class QuantAlgorithmBase(nn.Module, ABC):
    def __init__(self):
        super().__init__()
        self.is_observe = False

    def calib_forward(self, x, *args, **kwargs):
        return x

    @abstractmethod
    def forward(self, x, *args, **kwargs):
        raise NotImplementedError

    def trainable_params(self):
        return list(self.parameters())

    def export_ptq_params(self):
        return {name: param.detach().cpu() for name, param in self.named_parameters()}

    def load_ptq_params(self, params):
        named_params = dict(self.named_parameters())
        for name, value in params.items():
            if name not in named_params:
                continue
            param = named_params[name]
            param.data.copy_(value.to(device=param.device, dtype=param.dtype))
