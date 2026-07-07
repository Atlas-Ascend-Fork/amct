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
from typing import Any, Dict


class BaseAlgo(ABC):
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.name = self.__class__.__name__

    @abstractmethod
    def apply(self, model, *args, **kwargs):
        pass

    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        return self.config

    def validate_config(self):
        pass


class BaseQuantAlgo(BaseAlgo):
    def __init__(self, config=None):
        super().__init__(config)
        self.quant_dtype = config.get("quant_dtype", "int") if config else "int"
        self.weight_bits = config.get("weight_bits", 8) if config else 8
        self.activation_bits = config.get("activation_bits", 8) if config else 8