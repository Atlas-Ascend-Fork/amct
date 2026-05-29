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

from typing import Dict, Any, Optional


class QuantConfig:
    def __init__(
        self,
        algorithm: str = "minmax",
        weight_bits: int = 8,
        activation_bits: int = 8,
        quant_dtype: str = "int",
        group_size: Optional[int] = None,
        **kwargs
    ):
        self.algorithm = algorithm
        self.weight_bits = weight_bits
        self.activation_bits = activation_bits
        self.quant_dtype = quant_dtype
        self.group_size = group_size
        self.extra_config = kwargs

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "QuantConfig":
        return cls(**config_dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "algorithm": self.algorithm,
            "weight_bits": self.weight_bits,
            "activation_bits": self.activation_bits,
            "quant_dtype": self.quant_dtype,
            "group_size": self.group_size,
            **self.extra_config
        }