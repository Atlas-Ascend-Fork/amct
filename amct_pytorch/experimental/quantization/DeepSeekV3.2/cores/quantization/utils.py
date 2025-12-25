# coding=utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
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

from loguru import logger
from cores.quantization.node import ActivationQuantizer
from cores.quantization.linear import QuantLinear


def load_quant_params(layer, quant_params):
    if quant_params is None:
        return
    for name, mod in layer.named_modules():
        if isinstance(mod, ActivationQuantizer):
            if mod.is_per_tensor:
                logger.info(name, "load param")
                mod.scale = quant_params[f"{name}.scale"]
                mod.zero = quant_params[f"{name}.zero"]
                mod.maxval.data = quant_params[f"{name}.maxval"]
                mod.minval.data = quant_params[f"{name}.minval"]
            if hasattr(mod, "clip_factor_a_max"):
                if f"{name}.clip_factor_a_max" in quant_params:
                    mod.clip_factor_a_max = quant_params[f"{name}.clip_factor_a_max"]
                    mod.clip_factor_a_min = quant_params[f"{name}.clip_factor_a_min"]
            else:
                logger.info(f" {name} has not clip_factor_a_max and clip_factor_a_min")
        if isinstance(mod, QuantLinear):
            if hasattr(mod, "clip_factor_w_max"):
                if f"{name}.clip_factor_w_max" in quant_params:
                    mod.clip_factor_w_max = quant_params[f"{name}.clip_factor_w_max"]
                    mod.clip_factor_w_min = quant_params[f"{name}.clip_factor_w_min"]
            else:
                logger.info(f" {name} has not clip_factor_w_max and clip_factor_w_min")
    return quant_params