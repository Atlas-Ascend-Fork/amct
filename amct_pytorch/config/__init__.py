# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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

__all__ = [
    'INT4_AWQ_WEIGHT_QUANT_CFG',
    'INT4_GPTQ_WEIGHT_QUANT_CFG',
    'INT8_SMOOTHQUANT_CFG',
    'INT8_MINMAX_WEIGHT_QUANT_CFG',
    'HIFP8_OFMR_CFG',
    'FP8_OFMR_CFG',
    'MXFP8_QUANT_CFG',
    'MXFP4_AWQ_WEIGHT_QUANT_CFG',
    'set_default_config',
    'parse_config',
]

from amct_pytorch.config.config import INT4_AWQ_WEIGHT_QUANT_CFG, INT4_GPTQ_WEIGHT_QUANT_CFG
from amct_pytorch.config.config import INT8_SMOOTHQUANT_CFG, INT8_MINMAX_WEIGHT_QUANT_CFG
from amct_pytorch.config.config import HIFP8_OFMR_CFG, FP8_OFMR_CFG, MXFP8_QUANT_CFG
from amct_pytorch.config.config import MXFP4_AWQ_WEIGHT_QUANT_CFG
from amct_pytorch.config.parser import set_default_config, parse_config
