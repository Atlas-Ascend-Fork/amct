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
    'Registry', 'RegistryItem',
    'QUANT_ALGO_REGISTRY', 'ALGO_REGISTRY',
    'register_quant_algo', 'register_all_algorithms',
    'quantize', 'convert', 'algorithm_register',
    'INT4_AWQ_WEIGHT_QUANT_CFG', 'INT4_GPTQ_WEIGHT_QUANT_CFG',
    'INT8_SMOOTHQUANT_CFG', 'INT8_MINMAX_WEIGHT_QUANT_CFG',
    'HIFP8_OFMR_CFG', 'FP8_OFMR_CFG', 'MXFP8_QUANT_CFG',
    'MXFP4_AWQ_WEIGHT_QUANT_CFG', 'HIFP8_CAST_CFG', 'HIFP8_QUANTILE_CFG',
]

from amct_pytorch.common.utils.registry_factory import Registry, RegistryItem
from amct_pytorch.algorithms.registry_factory import (
    QUANT_ALGO_REGISTRY, ALGO_REGISTRY,
    register_quant_algo, register_all_algorithms,
)
from amct_pytorch.classic import quantize, convert, algorithm_register
from amct_pytorch.common.config import (
    INT4_AWQ_WEIGHT_QUANT_CFG, INT4_GPTQ_WEIGHT_QUANT_CFG,
    INT8_SMOOTHQUANT_CFG, INT8_MINMAX_WEIGHT_QUANT_CFG,
    HIFP8_OFMR_CFG, FP8_OFMR_CFG, MXFP8_QUANT_CFG,
    MXFP4_AWQ_WEIGHT_QUANT_CFG, HIFP8_CAST_CFG, HIFP8_QUANTILE_CFG,
)

# graph_based PTQ entry points depend on protobuf modules generated at build
# time (amct_pytorch_proto_compile target). In a pure source checkout the
# *_pb2.py files are absent, so a source-tree import deliberately swallows the
# ImportError and lets the rest of the package load.
try:
    from amct_pytorch.classic.ptq.graph_based.amct_pytorch import *  # noqa: F401,F403
except ImportError:
    pass
