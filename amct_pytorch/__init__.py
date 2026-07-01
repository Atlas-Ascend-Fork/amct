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
    'quantize', 'convert', 'algorithm_register',
    'INT4_AWQ_WEIGHT_QUANT_CFG', 'INT4_GPTQ_WEIGHT_QUANT_CFG',
    'INT8_SMOOTHQUANT_CFG',
    'INT8_MINMAX_WEIGHT_QUANT_CFG',
    'HIFP8_OFMR_CFG', 'FP8_OFMR_CFG', 'MXFP8_QUANT_CFG',
    'MXFP4_AWQ_WEIGHT_QUANT_CFG', 'HIFP8_CAST_CFG', 'HIFP8_QUANTILE_CFG',
]

from amct_pytorch.classic import quantize, convert, algorithm_register
from amct_pytorch.common.config import (
    INT4_AWQ_WEIGHT_QUANT_CFG, INT4_GPTQ_WEIGHT_QUANT_CFG,
    INT8_SMOOTHQUANT_CFG, INT8_MINMAX_WEIGHT_QUANT_CFG,
    HIFP8_OFMR_CFG, FP8_OFMR_CFG, MXFP8_QUANT_CFG,
    MXFP4_AWQ_WEIGHT_QUANT_CFG, HIFP8_CAST_CFG, HIFP8_QUANTILE_CFG,
)


# Classic graph-based interfaces (create_quant_config et al.) live in the
# graph_based subpackage, which pulls in onnx + compiled protobufs. Resolve them
# lazily (PEP 562) so LLM-only workflows (amct_pytorch.cli.llm / common.models)
# can `import amct_pytorch` without those deps; a missing dependency then
# surfaces loudly at the point of use instead of being swallowed at import.
def __getattr__(name):
    if name.startswith("__"):
        raise AttributeError(name)
    import importlib
    _graph = importlib.import_module(
        "amct_pytorch.classic.graph_based.amct_pytorch")
    try:
        value = getattr(_graph, name)
    except AttributeError:
        raise AttributeError(
            f"module 'amct_pytorch' has no attribute {name!r}") from None
    # Cache the resolved symbol in the module namespace so later accesses hit
    # it directly and never re-trigger __getattr__ (PEP 562 caches nothing).
    globals()[name] = value
    return value
