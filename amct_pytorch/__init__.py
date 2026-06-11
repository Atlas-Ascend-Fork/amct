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

from amct_pytorch.classic.graph_based.amct_pytorch import *

# Structured pruning facade -- exposed lazily (PEP 562) so ``import amct_pytorch`` stays
# light: the pruning subpackage and its torch-heavy internals load only on first access
# (e.g. ``amct.prune(...)``), mirroring the ``amct.quantize`` surface.
_PRUNING_EXPORTS = (
    'prune', 'get_last_report',
    'accuracy_based_auto_prune', 'size_budget_prune',
    'guarded_variance_hybrid_prune', 'guarded_recovery_hybrid_prune',
    'prune_finetune', 'prune_diagnose',
    'CNN_RECONSTRUCT_PRUNE_CFG', 'CNN_VARIANCE_PRUNE_CFG', 'DENSE_LOWVAR_PRUNE_CFG',
    'FULL_STRUCTURED_PRUNE_CFG', 'MOE_MASSVAR_PRUNE_CFG',
    'MOE_OUTPUT_MERGE_PRUNE_CFG', 'SENSITIVITY_ALLOC_PRUNE_CFG',
)
__all__ += list(_PRUNING_EXPORTS)


def __getattr__(name):
    # PEP 562 lazy lookup for the pruning facade (plus the secondary
    # ``get_last_auto_prune`` accessor, reachable but kept out of ``__all__``).
    if name in _PRUNING_EXPORTS or name == 'get_last_auto_prune':
        import amct_pytorch.pruning as _pruning
        return getattr(_pruning, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    # Surface the lazily-exposed pruning facade in dir()/REPL tab-completion.
    return sorted(set(globals()) | set(_PRUNING_EXPORTS) | {'get_last_auto_prune'})
