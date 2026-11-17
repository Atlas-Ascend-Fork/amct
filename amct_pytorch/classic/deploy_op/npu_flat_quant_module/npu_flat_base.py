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
"""NpuFlatQuant deploy base: convert-time packing + W4A4 forward slots.

The convert Pass builds the deploy class from the calibrated module
(registry-mapped). The base __init__ runs the convert chain —
reparameterize (bake W@M^-T, freeze transforms; the LN diag merge already
ran in the Pass) — then packs each projection into the operator format and
stores transform matrices / clip ratios as read-only parameters.

Forward stays abstract: the Qwen3/CLIP subclasses orchestrate the
torch_npu operators following each model's original forward structure."""

import torch
import torch.nn as nn


class NpuFlatQuantDeployMixin(nn.Module):
    """Base for NpuFlatQuant<model><Attention/MLP> deploy modules."""

    #: checked against type(quant_module).__name__ at construction
    _quant_module_prefix = None

    def __init__(self, quant_module=None):
        super().__init__()
        if quant_module is None:
            raise ValueError(f'{type(self).__name__} requires a quant module')
        expected = self._quant_module_prefix
        if not type(quant_module).__name__.startswith(expected):
            raise ValueError(
                f'cannot build {type(self).__name__} from '
                f'{type(quant_module).__name__}, expect {expected}*'
            )
        if not getattr(quant_module, '_calibrated', False):
            from amct_pytorch.common.utils.log import LOGGER

            LOGGER.logw(
                f'{type(quant_module).__name__} not calibrated; transforms '
                'are at init values'
            )
        # 1. bake transforms into weights (idempotent via _eval_mode); the
        #    LN diag merge already ran via get_external_reparam in the Pass
        quant_module.reparameterize()
        # 2. subclass hook: pack projections / store tensors
        self._from_quant_module(quant_module)

    def _from_quant_module(self, quant_module):
        """Pack + store deploy tensors; each model subclass implements."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # packing helpers shared by subclasses
    # ------------------------------------------------------------------
    @staticmethod
    def _pack_param(tensor):
        return torch.nn.Parameter(tensor, requires_grad=False)
