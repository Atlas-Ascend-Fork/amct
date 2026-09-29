# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
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
# ----------------------------------------------------------------------------
"""Construct local adapters without changing the parent transformer layer."""

import copy
from functools import lru_cache

from amct_pytorch.classic.deploy_op.shared_quantize import SharedDynamicQuantize


@lru_cache(maxsize=None)
def _adapter_type(original_type, forward):
    return type(
        f'Fused{original_type.__name__}', (original_type,), {'forward': forward}
    )


def make_local_adapter(original, forward, act_type):
    # Keep native attributes, training state and hooks. Copy registration maps so
    # adding the shared quantizer does not modify the original module.
    fused = copy.copy(original)
    fused.__class__ = _adapter_type(type(original), forward)
    for name in ('_modules', '_parameters', '_buffers'):
        setattr(fused, name, getattr(original, name).copy())
    fused.shared_quantize = SharedDynamicQuantize(act_type)
    fused.shared_quantize.train(original.training)
    return fused
