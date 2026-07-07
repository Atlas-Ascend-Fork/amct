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

__all__ = [
    'awq', 'gptq', 'smooth_quant', 'minmax',
    'auto_clip', 'omniquant', 'learnable_hadamard',
    'register_algorithms', 'AlgoBuildContext',
]

from dataclasses import dataclass


_REGISTERED = False


def register_algorithms():
    global _REGISTERED
    if _REGISTERED:
        return

    from .auto_clip import LAC, LWC
    from .auto_round import AutoRound
    from .omniquant import OmniQuant

    _REGISTERED = True


@dataclass
class AlgoBuildContext:
    matrix_size: int | None = None
    dim_size: int | None = None
