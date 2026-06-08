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
__all__ = [
    "BasePruningMethod",
    "VarianceChannelPruningMethod",
    "LowVarianceDensePruningMethod",
    "MassMoEPruningMethod",
    "MassVarianceMoEPruningMethod",
]

from .base import BasePruningMethod
from .cnn_variance import VarianceChannelPruningMethod
from .dense_low_variance import LowVarianceDensePruningMethod
from .moe_mass_pruning import MassMoEPruningMethod
from .moe_mass_variance_pruning import MassVarianceMoEPruningMethod
