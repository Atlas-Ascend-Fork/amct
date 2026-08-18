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
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, FrozenSet, List, Optional

import torch.nn as nn

from ..config import MethodSpec, PruneConfig
from ..context import PruneContext
from ..domains.base import BasePruningDomain
from ..report import PruneReport


class BasePruningMethod(ABC):
    domain: str = "base"
    name: str = "base"
    requires_data: bool = False
    requires_targets: bool = True
    # kwargs this method reads, so a typo is rejected instead of falling back to a
    # default the caller never asked for. None means undeclared (e.g. a method from a
    # caller-supplied registry), and the kwargs of such a method are left unchecked.
    accepted_kwargs: Optional[FrozenSet[str]] = None
    # Whether an accuracy-search trial of this method can be measured by masking the cut
    # into the model instead of pruning a copy. Only true for methods that just select
    # what to keep: one that rewrites weights (least-squares reconstruction, expert
    # merging) has no mask equivalent, so its trials keep copying.
    supports_masked_trial: bool = False

    @abstractmethod
    def apply(
        self,
        model: nn.Module,
        domain: BasePruningDomain,
        targets: List[Any],
        context: PruneContext,
        report: PruneReport,
        config: PruneConfig,
        spec: MethodSpec,
    ) -> None:
        raise NotImplementedError
