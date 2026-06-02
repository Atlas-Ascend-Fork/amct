from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List

import torch.nn as nn

from ..config import MethodSpec, PruneConfig
from ..context import PruneContext
from ..domains.base import BasePruningDomain
from ..report import PruneReport


class BasePruningMethod(ABC):
    domain: str = "base"
    name: str = "base"
    requires_data: bool = False

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
