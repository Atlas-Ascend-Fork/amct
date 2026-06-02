from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, List

import torch.nn as nn

from ..config import PruneConfig


class BasePruningDomain(ABC):
    name: str = "base"

    @abstractmethod
    def find_targets(self, model: nn.Module, config: PruneConfig) -> List[Any]:
        raise NotImplementedError
