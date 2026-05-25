from __future__ import annotations

from abc import ABC, abstractmethod

import torch.nn as nn

from ..config import PruneConfig
from ..report import PruneReport


class BasePruningStage(ABC):
    stage_name: str = "base"

    def __init__(self, config: PruneConfig) -> None:
        self.config = config

    @abstractmethod
    def apply(self, model: nn.Module, report: PruneReport) -> None:
        raise NotImplementedError
