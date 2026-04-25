from abc import ABC, abstractmethod
from typing import Any

class BaseWorkflow(ABC):
    def __init__(self, config=None):
        self.config = config or {}
        self.steps = []

    @abstractmethod
    def run(self, *args, **kwargs) -> Any:
        pass

    def add_step(self, step):
        self.steps.append(step)