from abc import ABC, abstractmethod
from typing import Any, Dict, Callable, Optional
from amct_pytorch.common.models import get_adapter

class BaseSolver(ABC):
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.optimizer = None
        self.adapter: Optional[Any] = None

    def set_adapter(self, model_name: str, adapter_type: str = None):
        self.adapter = get_adapter(model_name, adapter_type)

    @abstractmethod
    def solve(self, model, calibration_data: Callable) -> Any:
        pass

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer