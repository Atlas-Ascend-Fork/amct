from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseAlgo(ABC):
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.name = self.__class__.__name__

    @abstractmethod
    def apply(self, model, *args, **kwargs):
        pass

    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        return self.config

    def validate_config(self):
        pass


class BaseQuantAlgo(BaseAlgo):
    def __init__(self, config=None):
        super().__init__(config)
        self.quant_dtype = config.get("quant_dtype", "int") if config else "int"
        self.weight_bits = config.get("weight_bits", 8) if config else 8
        self.activation_bits = config.get("activation_bits", 8) if config else 8