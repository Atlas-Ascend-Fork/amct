from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional

class BaseModelAdapter(ABC):
    def __init__(self, model_name: str, model_config: Dict[str, Any] = None):
        self.model_name = model_name
        self.model_config = model_config or {}
        self.model_type = self._detect_model_type()

    def _detect_model_type(self) -> str:
        return "classic"

    @abstractmethod
    def get_quantizable_layers(self, model) -> List[str]:
        pass

    @abstractmethod
    def get_layer_type(self, layer_name: str) -> str:
        pass

    @classmethod
    def is_llm_adapter(cls) -> bool:
        return False

    @classmethod
    def is_classic_adapter(cls) -> bool:
        return True

    def get_model_info(self) -> Dict[str, Any]:
        return {"model_name": self.model_name, "model_type": self.model_type, "config": self.model_config}