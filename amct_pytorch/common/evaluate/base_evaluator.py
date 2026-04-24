from abc import ABC, abstractmethod
from typing import Dict, Any

class BaseEvaluator(ABC):
    @abstractmethod
    def evaluate(self, model, eval_data) -> Dict[str, float]:
        pass

    @abstractmethod
    def get_metrics(self) -> Dict[str, Any]:
        pass