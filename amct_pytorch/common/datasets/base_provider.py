from abc import ABC, abstractmethod
from typing import Iterator, Any

class BaseDataProvider(ABC):
    @abstractmethod
    def __iter__(self) -> Iterator[Any]:
        pass

    @abstractmethod
    def get_batch(self, batch_size: int) -> Any:
        pass

    @abstractmethod
    def get_sample_count(self) -> int:
        pass