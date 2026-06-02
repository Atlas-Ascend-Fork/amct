from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Iterator, Optional, Tuple

import torch


BatchAdapter = Callable[[Any], Tuple[Tuple[Any, ...], Dict[str, Any]]]


@dataclass(slots=True)
class PruneContext:
    data: Optional[Iterable[Any]] = None
    batch_adapter: Optional[BatchAdapter] = None

    def iter_model_inputs(self) -> Iterator[Tuple[Tuple[Any, ...], Dict[str, Any]]]:
        if self.data is None:
            return iter(())
        adapter = self.batch_adapter or default_batch_adapter

        def generator() -> Iterator[Tuple[Tuple[Any, ...], Dict[str, Any]]]:
            for batch in self.data:
                yield adapter(batch)

        return generator()


def default_batch_adapter(batch: Any) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
    if isinstance(batch, dict):
        return (), dict(batch)
    if torch.is_tensor(batch):
        return (batch,), {}
    if isinstance(batch, tuple):
        return batch, {}
    if isinstance(batch, list):
        return tuple(batch), {}
    raise TypeError(
        "Unsupported batch type for pruning data collection. "
        "Provide torch.Tensor / tuple / list / dict or pass a custom batch_adapter."
    )
