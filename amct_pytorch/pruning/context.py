# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Iterator, Mapping, Optional, Tuple

import torch


BatchAdapter = Callable[[Any], Tuple[Tuple[Any, ...], Dict[str, Any]]]


def as_batch_sequence(data: Any) -> Iterable[Any]:
    """Interpret ``data`` as a sequence of batches without materialising it.

    A Mapping (dict / BatchEncoding / OrderedDict) is one batch and must not be iterated
    into its keys; ``None`` carries no batches; every other iterable (list / tuple /
    DataLoader / generator / Tensor) already is a sequence of batches.
    """
    if data is None:
        return ()
    if isinstance(data, Mapping):
        return (data,)
    return data


@dataclass
class PruneContext:
    data: Optional[Any] = None
    batch_adapter: Optional[BatchAdapter] = None

    def iter_model_inputs(self) -> Iterator[Tuple[Tuple[Any, ...], Dict[str, Any]]]:
        adapter = self.batch_adapter or default_batch_adapter

        def generator() -> Iterator[Tuple[Tuple[Any, ...], Dict[str, Any]]]:
            for batch in as_batch_sequence(self.data):
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
