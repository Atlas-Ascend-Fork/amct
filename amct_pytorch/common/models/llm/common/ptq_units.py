# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------

from typing import Callable, Iterable, Optional, TypeVar
from dataclasses import dataclass, field
from typing import Any, Optional


T = TypeVar("T")


@dataclass
class PtqUnit:
    kind: str
    name: str
    layer_idx: Optional[int]
    module: Any
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def save_name(self) -> str:
        return self.name.replace(".", "_")


def make_ptq_unit(kind: str, name: str, layer_idx: int | None, module, metadata: Optional[dict] = None) -> PtqUnit:
    return PtqUnit(
        kind=kind,
        name=name,
        layer_idx=layer_idx,
        module=module,
        metadata={} if metadata is None else metadata,
    )


def iter_indexed_units(
    kind: str,
    name_prefix: str,
    layer_idx: int | None,
    items: Iterable[T],
    module_fn: Optional[Callable[[int, T], object]] = None,
    metadata_fn: Optional[Callable[[int, T], Optional[dict]]] = None,
):
    for idx, item in enumerate(items):
        module = module_fn(idx, item) if module_fn is not None else item
        if module is None:
            continue
        metadata = metadata_fn(idx, item) if metadata_fn is not None else None
        yield make_ptq_unit(
            kind=kind,
            name=f"{name_prefix}_{idx}",
            layer_idx=layer_idx,
            module=module,
            metadata=metadata,
        )
