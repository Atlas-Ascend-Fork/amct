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

from dataclasses import dataclass, field
from typing import Any, Callable, Optional


@dataclass(frozen=True)
class RegistryItem:
    name: str
    target: Any
    metadata: dict[str, Any] = field(default_factory=dict)


class Registry:
    def __init__(self, name: str):
        self._name = name
        self._items: dict[str, RegistryItem] = {}

    def __call__(self, obj: Optional[Callable] = None, **kwargs: Any):
        return self.register(obj=obj, **kwargs)

    def __contains__(self, key: object) -> bool:
        return key in self._items

    def __repr__(self):
        modules = ", ".join(sorted(self._items))
        return f"Registry(name='{self._name}', modules=[{modules}])"

    @property
    def name(self) -> str:
        return self._name

    def register(
        self,
        obj: Optional[Callable] = None,
        *,
        name: Optional[str] = None,
        force: bool = False,
        **metadata: Any,
    ):
        def decorator(target):
            key = name or target.__name__
            self._register(key, target, force=force, metadata=metadata)
            return target

        if obj is not None:
            return decorator(obj)
        return decorator

    def get(self, key: str) -> Any:
        item = self._items.get(key)
        if item is None:
            available = ", ".join(sorted(self._items)) or "<empty>"
            raise KeyError(
                f"'{key}' is not registered in '{self._name}'. "
                f"Available keys: {available}"
            )
        return item.target

    def get_item(self, key: str) -> RegistryItem:
        item = self._items.get(key)
        if item is None:
            available = ", ".join(sorted(self._items)) or "<empty>"
            raise KeyError(
                f"'{key}' is not registered in '{self._name}'. "
                f"Available keys: {available}"
            )
        return item

    def has(self, key: str) -> bool:
        return key in self._items

    def list_all(self) -> list[str]:
        return sorted(self._items)

    def items(self) -> dict[str, RegistryItem]:
        return dict(self._items)

    def _register(self, key: str, obj: Any, force: bool, metadata: dict[str, Any]):
        if key in self._items and not force:
            raise KeyError(
                f"'{key}' already registered in '{self._name}'. "
                f"Use @REGISTRY.register(force=True) to override."
            )
        self._items[key] = RegistryItem(name=key, target=obj, metadata=dict(metadata))
