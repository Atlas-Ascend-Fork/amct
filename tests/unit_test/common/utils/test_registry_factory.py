#!/usr/bin/env python3
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
import pytest

from amct_pytorch.common.utils.registry_factory import Registry, RegistryItem

BAR = 'bar'
BAZ = 'baz'


def test_register_via_decorator_uses_class_name():
    registry = Registry("test")

    @registry.register
    class Foo:
        pass

    assert "Foo" in registry
    assert registry.get("Foo") is Foo


def test_register_with_explicit_name_and_metadata():
    registry = Registry("test")

    @registry.register(name=BAR, description="hi", flag=True)
    class Bar:
        pass

    item = registry.get_item(BAR)
    assert isinstance(item, RegistryItem)
    assert item.name == BAR
    assert item.target is Bar
    assert item.metadata == {"description": "hi", "flag": True}


def test_call_alias_matches_register():
    registry = Registry("alias")

    @registry(name=BAZ)
    class Baz:
        pass

    assert registry.has(BAZ)
    assert registry.get(BAZ) is Baz


def test_duplicate_register_raises_without_force():
    registry = Registry("dup")

    @registry.register(name="x")
    class DummyAlgoA:
        pass

    with pytest.raises(KeyError, match="already registered"):
        @registry.register(name="x")
        class DummyAlgoB:
            pass


def test_duplicate_register_overrides_with_force():
    registry = Registry("dup")

    @registry.register(name="x")
    class DummyAlgoA:
        pass

    @registry.register(name="x", force=True)
    class DummyAlgoB:
        pass

    assert registry.get("x") is DummyAlgoB


def test_get_missing_lists_available_keys():
    registry = Registry("missing")

    @registry.register(name="alpha")
    class _A:
        pass

    @registry.register(name="beta")
    class _B:
        pass

    with pytest.raises(KeyError) as exc:
        registry.get("zeta")
    msg = str(exc.value)
    assert "zeta" in msg
    assert "alpha" in msg and "beta" in msg


def test_get_missing_on_empty_registry_shows_empty_marker():
    registry = Registry("empty")
    with pytest.raises(KeyError, match="<empty>"):
        registry.get("anything")


def test_list_all_returns_sorted_keys():
    registry = Registry("sort")
    for name in ("c", "a", "b"):
        registry.register(name=name)(type(name, (), {}))
    assert registry.list_all() == ["a", "b", "c"]


def test_items_returns_independent_copy():
    registry = Registry("copy")

    @registry.register(name="x")
    class _X:
        pass

    snapshot = registry.items()
    snapshot.pop("x")
    assert registry.has("x"), "mutating items() return value must not affect registry"


def test_repr_contains_name_and_keys():
    registry = Registry("display")
    registry.register(name="a")(type("A", (), {}))
    text = repr(registry)
    assert "display" in text
    assert "a" in text


def test_register_function_directly_without_decorator_call():
    registry = Registry("direct")

    def fn():
        return 42

    registry.register(fn, name="fn")
    assert registry.get("fn")() == 42


def test_get_item_missing_raises():
    registry = Registry("gi")
    with pytest.raises(KeyError, match="not registered"):
        registry.get_item("nope")

