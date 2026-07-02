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
"""Regression guard for ``setup.py`` ``package_data`` of the graph-based package.

Background: the graph-based compression subpackage was moved from
``amct_pytorch/graph_based_compression`` to ``amct_pytorch/classic/graph_based``
(commit ``facaf0e``). The ``package_data`` entry in ``setup.py`` kept pointing at
the old, now-nonexistent package name, so the protoc-generated ``*_pb2.py``
modules, the ``*.csv`` capacity tables and the ``*.so`` libraries were silently
NOT shipped in the built wheel/sdist.

The visible symptom: after ``pip install``, ``import amct_pytorch`` raised what
looked like a circular-import error, but the real cause was the missing
``.csv``/``.so`` resources (and a proto package that ``find_packages`` could no
longer reach) under the renamed package.

These checks parse ``setup.py`` with ``ast`` -- no build required -- and assert
that the ``package_data`` key matches the real on-disk layout, and that the
proto subpackage stays discoverable by ``find_packages`` (which is what actually
ships the generated ``*_pb2.py`` modules -- ``package_data`` only carries the
non-``.py`` resources). They fail loudly if the layout drifts again.
"""
import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SETUP_PY = REPO_ROOT / "setup.py"

# Current package that owns the graph-based resources.
GRAPH_PACKAGE = "amct_pytorch.classic.graph_based"
# The retired name that must never reappear as a package_data key.
LEGACY_PACKAGE = "amct_pytorch.classic.graph_based_compression"

# Directory (relative to GRAPH_PACKAGE root) holding the .proto sources and the
# protoc-generated *_pb2.py modules that the proto package imports at load time.
PROTO_REL_DIR = "amct_pytorch/proto"

# package_data globs whose target directory only exists *after* a build (the
# files are compiler/protoc artifacts, not tracked in git). Their directories
# are absent on a fresh checkout, so existence cannot be asserted -- only that
# the relative path stays anchored inside the package tree.
BUILD_ARTIFACT_DIRS = frozenset({"lib"})


def _returned_dict(func_node):
    """Return the first ``ast.Dict`` returned inside ``func_node``, or None."""
    for sub in ast.walk(func_node):
        if isinstance(sub, ast.Return) and isinstance(sub.value, ast.Dict):
            return sub.value
    return None


def _package_data():
    """Return the dict literal returned by ``get_package_data()`` in setup.py."""
    tree = ast.parse(SETUP_PY.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not (isinstance(node, ast.FunctionDef) and node.name == "get_package_data"):
            continue
        dict_node = _returned_dict(node)
        if dict_node is not None:
            return _literal_dict(dict_node)
    raise AssertionError("get_package_data() with a dict literal not found in setup.py")


def _literal_dict(dict_node):
    """Convert an ast.Dict of str -> [str, ...] into a real dict."""
    result = {}
    for key_node, val_node in zip(dict_node.keys, dict_node.values):
        if not isinstance(key_node, ast.Constant):
            continue
        patterns = [
            e.value for e in getattr(val_node, "elts", [])
            if isinstance(e, ast.Constant)
        ]
        result[key_node.value] = patterns
    return result


def test_package_data_key_uses_current_layout():
    """package_data must key on the current package, not the retired name.

    This is the exact mismatch that stopped the generated pb2/csv/so resources
    from being packaged after the ``graph_based_compression`` -> ``graph_based``
    move. Guarding the key catches the regression without building anything.
    """
    data = _package_data()
    assert GRAPH_PACKAGE in data, (
        f"setup.py package_data must contain key '{GRAPH_PACKAGE}' so the "
        f"graph-based proto/csv/so resources are shipped. Found keys: "
        f"{sorted(data)}"
    )
    assert LEGACY_PACKAGE not in data, (
        f"setup.py package_data still references the retired package "
        f"'{LEGACY_PACKAGE}'; this package no longer exists, so its resources "
        f"are never packaged."
    )


def test_proto_package_is_discoverable_by_find_packages():
    """The proto subpackage must be importable as a package after install.

    The protoc-generated ``*_pb2.py`` modules ship as ordinary package modules
    via ``setuptools.find_packages()`` -- NOT via ``package_data`` (which only
    carries non-``.py`` resources like ``.proto``/``.csv``/``.so``). For that to
    work the proto directory must be a real package, i.e. carry an
    ``__init__.py``. If it loses its ``__init__.py``, ``find_packages`` skips it,
    the ``*_pb2.py`` modules never ship, and ``import amct_pytorch`` breaks after
    install with a misleading circular-import error.
    """
    proto_dir = REPO_ROOT / GRAPH_PACKAGE.replace(".", "/") / PROTO_REL_DIR
    init_path = proto_dir / "__init__.py"
    assert init_path.is_file(), (
        f"proto package must carry an __init__.py at {init_path} so "
        f"find_packages() discovers it and ships the generated *_pb2.py modules; "
        f"package_data does not carry .py files."
    )


def test_package_data_patterns_point_at_existing_dirs():
    """Source-tracked package_data dirs must exist; build-artifact dirs must be anchored.

    A stale relative path packages nothing silently. For directories that hold
    git-tracked sources (proto, capacity) we assert the directory exists. For
    build-artifact dirs (``lib`` -- the ``.so`` is compiled, not committed) the
    directory is absent on a fresh checkout, so we only assert the glob root
    stays inside the package tree, catching a ``common/proto``-style dead path
    without requiring a build.
    """
    pkg_root = REPO_ROOT / GRAPH_PACKAGE.replace(".", "/")
    patterns = _package_data().get(GRAPH_PACKAGE, [])
    assert patterns, f"no package_data patterns declared for '{GRAPH_PACKAGE}'"

    missing_dirs = []
    escaped_dirs = []
    for pattern in patterns:
        glob_dir = (pkg_root / pattern).parent
        top = Path(pattern).parts[0]
        if top in BUILD_ARTIFACT_DIRS:
            # Build artifact: dir may not exist yet, but the path must resolve
            # inside the package tree (not via .. or an absolute escape).
            if pkg_root not in glob_dir.resolve().parents and \
                    glob_dir.resolve() != pkg_root.resolve():
                escaped_dirs.append(f"{pattern} -> {glob_dir}")
            continue
        if not glob_dir.is_dir():
            missing_dirs.append(f"{pattern} -> {glob_dir}")
    assert not missing_dirs, (
        f"package_data patterns for '{GRAPH_PACKAGE}' reference non-existent "
        f"directories (relative to {pkg_root}): {missing_dirs}"
    )
    assert not escaped_dirs, (
        f"build-artifact package_data patterns for '{GRAPH_PACKAGE}' resolve "
        f"outside the package tree (relative to {pkg_root}): {escaped_dirs}"
    )


def test_proto_init_pb2_imports_have_matching_proto_sources():
    """Each ``*_pb2`` imported by proto/__init__.py has a ``.proto`` to build it.

    Ties the runtime import contract to the packaged sources: if a pb2 module is
    imported but its ``.proto`` is gone (or renamed), the build cannot generate
    it and the install breaks. Pure static check, no build required.
    """
    proto_dir = REPO_ROOT / GRAPH_PACKAGE.replace(".", "/") / PROTO_REL_DIR
    init_path = proto_dir / "__init__.py"
    if not init_path.is_file():
        pytest.skip(f"proto package __init__ not found at {init_path}")

    tree = ast.parse(init_path.read_text(encoding="utf-8"))
    pb2_modules = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level >= 1:
            pb2_modules.extend(
                a.name for a in node.names if a.name.endswith("_pb2")
            )
    assert pb2_modules, f"no relative *_pb2 imports found in {init_path}"

    missing = []
    for mod in pb2_modules:
        proto_name = mod[: -len("_pb2")] + ".proto"
        if not (proto_dir / proto_name).is_file():
            missing.append(f"{mod} (expected {proto_name})")
    assert not missing, (
        f"proto/__init__.py imports pb2 modules without a matching .proto "
        f"source in {proto_dir}: {missing}"
    )
