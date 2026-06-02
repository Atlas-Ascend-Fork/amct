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
"""Regression guard for the top-level ``amct_pytorch`` public API.

Background: the graph-based interfaces (``create_quant_config`` and friends)
are defined in ``amct_pytorch.classic.graph_based.amct_pytorch`` and surfaced at
the top level by a single wildcard re-export in ``amct_pytorch/__init__.py``.
Refactoring the package layout once dropped that line, which silently removed
~25 interfaces from ``amct_pytorch.*``. These tests make that regression fail
loudly.

The static checks parse source with ``ast`` and need no built package, so they
run in any CI environment. The behavioural check imports the real package and is
skipped when the generated protobuf modules are absent (unbuilt source tree).
"""
import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
TOP_INIT = REPO_ROOT / "amct_pytorch" / "__init__.py"
GRAPH_INIT = (
    REPO_ROOT
    / "amct_pytorch"
    / "classic"
    / "graph_based"
    / "amct_pytorch"
    / "__init__.py"
)

# Module whose ``__all__`` must reach the top level via wildcard re-export.
GRAPH_MODULE = "amct_pytorch.classic.graph_based.amct_pytorch"

# Baseline anchor: a hand-picked subset that MUST always be public. This guards
# against the ``__all__`` list being accidentally emptied or gutted -- in that
# case deriving the expected set purely from ``__all__`` would vacuously pass,
# so we assert these names are always present (see the subset test below).
BASELINE_INTERFACES = (
    "create_quant_config",
    "quantize_model",
    "save_model",
    "create_quant_cali_config",
    "create_quant_retrain_config",
    "accuracy_based_auto_calibration",
    "create_distill_config",
    "auto_channel_prune_search",
)


def _wildcard_reexport_targets(init_path):
    """Return modules that ``init_path`` re-exports via ``from X import *``."""
    tree = ast.parse(init_path.read_text(encoding="utf-8"))
    targets = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            if any(alias.name == "*" for alias in node.names):
                targets.append(node.module)
    return targets


def _module_dunder_all(init_path):
    """Return the literal ``__all__`` list defined in ``init_path``."""
    tree = ast.parse(init_path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "__all__":
                    return [
                        elt.value
                        for elt in node.value.elts
                        if isinstance(elt, ast.Constant)
                    ]
    return []


def test_top_init_reexports_graph_based_module():
    """The wildcard re-export of the graph-based package must be present.

    This is the exact line whose removal hid ``create_quant_config`` from the
    top-level namespace. Guarding it statically catches the regression even
    when the package cannot be imported (protobuf not generated).
    """
    targets = _wildcard_reexport_targets(TOP_INIT)
    assert GRAPH_MODULE in targets, (
        f"amct_pytorch/__init__.py must re-export '{GRAPH_MODULE}' via "
        f"'from {GRAPH_MODULE} import *' so graph-based interfaces such as "
        f"create_quant_config stay reachable as amct_pytorch.*. "
        f"Found wildcard re-exports: {targets}"
    )


def test_graph_module_all_covers_baseline_interfaces():
    """The graph module ``__all__`` must contain every baseline interface.

    Deriving the "expected top-level API" from ``__all__`` alone would pass
    vacuously if ``__all__`` were emptied. This anchors a known-good subset so
    that silently dropping any of these names fails the suite.
    """
    exported = set(_module_dunder_all(GRAPH_INIT))
    missing = [name for name in BASELINE_INTERFACES if name not in exported]
    assert not missing, (
        f"Baseline public interfaces missing from __all__ of {GRAPH_MODULE}: "
        f"{missing}. Declared names: {sorted(exported)}"
    )


def test_top_level_interfaces_importable_when_built():
    """Behavioural guard: every ``__all__`` name is reachable as ``amct_pytorch.*``.

    Skipped on an unbuilt source tree (generated ``*_pb2.py`` missing); runs as a
    true regression test wherever the package is installed/built. Covers the full
    graph-module ``__all__`` (not just the baseline), so any interface dropped
    from the top-level namespace is caught.
    """
    try:
        import amct_pytorch
    except ImportError as exc:
        pytest.skip(f"amct_pytorch not importable in this environment: {exc}")

    expected = _module_dunder_all(GRAPH_INIT)
    assert expected, "graph module __all__ unexpectedly empty"
    missing = [name for name in expected if not hasattr(amct_pytorch, name)]
    assert not missing, (
        f"amct_pytorch is missing top-level interfaces {missing}; the "
        f"graph-based wildcard re-export in __init__.py was likely dropped."
    )
