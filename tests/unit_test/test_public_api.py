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
import importlib
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

# Non-graph interfaces exposed at the top level, grouped by their source module.
# These are bound by explicit ``from <module> import ...`` statements (not the
# graph wildcard), so they have a different failure mode worth guarding too.
CLASSIC_SOURCE = "amct_pytorch.classic"
CLASSIC_INTERFACES = ("quantize", "convert", "algorithm_register")

CONFIG_SOURCE = "amct_pytorch.common.config"
CONFIG_INTERFACES = (
    "INT4_AWQ_WEIGHT_QUANT_CFG",
    "INT4_GPTQ_WEIGHT_QUANT_CFG",
    "INT8_SMOOTHQUANT_CFG",
    "INT8_MINMAX_WEIGHT_QUANT_CFG",
    "HIFP8_OFMR_CFG",
    "FP8_OFMR_CFG",
    "MXFP8_QUANT_CFG",
    "MXFP4_AWQ_WEIGHT_QUANT_CFG",
    "HIFP8_CAST_CFG",
    "HIFP8_QUANTILE_CFG",
)

# name -> source module that must explicitly import it into the top level.
NON_GRAPH_INTERFACES = {name: CLASSIC_SOURCE for name in CLASSIC_INTERFACES}
NON_GRAPH_INTERFACES.update({name: CONFIG_SOURCE for name in CONFIG_INTERFACES})


def _module_dunder_all(init_path):
    """Return the literal ``__all__`` list defined in ``init_path``."""
    tree = ast.parse(init_path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        names = {t.id for t in node.targets if isinstance(t, ast.Name)}
        if "__all__" not in names:
            continue
        return [e.value for e in node.value.elts if isinstance(e, ast.Constant)]
    return []


def _explicit_imported_names(init_path, module):
    """Return names explicitly imported via ``from <module> import ...``."""
    tree = ast.parse(init_path.read_text(encoding="utf-8"))
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == module:
            names.extend(a.name for a in node.names if a.name != "*")
    return names


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
        importlib.import_module(GRAPH_MODULE)
    except ImportError as exc:
        pytest.skip(
            f"graph-based package not importable in this environment: {exc}")

    expected = _module_dunder_all(GRAPH_INIT)
    assert expected, "graph module __all__ unexpectedly empty"
    missing = [name for name in expected if not hasattr(amct_pytorch, name)]
    assert not missing, (
        f"amct_pytorch is missing top-level interfaces {missing}; the "
        f"graph-based wildcard re-export in __init__.py was likely dropped."
    )


@pytest.mark.parametrize("name", sorted(NON_GRAPH_INTERFACES))
def test_non_graph_interface_in_top_all(name):
    """Each non-graph public interface stays declared in the top-level ``__all__``.

    Static check (no build needed): guards the contract that ``quantize`` /
    ``convert`` / ``algorithm_register`` and the ``*_CFG`` config constants are
    advertised as part of the public API.
    """
    top_all = _module_dunder_all(TOP_INIT)
    assert name in top_all, (
        f"'{name}' dropped from amct_pytorch.__all__; it is part of the public "
        f"top-level API. Declared names: {top_all}"
    )


@pytest.mark.parametrize("name", sorted(NON_GRAPH_INTERFACES))
def test_non_graph_interface_explicitly_imported(name):
    """Each non-graph interface is bound by an explicit ``from <src> import``.

    Static check: presence in ``__all__`` alone does not bind the name; the
    matching import must exist or ``amct_pytorch.<name>`` raises AttributeError.
    """
    source = NON_GRAPH_INTERFACES[name]
    imported = _explicit_imported_names(TOP_INIT, source)
    assert name in imported, (
        f"'{name}' is in __all__ but not imported from '{source}' in "
        f"amct_pytorch/__init__.py, so amct_pytorch.{name} would fail. "
        f"Names imported from {source}: {imported}"
    )


def test_config_interfaces_importable():
    """Behavioural guard for the config constants (build-independent).

    ``amct_pytorch.common.config`` does not pull in the graph protobuf chain, so
    this runs in any environment and verifies the constants resolve to real
    objects at their source module.
    """
    config = importlib.import_module(CONFIG_SOURCE)
    missing = [n for n in CONFIG_INTERFACES if not hasattr(config, n)]
    assert not missing, (
        f"{CONFIG_SOURCE} is missing config constants {missing}; the top-level "
        f"amct_pytorch re-export of these would break."
    )


def test_non_graph_interfaces_importable_when_built():
    """Behavioural guard: non-graph interfaces reachable as ``amct_pytorch.*``.

    Skipped on an unbuilt source tree (``amct_pytorch.classic`` transitively
    imports the graph protobuf chain). Where built, asserts every non-graph
    public name actually resolves on the top-level package.
    """
    try:
        import amct_pytorch
    except ImportError as exc:
        pytest.skip(f"amct_pytorch not importable in this environment: {exc}")

    missing = [n for n in NON_GRAPH_INTERFACES if not hasattr(amct_pytorch, n)]
    assert not missing, (
        f"amct_pytorch is missing non-graph top-level interfaces {missing}; an "
        f"explicit import in __init__.py was likely dropped."
    )
