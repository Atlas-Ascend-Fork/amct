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
import torch
import torch.nn as nn

from amct_pytorch.common.models.llm.common.ptq_params import (
    PtqParamHandler,
    PtqParamStore,
)
from amct_pytorch.common.models.llm.common.ptq_units import make_ptq_unit

MLP_UNIT = 'mlp'

# ---- PtqParamHandler.export_trainable_module ----------------------------

LOADED = 'loaded'
TRAINABLE = 'trainable'

MISSING = 'missing'


def test_export_trainable_module_collects_only_trainable_params():
    m = nn.Module()
    m.trainable = nn.Parameter(torch.ones(3))
    m.frozen = nn.Parameter(torch.zeros(3), requires_grad=False)

    out = PtqParamHandler.export_trainable_module(m)
    assert set(out) == {TRAINABLE}
    assert torch.equal(out[TRAINABLE], torch.ones(3))
    assert out[TRAINABLE].device.type == "cpu"


def test_export_trainable_module_returns_empty_when_no_grad_params():
    m = nn.Module()
    m.frozen = nn.Parameter(torch.zeros(3), requires_grad=False)
    assert not PtqParamHandler.export_trainable_module(m)


# ---- PtqParamHandler.export_module --------------------------------------


class _Exportable(nn.Module):
    def __init__(self, payload):
        super().__init__()
        self._payload = payload

    def export_ptq_params(self):
        return self._payload


def test_export_module_collects_named_submodule_params():
    parent = nn.Module()
    parent.a = _Exportable({"x": torch.tensor([1.0])})
    parent.b = _Exportable({"y": torch.tensor([2.0])})

    out = PtqParamHandler.export_module(parent)
    assert set(out) == {"a", "b"}
    assert torch.equal(out["a"]["x"], torch.tensor([1.0]))
    assert torch.equal(out["b"]["y"], torch.tensor([2.0]))


def test_export_module_skips_root_module():
    # A module that itself has export_ptq_params should not appear under "" key.
    parent = _Exportable({"k": torch.tensor([1.0])})
    parent.add_module("child", _Exportable({"k": torch.tensor([2.0])}))
    out = PtqParamHandler.export_module(parent)
    assert "" not in out
    assert "child" in out


def test_export_module_skips_submodules_returning_empty_or_none():
    parent = nn.Module()
    parent.empty = _Exportable({})
    parent.none = _Exportable(None)
    parent.real = _Exportable({"k": torch.zeros(1)})

    out = PtqParamHandler.export_module(parent)
    assert set(out) == {"real"}


def test_export_module_ignores_modules_without_export_ptq_params():
    parent = nn.Module()
    parent.linear = nn.Linear(4, 4)
    parent.flagged = _Exportable({"k": torch.zeros(1)})

    out = PtqParamHandler.export_module(parent)
    assert set(out) == {"flagged"}


# ---- PtqParamHandler.load_trainable_module / load_module ----------------


def test_load_trainable_module_copies_into_named_params():
    m = nn.Module()
    m.weight = nn.Parameter(torch.zeros(3))
    PtqParamHandler.load_trainable_module(m, {"weight": torch.tensor([1.0, 2.0, 3.0])})
    assert torch.equal(m.weight.data, torch.tensor([1.0, 2.0, 3.0]))


def test_load_trainable_module_raises_on_unknown_param():
    m = nn.Module()
    m.weight = nn.Parameter(torch.zeros(3))
    with pytest.raises(KeyError, match=f"Parameter '{MISSING}'"):
        PtqParamHandler.load_trainable_module(m, {MISSING: torch.zeros(3)})


class _Loadable(nn.Module):
    def __init__(self):
        super().__init__()
        self.received = None

    def load_ptq_params(self, params):
        self.received = params


def test_load_module_dispatches_to_named_submodules():
    parent = nn.Module()
    parent.a = _Loadable()
    parent.b = _Loadable()
    PtqParamHandler.load_module(parent, {"a": {"k": 1}, "b": {"k": 2}})
    assert parent.a.received == {"k": 1}
    assert parent.b.received == {"k": 2}


def test_load_module_raises_when_submodule_missing():
    parent = nn.Module()
    parent.a = _Loadable()
    with pytest.raises(KeyError, match=f"Submodule '{MISSING}'"):
        PtqParamHandler.load_module(parent, {MISSING: {}})


def test_load_module_raises_when_submodule_has_no_load_ptq_params():
    parent = nn.Module()
    parent.linear = nn.Linear(4, 4)  # no load_ptq_params method
    with pytest.raises(KeyError, match="does not implement load_ptq_params"):
        PtqParamHandler.load_module(parent, {"linear": {}})


# ---- PtqParamHandler.export_unit / load_unit ----------------------------


def test_export_unit_uses_module_export_ptq_params_when_available():
    handler = PtqParamHandler()
    unit = make_ptq_unit(MLP_UNIT, MLP_UNIT, 0, module=_Exportable({"k": torch.tensor([1.0])}))
    out = handler.export_unit(unit)
    assert torch.equal(out["k"], torch.tensor([1.0]))


def test_export_unit_falls_back_to_export_module_then_trainable():
    handler = PtqParamHandler()

    # A nested module without top-level export_ptq_params, but with a nested one.
    parent = nn.Module()
    parent.child = _Exportable({"k": torch.tensor([5.0])})
    unit = make_ptq_unit(MLP_UNIT, MLP_UNIT, 0, module=parent)
    out = handler.export_unit(unit)
    assert "child" in out

    # If both submodule and trainable export are empty, expect empty dict from
    # the trainable fallback (no requires_grad params).
    empty_parent = nn.Module()
    empty_parent.frozen = nn.Parameter(torch.zeros(2), requires_grad=False)
    empty_unit = make_ptq_unit(MLP_UNIT, MLP_UNIT, 0, module=empty_parent)
    assert handler.export_unit(empty_unit) == {}


def test_load_unit_dispatches_to_module_load_ptq_params_when_available():
    handler = PtqParamHandler()
    mod = _Loadable()
    unit = make_ptq_unit(MLP_UNIT, MLP_UNIT, 0, module=mod)
    handler.load_unit(unit, {"k": 1})
    assert mod.received == {"k": 1}


def test_load_unit_routes_dict_of_dicts_to_load_module():
    handler = PtqParamHandler()
    parent = nn.Module()
    parent.child = _Loadable()
    unit = make_ptq_unit(MLP_UNIT, MLP_UNIT, 0, module=parent)
    handler.load_unit(unit, {"child": {"k": 1}})
    assert parent.child.received == {"k": 1}


def test_load_unit_routes_flat_dict_to_trainable_loader():
    handler = PtqParamHandler()
    parent = nn.Module()
    parent.weight = nn.Parameter(torch.zeros(2))
    unit = make_ptq_unit(MLP_UNIT, MLP_UNIT, 0, module=parent)
    handler.load_unit(unit, {"weight": torch.tensor([1.0, 2.0])})
    assert torch.equal(parent.weight.data, torch.tensor([1.0, 2.0]))


# ---- PtqParamStore -------------------------------------------------------


def _make_store():
    return PtqParamStore(PtqParamHandler(), iter_ptq_units_fn=lambda layer_idx, block: iter([]))


def test_load_saved_unit_layer_indexed_filename(tmp_path):
    handler = PtqParamHandler()
    store = PtqParamStore(handler, iter_ptq_units_fn=lambda *a: iter([]))
    mod = _Loadable()
    unit = make_ptq_unit(MLP_UNIT, MLP_UNIT, layer_idx=3, module=mod)
    payload = {"k": 1}
    torch.save(payload, tmp_path / "layer_3_mlp.pt")
    assert store.load_saved_unit(str(tmp_path), unit) is True
    assert mod.received == {"k": 1}


def test_load_saved_unit_unindexed_filename_when_layer_idx_none(tmp_path):
    store = PtqParamStore(PtqParamHandler(), iter_ptq_units_fn=lambda *a: iter([]))
    mod = _Loadable()
    unit = make_ptq_unit("global", "global", layer_idx=None, module=mod)
    torch.save({"k": 9}, tmp_path / "global.pt")
    assert store.load_saved_unit(str(tmp_path), unit) is True
    assert mod.received == {"k": 9}


def test_load_saved_unit_returns_false_on_missing_when_not_strict(tmp_path):
    store = PtqParamStore(PtqParamHandler(), iter_ptq_units_fn=lambda *a: iter([]))
    unit = make_ptq_unit(MLP_UNIT, MLP_UNIT, layer_idx=0, module=_Loadable())
    assert store.load_saved_unit(str(tmp_path), unit, strict=False) is False


def test_load_saved_unit_raises_on_missing_when_strict(tmp_path):
    store = PtqParamStore(PtqParamHandler(), iter_ptq_units_fn=lambda *a: iter([]))
    unit = make_ptq_unit(MLP_UNIT, MLP_UNIT, layer_idx=0, module=_Loadable())
    with pytest.raises(FileNotFoundError, match="not found"):
        store.load_saved_unit(str(tmp_path), unit, strict=True)


def test_load_layer_collects_loaded_and_missing(tmp_path):
    units = [
        make_ptq_unit(MLP_UNIT, LOADED, layer_idx=2, module=_Loadable()),
        make_ptq_unit(MLP_UNIT, MISSING, layer_idx=2, module=_Loadable()),
    ]
    store = PtqParamStore(
        PtqParamHandler(), iter_ptq_units_fn=lambda layer_idx, block: iter(units)
    )
    torch.save({"k": 1}, tmp_path / "layer_2_loaded.pt")

    result = store.load_layer(layer_idx=2, block=None, param_dir=str(tmp_path))
    assert result == {LOADED: [LOADED], MISSING: [MISSING]}


def test_load_layer_strict_raises_when_any_missing(tmp_path):
    # Note: strict=True is forwarded to load_saved_unit, which raises per-unit
    # before load_layer's aggregate check is reached.
    units = [make_ptq_unit(MLP_UNIT, MISSING, layer_idx=0, module=_Loadable())]
    store = PtqParamStore(
        PtqParamHandler(), iter_ptq_units_fn=lambda layer_idx, block: iter(units)
    )
    with pytest.raises(FileNotFoundError, match=f"not found for unit '{MISSING}'"):
        store.load_layer(layer_idx=0, block=None, param_dir=str(tmp_path), strict=True)


def test_ptq_param_store_load_layer_strict_raises_on_missing(tmp_path):
    def _fake_iter_units(layer_idx, block):
        unit = make_ptq_unit(MLP_UNIT, MLP_UNIT, layer_idx=layer_idx, module=block)
        yield unit
    handler = PtqParamHandler()
    store = PtqParamStore(handler, _fake_iter_units)
    block = nn.Module()
    with pytest.raises(FileNotFoundError, match="PTQ params not found"):
        store.load_layer(0, block, str(tmp_path), strict=True)


def test_ptq_param_store_load_layer_nonstrict_missing_ok(tmp_path):
    def _fake_iter_units(layer_idx, block):
        unit = make_ptq_unit(MLP_UNIT, MLP_UNIT, layer_idx=layer_idx, module=block)
        yield unit
    handler = PtqParamHandler()
    store = PtqParamStore(handler, _fake_iter_units)
    block = nn.Module()
    result = store.load_layer(0, block, str(tmp_path), strict=False)
    assert not result[LOADED]
    assert result[MISSING] == [MLP_UNIT]


def test_ptq_param_store_init():
    handler = PtqParamHandler()
    store = PtqParamStore(handler, lambda layer, block: [])
    assert store is not None

