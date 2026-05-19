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

from amct_pytorch.common.models.llm.common.ptq_units import (
    PtqUnit,
    iter_indexed_units,
    make_ptq_unit,
)

EXPERT_IDX = 'expert_idx'
MLP = 'mlp'

SRC = 'src'


def test_make_ptq_unit_defaults_metadata_to_empty_dict():
    unit = make_ptq_unit(MLP, MLP, 3, module="m")
    assert isinstance(unit, PtqUnit)
    assert unit.kind == MLP and unit.name == MLP
    assert unit.layer_idx == 3 and unit.module == "m"
    assert unit.metadata == {}


def test_make_ptq_unit_preserves_provided_metadata():
    unit = make_ptq_unit("moe", "expert_0", 1, module="m", metadata={EXPERT_IDX: 0})
    assert unit.metadata == {EXPERT_IDX: 0}


def test_save_name_replaces_dots_with_underscores():
    assert make_ptq_unit(MLP, "block.mlp.up_proj", 0, module=None).save_name == "block_mlp_up_proj"


def test_iter_indexed_units_skips_none_modules():
    items = ["m0", None, "m2"]
    units = list(iter_indexed_units(
        kind="moe", name_prefix="expert", layer_idx=4, items=items
    ))
    assert [u.name for u in units] == ["expert_0", "expert_2"]
    assert [u.module for u in units] == ["m0", "m2"]
    assert all(u.kind == "moe" and u.layer_idx == 4 for u in units)


def test_iter_indexed_units_module_fn_overrides_item_as_module():
    items = [1, 2, 3]
    units = list(iter_indexed_units(
        kind="moe",
        name_prefix="expert",
        layer_idx=0,
        items=items,
        module_fn=lambda idx, item: f"mod-{item * 10}",
    ))
    assert [u.module for u in units] == ["mod-10", "mod-20", "mod-30"]


def test_iter_indexed_units_module_fn_returning_none_is_skipped():
    items = ["a", "b", "c"]
    units = list(iter_indexed_units(
        kind="moe",
        name_prefix="expert",
        layer_idx=0,
        items=items,
        module_fn=lambda idx, _: None if idx == 1 else _,
    ))
    assert [u.name for u in units] == ["expert_0", "expert_2"]


def test_iter_indexed_units_metadata_fn_attaches_per_item_metadata():
    items = ["a", "b"]
    units = list(iter_indexed_units(
        kind="moe",
        name_prefix="e",
        layer_idx=2,
        items=items,
        metadata_fn=lambda idx, item: {EXPERT_IDX: idx, SRC: item},
    ))
    assert units[0].metadata == {EXPERT_IDX: 0, SRC: "a"}
    assert units[1].metadata == {EXPERT_IDX: 1, SRC: "b"}

