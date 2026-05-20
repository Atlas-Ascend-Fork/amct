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

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------
"""Tests for BitPolicy / _GroupBits / _check_complete / _has_lt_16."""

from types import SimpleNamespace

import pytest

from amct_pytorch.quantization.bit_policy import (
    BitPolicy,
    LayerBits,
    _check_complete,
    _GroupBits,
    _has_lt_16,
    ensure_bit_policy,
)

# ---- BitPolicy.__init__ ----------------------------------------------------

W_BITS = 'w_bits'

A_BITS = 'a_bits'


def test_init_defaults_to_16():
    bp = BitPolicy()
    assert bp.w_bits == 16
    assert bp.a_bits == 16
    assert bp.has_quant_linear() is False
    assert bp.has_quant_cache() is False


def test_init_raises_on_incomplete_top_level_bits():
    with pytest.raises(ValueError, match="Incomplete bit entry at top level"):
        BitPolicy({W_BITS: 8})


def test_init_cfg_none_is_treated_as_empty():
    bp = BitPolicy(None)
    assert bp.cfg == {}


# ---- BitPolicy.from_yaml ---------------------------------------------------


def test_from_yaml_loads_valid_config(tmp_path):
    yaml_file = tmp_path / "bits.yaml"
    yaml_file.write_text("w_bits: 4\na_bits: 8\n")
    bp = BitPolicy.from_yaml(str(yaml_file))
    assert bp.w_bits == 4
    assert bp.a_bits == 8


def test_from_yaml_empty_file_returns_defaults(tmp_path):
    yaml_file = tmp_path / "empty.yaml"
    yaml_file.write_text("")
    bp = BitPolicy.from_yaml(str(yaml_file))
    assert bp.w_bits == 16
    assert bp.a_bits == 16


def test_from_yaml_raises_on_non_mapping(tmp_path):
    yaml_file = tmp_path / "list.yaml"
    yaml_file.write_text("- item\n")
    with pytest.raises(ValueError, match="must be a mapping at top level"):
        BitPolicy.from_yaml(str(yaml_file))


# ---- BitPolicy._validate (via __init__) -------------------------------------


def test_init_validates_subtree_incompleteness():
    with pytest.raises(ValueError, match="Incomplete bit entry at 'mlp.gate_proj'"):
        BitPolicy({W_BITS: 16, A_BITS: 16, "mlp": {"gate_proj": {W_BITS: 8}}})


def test_init_validates_nested_subtree_incompleteness():
    with pytest.raises(ValueError, match="Incomplete bit entry at 'mlp.gate_proj'"):
        BitPolicy({"mlp": {"gate_proj": {W_BITS: 8, "up_proj": {W_BITS: 8}}}})


# ---- _check_complete --------------------------------------------------------


def test_check_complete_raises_on_mismatched_bits():
    with pytest.raises(ValueError, match="Incomplete bit entry at 'test.group'"):
        _check_complete({W_BITS: 4}, "test.group")


def test_check_complete_passes_on_complete_node():
    # Should not raise.
    _check_complete({W_BITS: 8, A_BITS: 8}, "test")


def test_check_complete_recurse_into_nested_dicts():
    with pytest.raises(ValueError, match="Incomplete bit entry at 'test.nested'"):
        _check_complete({"nested": {W_BITS: 4}}, "test")


def test_check_complete_ignores_non_bit_non_dict_values():
    # int values that are not bit keys should be skipped.
    _check_complete({W_BITS: 8, A_BITS: 8, "extra": 42, "nested": {W_BITS: 8, A_BITS: 8}}, "root")


# ---- _has_lt_16 ------------------------------------------------------------


def test_has_lt_16_true_for_low_bits():
    assert _has_lt_16({W_BITS: 4, A_BITS: 8}) is True


def test_has_lt_16_false_for_all_16():
    assert _has_lt_16({W_BITS: 16, A_BITS: 16}) is False


def test_has_lt_16_true_in_nested_dict():
    assert _has_lt_16({"proj": {W_BITS: 16, A_BITS: 4}}) is True


def test_has_lt_16_false_with_non_int_values():
    assert _has_lt_16({W_BITS: 16, A_BITS: "8"}) is False


# ---- BitPolicy.has_quant_linear / has_quant_cache --------------------------


def test_has_quant_linear_true_when_top_level_below_16():
    bp = BitPolicy({W_BITS: 8, A_BITS: 16})
    assert bp.has_quant_linear() is True


def test_has_quant_linear_true_from_nested_group():
    bp = BitPolicy({"mlp": {"gate_proj": {W_BITS: 8, A_BITS: 8}}, W_BITS: 16, A_BITS: 16})
    assert bp.has_quant_linear() is True


def test_has_quant_linear_false_when_all_bits_16():
    bp = BitPolicy({W_BITS: 16, A_BITS: 16, "mlp": {"gate_proj": {W_BITS: 16, A_BITS: 16}}})
    assert bp.has_quant_linear() is False


def test_has_quant_cache_true_when_key_below_16():
    bp = BitPolicy({"attn-cache": {"q": 8, "k": 16}})
    assert bp.has_quant_cache() is True


def test_has_quant_cache_false_when_all_keys_16():
    bp = BitPolicy({"attn-cache": {"q": 16, "k": 16}})
    assert bp.has_quant_cache() is False


def test_has_quant_cache_false_when_no_cache_group():
    bp = BitPolicy({W_BITS: 8, A_BITS: 8})
    assert bp.has_quant_cache() is False


# ---- BitPolicy.linear_bits -------------------------------------------------


def test_linear_bits_falls_back_to_top_level():
    bp = BitPolicy({W_BITS: 4, A_BITS: 8})
    assert bp.linear_bits(name="q_proj", group="attn-linear") == (4, 8)


def test_linear_bits_resolves_group_level_override():
    bp = BitPolicy({
        W_BITS: 16, A_BITS: 16,
        "attn-linear": {W_BITS: 8, A_BITS: 8},
    })
    assert bp.linear_bits(name="q_proj", group="attn-linear") == (8, 8)


def test_linear_bits_resolves_projection_level_override():
    bp = BitPolicy({
        W_BITS: 16, A_BITS: 16,
        "attn-linear": {
            W_BITS: 8, A_BITS: 8,
            "q_proj": {W_BITS: 4, A_BITS: 4},
        },
    })
    assert bp.linear_bits(name="q_proj", group="attn-linear") == (4, 4)


def test_linear_bits_dotted_group():
    bp = BitPolicy({
        W_BITS: 16, A_BITS: 16,
        "moe": {
            "routed": {W_BITS: 8, A_BITS: 8},
        },
    })
    assert bp.linear_bits(name="gate_proj", group="moe.routed") == (8, 8)


def test_linear_bits_no_name_no_group_returns_global():
    bp = BitPolicy({W_BITS: 4, A_BITS: 8})
    assert bp.linear_bits() == (4, 8)


# ---- BitPolicy.cache_bits --------------------------------------------------


def test_cache_bits_returns_default_16_for_missing_key():
    bp = BitPolicy()
    assert bp.cache_bits("q") == 16


def test_cache_bits_returns_configured_value():
    bp = BitPolicy({"attn-cache": {"q": 8, "k": 4}})
    assert bp.cache_bits("q") == 8
    assert bp.cache_bits("k") == 4


def test_cache_bits_returns_default_16_when_no_cache_group():
    bp = BitPolicy({W_BITS: 16, A_BITS: 16})
    assert bp.cache_bits("q") == 16


# ---- BitPolicy.summary ------------------------------------------------------


def test_summary_includes_bitpolicy_prefix():
    bp = BitPolicy({W_BITS: 8, A_BITS: 8})
    s = bp.summary()
    assert s.startswith("BitPolicy:")


# ---- _GroupBits -------------------------------------------------------------


def test_group_bits_getitem_returns_layer_bits():
    bp = BitPolicy({
        "attn-linear": {
            "q_proj": {W_BITS: 8, A_BITS: 4},
            "k_proj": {W_BITS: 8, A_BITS: 4},
        },
    })
    gb = _GroupBits(bp, "attn-linear")
    q = gb["q_proj"]
    assert isinstance(q, LayerBits)
    assert q.w == 8 and q.a == 4


def test_group_bits_default_returns_group_level_bits():
    bp = BitPolicy({
        "attn-linear": {W_BITS: 4, A_BITS: 8},
    })
    gb = _GroupBits(bp, "attn-linear")
    d = gb.default
    assert d.w == 4 and d.a == 8


def test_group_bits_default_falls_back_to_top_level():
    bp = BitPolicy({W_BITS: 8, A_BITS: 4})
    gb = _GroupBits(bp, "mlp")
    d = gb.default
    assert d.w == 8 and d.a == 4


def test_ensure_bit_policy_creates_from_args():
    args = SimpleNamespace(
        w_bits=8, a_bits=8, q_bits=4, k_bits=4, p_bits=4, v_bits=4,
        bit_policy=None,
    )
    policy = ensure_bit_policy(args)
    assert policy.w_bits == 8
    assert policy.a_bits == 8
    assert hasattr(args, "bit_policy")


def test_ensure_bit_policy_reuses_existing():
    existing = BitPolicy({"w_bits": 4, "a_bits": 4})
    args = SimpleNamespace(bit_policy=existing)
    policy = ensure_bit_policy(args)
    assert policy is existing


