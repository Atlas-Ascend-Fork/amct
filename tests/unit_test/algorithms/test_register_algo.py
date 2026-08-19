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


import copy

import pytest

from amct_pytorch.algorithms.register_algo import Algorithm

ALG = 'alg'


class _Op:
    pass


class _OpA(_Op):
    pass


class _OpB(_Op):
    pass


class _OpC(_Op):
    pass


def test_initial_state_is_empty():
    a = Algorithm()
    assert not a.algo
    assert not a.quant_to_deploy
    assert not a.quant_op


def test_register_records_src_to_quant_mapping():
    a = Algorithm()
    a.register(ALG, "Linear", _OpA, _OpB)
    assert a.algo == {ALG: {"Linear": _OpA}}


def test_multiple_src_ops_share_algorithm_name():
    a = Algorithm()
    a.register(ALG, "Linear", _OpA, _OpB)
    a.register(ALG, "Conv2d", _OpC, _OpB)
    assert a.algo[ALG] == {"Linear": _OpA, "Conv2d": _OpC}


def test_register_scalar_deploy_op_appended_to_list():
    a = Algorithm()
    a.register(ALG, "Linear", _OpA, _OpB)
    assert a.quant_to_deploy[_OpA] == [_OpB]


def test_register_list_deploy_op_extends_existing_list():
    a = Algorithm()
    a.register(ALG, "Linear", _OpA, [_OpB, _OpC])
    assert set(a.quant_to_deploy[_OpA]) == {_OpB, _OpC}


def test_quant_to_deploy_is_deduplicated():
    a = Algorithm()
    a.register(ALG, "Linear", _OpA, _OpB)
    a.register(ALG, "Linear", _OpA, _OpB)
    assert a.quant_to_deploy[_OpA] == [_OpB]


def test_repeated_registration_can_extend_deploy_with_new_target():
    a = Algorithm()
    a.register(ALG, "Linear", _OpA, _OpB)
    a.register(ALG, "Linear", _OpA, _OpC)
    assert set(a.quant_to_deploy[_OpA]) == {_OpB, _OpC}


def test_name_index_maps_class_name_to_class():
    a = Algorithm()
    a.register(ALG, "Linear", _OpA, _OpB)
    assert a.name_to_quant_op == {"_OpA": _OpA}
    assert a.get_quant_op_by_name("_OpA") is _OpA


def test_name_index_skips_string_quant_op():
    a = Algorithm()
    a.register(ALG, None, "FP8Linear", _OpB)
    assert a.name_to_quant_op == {}


def test_name_index_allows_same_class_registered_twice():
    a = Algorithm()
    a.register(ALG, "Linear", _OpA, _OpB)
    a.register("other", "Conv2d", _OpA, _OpC)
    assert a.name_to_quant_op == {"_OpA": _OpA}


def test_name_index_rejects_two_classes_with_the_same_name():
    a = Algorithm()
    a.register(ALG, "Linear", _OpA, _OpB)
    clash = type("_OpA", (_Op,), {})
    with pytest.raises(ValueError, match="already registered"):
        a.register("other", "Linear", clash, _OpC)


def test_name_clash_leaves_no_half_registration():
    a = Algorithm()
    a.register(ALG, "Linear", _OpA, _OpB)
    algo_before = copy.deepcopy(a.algo)
    deploy_before = dict(a.quant_to_deploy)
    clash = type("_OpA", (_Op,), {})
    with pytest.raises(ValueError):
        a.register("other", "Linear", clash, _OpC)
    assert a.algo == algo_before
    assert a.quant_to_deploy == deploy_before
    assert clash not in a.quant_to_deploy
    assert a.name_to_quant_op["_OpA"] is _OpA


@pytest.mark.parametrize("bad", ["", None, 1, _OpA])
def test_get_quant_op_by_name_rejects_non_str(bad):
    a = Algorithm()
    with pytest.raises(ValueError):
        a.get_quant_op_by_name(bad)


def test_get_quant_op_by_name_rejects_unknown_name():
    a = Algorithm()
    a.register(ALG, "Linear", _OpA, _OpB)
    with pytest.raises(ValueError, match="not registered"):
        a.get_quant_op_by_name("_OpZ")


def test_production_registry_indexes_real_quant_modules():
    from amct_pytorch.algorithms import AlgorithmRegistry
    from amct_pytorch.classic.quantize_op.gptq_module import GPTQuant
    from amct_pytorch.classic.quantize_op.linear_awq_module import LinearAWQuant
    from amct_pytorch.classic.quantize_op.minmax_module import MinMaxQuant

    for cls in (MinMaxQuant, LinearAWQuant, GPTQuant):
        assert AlgorithmRegistry.get_quant_op_by_name(cls.__name__) is cls
    # Only classes are indexed. The string registrations (e.g. the 'FP8Linear'
    # placeholder) contribute no non-class values, whatever else a full session
    # has registered by now.
    assert all(
        isinstance(value, type) for value in AlgorithmRegistry.name_to_quant_op.values()
    )
    assert all(
        name == value.__name__
        for name, value in AlgorithmRegistry.name_to_quant_op.items()
    )
