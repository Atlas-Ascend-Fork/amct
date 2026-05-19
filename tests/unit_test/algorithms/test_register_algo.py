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

