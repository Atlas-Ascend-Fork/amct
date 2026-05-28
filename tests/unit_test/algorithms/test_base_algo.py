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

from amct_pytorch.algorithms.common.base_algo import BaseAlgo, BaseQuantAlgo

# ---- BaseAlgo (abstract) ------------------------------------------------


def test_cannot_instantiate_abstract_base_algo():
    with pytest.raises(TypeError):
        BaseAlgo()


class _DummyAlgo(BaseAlgo):
    def apply(self, model, *args, **kwargs):
        return model

    def get_config(self):
        return self.config


def test_concrete_subclass_records_config_and_name():
    algo = _DummyAlgo({"foo": 1})
    assert algo.config == {"foo": 1}
    assert algo.name == "_DummyAlgo"


def test_default_config_when_none_passed():
    algo = _DummyAlgo()
    assert algo.config == {}
    assert algo.get_config() == {}


def test_validate_config_default_is_no_op():
    algo = _DummyAlgo({"x": 1})
    # Should not raise and should return None.
    assert algo.validate_config() is None


# ---- BaseQuantAlgo ------------------------------------------------------


class _DummyQuantAlgo(BaseQuantAlgo):
    def apply(self, model, *args, **kwargs):
        return model

    def get_config(self):
        return self.config


def test_quant_algo_defaults_when_no_config():
    algo = _DummyQuantAlgo()
    assert algo.quant_dtype == "int"
    assert algo.weight_bits == 8
    assert algo.activation_bits == 8


def test_quant_algo_reads_overrides_from_config():
    algo = _DummyQuantAlgo(
        config={"quant_dtype": "mxfp", "weight_bits": 4, "activation_bits": 16}
    )
    assert algo.quant_dtype == "mxfp"
    assert algo.weight_bits == 4
    assert algo.activation_bits == 16


def test_quant_algo_falls_back_to_int_when_dtype_key_missing():
    algo = _DummyQuantAlgo(config={"weight_bits": 4})
    assert algo.quant_dtype == "int"
    assert algo.weight_bits == 4
    assert algo.activation_bits == 8


def test_base_algo_init_with_empty_config():
    class _Concrete(BaseAlgo):
        def apply(self, model, *args, **kwargs):
            pass

        def get_config(self):
            return super().get_config()
    algo = _Concrete({})
    assert algo.config == {}
    assert algo.name == "_Concrete"
    assert algo.get_config() == {}


def test_abstract_apply_body_executes_via_super_call():
    class _CallsSuper(BaseAlgo):
        def apply(self, model, *args, **kwargs):
            super().apply(model, *args, **kwargs)

        def get_config(self):
            return super().get_config()
    obj = _CallsSuper({})
    obj.apply(None)
