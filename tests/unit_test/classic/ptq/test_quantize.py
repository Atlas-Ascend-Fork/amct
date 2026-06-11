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

import sys
from unittest.mock import patch

import pytest
from torch import nn

from amct_pytorch.classic.quantize import (
    algorithm_register,
    convert,
    quantize,
)
from amct_pytorch.quantize_op.base_quant_module import BaseQuantizeModule

QUANTIZE_MODULE = sys.modules["amct_pytorch.classic.quantize"]


def test_base_quantize_module_keeps_legacy_import_path():
    from amct_pytorch.classic.quantize_op.base_quant_module import BaseQuantizeModule as ClassicBaseQuantizeModule
    from amct_pytorch.quantize_op.base_quant_module import BaseQuantizeModule as LegacyBaseQuantizeModule

    assert LegacyBaseQuantizeModule is BaseQuantizeModule
    assert ClassicBaseQuantizeModule is BaseQuantizeModule


def test_classic_quantize_imports_without_ptq_package_layer():
    from amct_pytorch.classic.quantize import algorithm_register as classic_algorithm_register

    assert classic_algorithm_register is algorithm_register


class _DummyQuantOp(BaseQuantizeModule):
    pass


class _DummyDeployOp(nn.Module):
    pass


def test_algorithm_register_delegates_to_registry():
    with patch("amct_pytorch.algorithms.AlgorithmRegistry.register") as mock_reg:
        algorithm_register("alg_x", "Linear", _DummyQuantOp, _DummyDeployOp)
    mock_reg.assert_called_once_with("alg_x", "Linear", _DummyQuantOp, deploy_op=_DummyDeployOp)


def test_algorithm_register_allows_none_deploy_op():
    with patch("amct_pytorch.algorithms.AlgorithmRegistry.register") as mock_reg:
        algorithm_register("alg_y", "Linear", _DummyQuantOp)
    mock_reg.assert_called_once_with("alg_y", "Linear", _DummyQuantOp, deploy_op=None)


def test_algorithm_register_rejects_non_module_quant_op():
    class _NotAQuantOp:
        pass

    with pytest.raises(TypeError):
        algorithm_register("alg_z", "Linear", _NotAQuantOp, _DummyDeployOp)


def test_convert_runs_replace_pass_on_model():
    model = nn.Linear(4, 4)
    with patch.object(QUANTIZE_MODULE, "ModelOptimizer") as mock_opt_cls, \
         patch.object(QUANTIZE_MODULE, "ReplaceNpuQuantModulePass") as mock_pass_cls:
        opt = mock_opt_cls.return_value
        convert(model)

    mock_pass_cls.assert_called_once_with()
    opt.add_pass.assert_called_once_with(mock_pass_cls.return_value)
    opt.do_optimizer.assert_called_once_with(model)


def test_convert_rejects_non_module():
    with pytest.raises(TypeError):
        convert("not_a_module")


def test_quantize_uses_default_config_when_none():
    model = nn.Linear(4, 4)
    sentinel_default = {"_default": True}
    sentinel_layer = {"layer": "cfg"}
    with patch.object(QUANTIZE_MODULE, "set_default_config",
               return_value=sentinel_default) as mock_default, \
         patch.object(QUANTIZE_MODULE, "parse_config",
               return_value=sentinel_layer) as mock_parse, \
         patch.object(QUANTIZE_MODULE, "ModelOptimizer") as mock_opt_cls, \
         patch.object(QUANTIZE_MODULE, "InsertQuantizeModulePass") as mock_pass_cls:
        opt = mock_opt_cls.return_value
        quantize(model, None)

    mock_default.assert_called_once_with()
    mock_parse.assert_called_once()
    args, _ = mock_parse.call_args
    assert args[0] is model
    assert args[1] is sentinel_default
    mock_pass_cls.assert_called_once_with(sentinel_layer)
    opt.add_pass.assert_called_once_with(mock_pass_cls.return_value)
    opt.do_optimizer.assert_called_once_with(model)


def test_quantize_passes_user_config_through():
    model = nn.Linear(4, 4)
    user_cfg = {"granularity": "tensor"}
    with patch.object(QUANTIZE_MODULE, "set_default_config") as mock_default, \
         patch.object(QUANTIZE_MODULE, "parse_config",
               return_value={"layer": "cfg"}) as mock_parse, \
         patch.object(QUANTIZE_MODULE, "ModelOptimizer"), \
         patch.object(QUANTIZE_MODULE, "InsertQuantizeModulePass"):
        quantize(model, user_cfg)

    mock_default.assert_not_called()
    args, _ = mock_parse.call_args
    assert args[1] is user_cfg


def test_quantize_rejects_non_module_model():
    with pytest.raises(TypeError):
        quantize("not_a_module")


def test_quantize_rejects_non_dict_config():
    model = nn.Linear(4, 4)
    with pytest.raises(TypeError):
        quantize(model, "bad_config")
