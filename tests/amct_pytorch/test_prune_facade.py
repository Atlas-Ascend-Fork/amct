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
"""amct.prune façade + PrunePass + preset 配置 smoke 测试。"""

from __future__ import annotations

import unittest

from mock_torch_npu import *  # noqa: F401,F403

import torch
import torch.nn as nn

from amct_pytorch.pruning import (
    CNN_VARIANCE_PRUNE_CFG,
    DENSE_LOWVAR_PRUNE_CFG,
    FULL_STRUCTURED_PRUNE_CFG,
    MOE_MASSVAR_PRUNE_CFG,
    PruneConfig,
    PrunePass,
    PruneReport,
    get_last_report,
    prune,
)
from amct_pytorch.pruning.toy_models import TinyCNN, TinyMoE, TinyTransformerModel
from amct_pytorch.pruning.utils import count_parameters


class FacadeTest(unittest.TestCase):
    def test_dense_low_variance_with_dict_config(self) -> None:
        torch.manual_seed(0)
        model = TinyTransformerModel(hidden_size=32, intermediate_size=64, vocab_size=50)
        before = count_parameters(model)
        calib = [torch.randint(0, 50, (2, 8)) for _ in range(3)]
        ret = prune(model, DENSE_LOWVAR_PRUNE_CFG, data=calib)
        self.assertIsNone(ret)  # 与 amct.quantize 一致：façade 不返回值
        report = get_last_report()
        self.assertIsInstance(report, PruneReport)
        self.assertLess(report.params_after, before)

    def test_pruneconfig_object_also_accepted(self) -> None:
        torch.manual_seed(0)
        model = TinyTransformerModel(hidden_size=32, intermediate_size=64, vocab_size=50)
        cfg = PruneConfig(**DENSE_LOWVAR_PRUNE_CFG)
        calib = [torch.randint(0, 50, (2, 8)) for _ in range(3)]
        prune(model, cfg, data=calib)
        self.assertIsInstance(get_last_report(), PruneReport)

    def test_none_config_does_not_crash(self) -> None:
        torch.manual_seed(0)
        model = TinyTransformerModel(hidden_size=32, intermediate_size=64, vocab_size=50)
        prune(model, None)
        self.assertIsInstance(get_last_report(), PruneReport)

    def test_rejects_non_module(self) -> None:
        with self.assertRaises(TypeError):
            prune("not a module", DENSE_LOWVAR_PRUNE_CFG)

    def test_rejects_bad_config_type(self) -> None:
        with self.assertRaises(TypeError):
            prune(TinyTransformerModel(hidden_size=16, intermediate_size=32, vocab_size=20), 42)

    def test_cnn_preset(self) -> None:
        torch.manual_seed(0)
        model = TinyCNN(num_classes=10)
        calib = [torch.randn(2, 3, 16, 16) for _ in range(3)]
        before = count_parameters(model)
        prune(model, CNN_VARIANCE_PRUNE_CFG, data=calib)
        report = get_last_report()
        self.assertLess(report.params_after, before)
        # 验证模型对剪枝后形状仍可前向
        out = model(torch.randn(1, 3, 16, 16))
        self.assertEqual(out.shape[-1], 10)

    def test_moe_preset(self) -> None:
        torch.manual_seed(0)
        model = TinyMoE(num_experts=8, hidden_size=16)
        calib = [torch.randn(2, 4, 16) for _ in range(3)]
        before = count_parameters(model)
        prune(model, MOE_MASSVAR_PRUNE_CFG, data=calib)
        self.assertLess(get_last_report().params_after, before)


class PrunePassTest(unittest.TestCase):
    def test_run_modifies_model_inplace(self) -> None:
        torch.manual_seed(0)
        model = TinyTransformerModel(hidden_size=32, intermediate_size=64, vocab_size=50)
        before = count_parameters(model)
        calib = [torch.randint(0, 50, (2, 8)) for _ in range(3)]
        p = PrunePass(DENSE_LOWVAR_PRUNE_CFG, data=calib)
        p.run(model)
        self.assertIsNotNone(p.last_report)
        assert p.last_report is not None
        self.assertLess(p.last_report.params_after, before)
        # last_report.params_after 应当与 count_parameters(model) 一致
        self.assertEqual(p.last_report.params_after, count_parameters(model))

    def test_rejects_non_module(self) -> None:
        p = PrunePass(DENSE_LOWVAR_PRUNE_CFG)
        with self.assertRaises(TypeError):
            p.run("not a module")

    def test_composability_with_model_optimizer(self) -> None:
        from amct_pytorch.classic.ptq.optimizer.model_optimizer import ModelOptimizer  # 局部导入避免 NPU 副作用环境受影响

        torch.manual_seed(0)
        model = TinyTransformerModel(hidden_size=32, intermediate_size=64, vocab_size=50)
        calib = [torch.randint(0, 50, (2, 8)) for _ in range(3)]
        before = count_parameters(model)
        opt = ModelOptimizer()
        opt.add_pass(PrunePass(DENSE_LOWVAR_PRUNE_CFG, data=calib))
        opt.do_optimizer(model)
        self.assertLess(count_parameters(model), before)


class PresetShapeTest(unittest.TestCase):
    def test_preset_dicts_construct_valid_pruneconfig(self) -> None:
        for cfg_dict in (
            CNN_VARIANCE_PRUNE_CFG,
            DENSE_LOWVAR_PRUNE_CFG,
            MOE_MASSVAR_PRUNE_CFG,
            FULL_STRUCTURED_PRUNE_CFG,
        ):
            cfg = PruneConfig(**cfg_dict)
            cfg.validate()
            resolved = cfg.resolved_methods()
            self.assertGreater(len(resolved), 0)


if __name__ == "__main__":
    unittest.main()
