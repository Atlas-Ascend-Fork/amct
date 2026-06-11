#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
"""按层敏感度分配剪枝（allocation）用例：water-fill 数学、守卫、默认零变化。"""

import copy
import os
import sys
import unittest
import warnings
from unittest import mock

import torch
import torch.nn.functional as F
from torch import nn

sys.path.insert(0, os.path.dirname(__file__))

from mini_models import create_mini_mlp
from amct_pytorch.experimental.pruning import (
    DENSE_LOWVAR_PRUNE_CFG,
    SENSITIVITY_ALLOC_PRUNE_CFG,
    prune,
    get_last_report,
)
from amct_pytorch.experimental.pruning.allocation import (
    layer_prefixes,
    measure_layer_sensitivity,
    water_fill_ratios,
)
from amct_pytorch.experimental.pruning.calib import calib_nll
from amct_pytorch.experimental.pruning.config import PruneConfig


def _calib_tokens(n=6):
    torch.manual_seed(0)
    return [torch.randint(0, 1000, (4, 20)) for _ in range(n)]


class _FFNBlock(nn.Module):
    """dense 域可识别的两层 FFN（dense1→gelu→dense2）。"""

    def __init__(self, hidden, inter):
        super().__init__()
        self.dense1 = nn.Linear(hidden, inter)
        self.dense2 = nn.Linear(inter, hidden)

    def forward(self, x):
        return self.dense2(F.gelu(self.dense1(x)))


class _HeadedMLP(nn.Module):
    """layers.{i} 块之外还有顶层 head FFN：验证逐层隔离不误剪层外目标。"""

    def __init__(self, n_layers=3, hidden=64, inter=128):
        super().__init__()
        self.embeddings = nn.Embedding(1000, hidden)
        self.layers = nn.ModuleList([_FFNBlock(hidden, inter) for _ in range(n_layers)])
        self.head_mlp = _FFNBlock(hidden, inter)
        self.classifier = nn.Linear(hidden, 1000)

    def forward(self, input_ids):
        h = self.embeddings(input_ids)
        for layer in self.layers:
            h = layer(h)
        return self.classifier(self.head_mlp(h))


class _PlainMLP(nn.Module):
    """无 layers/blocks/h 编号块的纯 MLP：keys<2 时必须真实 uniform 剪枝而非 no-op。"""

    def __init__(self, hidden=64, inter=128):
        super().__init__()
        self.embeddings = nn.Embedding(1000, hidden)
        self.ffn = _FFNBlock(hidden, inter)
        self.classifier = nn.Linear(hidden, 1000)

    def forward(self, input_ids):
        return self.classifier(self.ffn(self.embeddings(input_ids)))


class TestWaterFill(unittest.TestCase):
    """water-fill 分配的数学性质。"""

    def test_sum_preserved_and_ordering(self):
        sens = [0.1, 0.3, 0.5, 0.7]
        cut = 0.5
        ratios = water_fill_ratios(sens, cut)
        # 总削减量与 uniform 相同（未触发 ceil 截断时严格成立）
        self.assertAlmostEqual(sum(ratios), cut * len(sens), places=4)
        # robust（低敏感）层分到更大比例
        self.assertGreater(ratios[0], ratios[-1])
        for r in ratios:
            self.assertGreaterEqual(r, 0.05)
            self.assertLessEqual(r, 0.9)

    def test_floor_ceil_clipping(self):
        # 极端敏感差异：低敏感层吃满 → ceil 截断；高敏感层 → floor 托底
        ratios = water_fill_ratios([0.0, 0.0, 1.0, 1.0], 0.6, floor=0.1, ceil=0.7)
        for r in ratios:
            self.assertGreaterEqual(r, 0.1 * 0.6 / 1.0 - 1e-9)  # floor*scale 下界
            self.assertLessEqual(r, 0.7)
        self.assertLessEqual(max(ratios), 0.7)

    def test_dict_input_and_empty(self):
        ratios = water_fill_ratios({"a": 0.2, "b": 0.4}, 0.3)
        self.assertEqual(len(ratios), 2)
        with self.assertRaises(ValueError):
            water_fill_ratios([], 0.5)

    def test_budget_preserved_under_clipping(self):
        # 极端敏感差 + 高 cut：ceil 截断后残差必须摊回未夹紧层，总量严格保住
        ratios = water_fill_ratios([0.0, 1.0], 0.85)
        self.assertAlmostEqual(sum(ratios), 0.85 * 2, places=6)
        for r in ratios:
            self.assertGreaterEqual(r, 0.05 - 1e-9)
            self.assertLessEqual(r, 0.9 + 1e-9)
        # robust（低敏感）层仍分到更大比例
        self.assertGreater(ratios[0], ratios[1])

    def test_infeasible_budget_raises(self):
        # 预算超出 ceil*N 上界：不可分配，必须报错不静默
        with self.assertRaises(ValueError):
            water_fill_ratios([0.0, 1.0], 0.95, floor=0.05, ceil=0.9)


class TestSensitivityAllocation(unittest.TestCase):
    """敏感度分配 + calib_nll 守卫：不劣于 uniform；报告字段就位。"""

    def test_guard_never_worse_than_uniform(self):
        torch.manual_seed(0)
        calib = _calib_tokens()
        model, _ = create_mini_mlp()
        model.eval()
        baseline = copy.deepcopy(model)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            prune(model, SENSITIVITY_ALLOC_PRUNE_CFG, data=calib)
        report = get_last_report()
        self.assertIsNotNone(report)
        self.assertIn(report.allocation_choice, {"uniform", "sensitivity"})
        self.assertLess(report.params_after, report.params_before)

        # 同口径 uniform 候选（逐层 0.5）作对照；守卫保证选择不劣于 uniform
        from amct_pytorch.experimental.pruning.allocation import (
            _apply_layer_ratios,
            _base_config_dict,
        )

        cfg = PruneConfig(**{k: v for k, v in SENSITIVITY_ALLOC_PRUNE_CFG.items()})
        keys = layer_prefixes(baseline)
        uni = _apply_layer_ratios(
            copy.deepcopy(baseline), _base_config_dict(cfg), keys,
            [0.5] * len(keys), calib, None,
        )
        self.assertLessEqual(calib_nll(model, calib), calib_nll(uni, calib) + 1e-6)

    def test_measure_layer_sensitivity_shape(self):
        calib = _calib_tokens(4)
        model, _ = create_mini_mlp()
        model.eval()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sens = measure_layer_sensitivity(model, calib, ref_ratio=0.5)
        keys = layer_prefixes(model)
        self.assertEqual(sorted(sens), sorted(keys))
        for drop in sens.values():
            self.assertGreaterEqual(drop, 0.0)
            self.assertLessEqual(drop, 1.0)

    def test_requires_data(self):
        model, _ = create_mini_mlp()
        with self.assertRaises(ValueError):
            prune(model, SENSITIVITY_ALLOC_PRUNE_CFG, data=None)


class TestIsolationOutsideLayers(unittest.TestCase):
    """逐层隔离必须覆盖层前缀之外的全部可剪目标（head/pooler/顶层 FFN）。"""

    def test_head_mlp_untouched(self):
        torch.manual_seed(0)
        calib = _calib_tokens(4)
        model = _HeadedMLP().eval()
        inter = model.head_mlp.dense1.out_features
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            prune(model, SENSITIVITY_ALLOC_PRUNE_CFG, data=calib)
        # head FFN 不属于任何 layers.{i} 前缀 → 任何 pass 都不得动它
        self.assertEqual(model.head_mlp.dense1.out_features, inter)
        self.assertEqual(model.head_mlp.dense2.in_features, inter)
        # 编号层本体确实被剪
        self.assertLess(model.layers[0].dense1.out_features, inter)
        report = get_last_report()
        self.assertLess(report.params_after, report.params_before)


class TestNoLayerUniformFallback(unittest.TestCase):
    """keys<2 不允许 warn 后 no-op：必须真实跑 uniform 剪枝。"""

    def test_plain_mlp_actually_pruned(self):
        torch.manual_seed(0)
        calib = _calib_tokens(4)
        model = _PlainMLP().eval()
        inter = model.ffn.dense1.out_features
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            prune(model, SENSITIVITY_ALLOC_PRUNE_CFG, data=calib)
        self.assertTrue(any("uniform" in str(w.message) for w in caught))
        report = get_last_report()
        self.assertEqual(report.allocation_choice, "uniform")
        self.assertLess(report.params_after, report.params_before)
        self.assertLess(model.ffn.dense1.out_features, inter)


class TestGuardEdgeCases(unittest.TestCase):
    """守卫的 NaN 兜底与 uniform 回退分支必须真实可达。"""

    def test_nan_nll_warns_and_falls_back_uniform(self):
        report, messages = self._run_with_mock_nll([float("nan"), float("nan")])
        self.assertEqual(report.allocation_choice, "uniform")
        self.assertTrue(any("NaN" in m for m in messages))

    def test_sensitivity_worse_picks_uniform(self):
        # 敏感度候选 NLL 更差（mock）→ 必走 uniform 回退分支并记录选择
        report, messages = self._run_with_mock_nll([5.0, 1.0])
        self.assertEqual(report.allocation_choice, "uniform")
        self.assertTrue(any("reverting to uniform" in m for m in messages))
        self.assertLess(report.params_after, report.params_before)

    def _run_with_mock_nll(self, side_effect):
        torch.manual_seed(0)
        calib = _calib_tokens(4)
        model, _ = create_mini_mlp()
        model.eval()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with mock.patch(
                "amct_pytorch.experimental.pruning.allocation.calib_nll",
                side_effect=side_effect,
            ):
                prune(model, SENSITIVITY_ALLOC_PRUNE_CFG, data=calib)
        return get_last_report(), [str(w.message) for w in caught]


class TestCalibNLLInputGuard(unittest.TestCase):
    """calib_nll 只接受整型 token 批：浮点批必须报清晰 ValueError。"""

    def test_float_batch_raises(self):
        model, _ = create_mini_mlp()
        model.eval()
        with self.assertRaises(ValueError):
            calib_nll(model, [torch.randn(4, 20)])


class TestDefaultUnchanged(unittest.TestCase):
    """allocation=None（默认）= 现行 uniform 行为，零变化。"""

    def test_default_config_has_no_allocation(self):
        self.assertIsNone(PruneConfig().allocation)

    def test_default_prune_path_unchanged(self):
        calib = _calib_tokens(4)
        model, _ = create_mini_mlp()
        model.eval()
        prune(model, DENSE_LOWVAR_PRUNE_CFG, data=calib)
        report = get_last_report()
        self.assertIsNone(report.allocation_choice)
        self.assertLess(report.params_after, report.params_before)

    def test_allocation_validation(self):
        with self.assertRaises(ValueError):
            PruneConfig(allocation={"strategy": "bogus"}).validate()
        with self.assertRaises(ValueError):
            PruneConfig(
                allocation={"strategy": "sensitivity", "guard": "bogus"}
            ).validate()
        with self.assertRaises(ValueError):
            PruneConfig(
                allocation={"strategy": "sensitivity", "min_ratio": 0.8, "max_ratio": 0.2}
            ).validate()
        PruneConfig(allocation={"strategy": "sensitivity"}).validate()  # 合法不抛


if __name__ == "__main__":
    unittest.main()
