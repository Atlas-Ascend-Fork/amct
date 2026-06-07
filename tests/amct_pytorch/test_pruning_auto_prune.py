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
"""容差驱动自动剪枝率搜索（accuracy_based_auto_prune）用例。

核心逻辑用手写 mini 模型做确定性断言（不依赖 transformers）；
另含一个真实 Qwen3 的端到端用例（有 transformers 时才跑）。
"""
import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.dirname(__file__))

from mini_models import create_mini_mlp
from amct_pytorch.experimental.pruning import (
    prune,
    accuracy_based_auto_prune,
    get_last_auto_prune,
)

try:
    import transformers  # noqa: F401
    _HAS_TF = True
except Exception:  # pragma: no cover
    _HAS_TF = False


def _calib(n=6):
    return [torch.randint(0, 1000, (4, 20)) for _ in range(n)]


DENSE_CFG = {
    "methods": {"dense": {"name": "low_variance"}},
    "missing_data_policy": "warn_skip",
}


class TestToleranceSelection(unittest.TestCase):
    """容差越大 → 选中的剪枝率（与权重削减）越大；过小容差不剪。"""

    def test_higher_tolerance_allows_more_pruning(self):
        """搜索逻辑：容差放宽 → 选中剪枝率与权重削减单调不降。

        用*参数量*作为质量（剪得越多质量越低，严格单调），使断言确定、
        不受随机初始化模型 top-1 噪声影响（二分查找以损失单调为前提）。
        """
        calib = _calib()
        chosen = {}
        reductions = {}

        def quality_by_params(p0):
            return lambda m: sum(p.numel() for p in m.parameters()) / p0

        for tol in (0.05, 0.15, 0.40):
            model, _ = create_mini_mlp()
            model.eval()
            p0 = sum(p.numel() for p in model.parameters())
            res = accuracy_based_auto_prune(model, DENSE_CFG, data=calib,
                                     tolerance=tol, quality_fn=quality_by_params(p0))
            chosen[tol] = res.chosen_ratio if res.chosen_ratio is not None else 0.0
            reductions[tol] = res.weight_reduction
            if res.chosen_ratio is not None:
                self.assertLessEqual(res.quality_drop, tol + 1e-9)
            with torch.no_grad():
                out = model(torch.randint(0, 1000, (2, 20)))
            self.assertTrue(torch.isfinite(out).all())
        # 单调不降：容差放宽不应让剪枝率/削减变小
        self.assertLessEqual(chosen[0.05], chosen[0.15])
        self.assertLessEqual(chosen[0.15], chosen[0.40])
        self.assertLessEqual(reductions[0.05], reductions[0.40])

    def test_zero_tolerance_is_conservative(self):
        """容差=0 时只接受零损失的剪枝率；通常不剪，原模型参数不变。"""
        calib = _calib()
        model, _ = create_mini_mlp()
        model.eval()
        p_before = sum(p.numel() for p in model.parameters())
        res = accuracy_based_auto_prune(model, DENSE_CFG, data=calib,
                                 tolerance=0.0, eval_data=_calib(4))
        p_after = sum(p.numel() for p in model.parameters())
        # 选中点（若有）损失必须为 0
        if res.chosen_ratio is not None:
            self.assertEqual(res.quality_drop, 0.0)
        else:
            self.assertFalse(res.applied)
            self.assertEqual(p_after, p_before)

    def test_apply_false_leaves_model_untouched(self):
        calib = _calib()
        model, _ = create_mini_mlp()
        model.eval()
        p_before = sum(p.numel() for p in model.parameters())
        res = accuracy_based_auto_prune(model, DENSE_CFG, data=calib, tolerance=0.5,
                                 eval_data=_calib(4), apply=False)
        # 仅搜索，不改原模型
        self.assertEqual(sum(p.numel() for p in model.parameters()), p_before)
        self.assertFalse(res.applied)
        # 但确实找到了一个候选率与预估削减
        self.assertIsNotNone(res.chosen_ratio)
        self.assertGreater(res.weight_reduction, 0.0)

    def test_custom_quality_fn(self):
        """自定义 quality_fn：用参数量作为“质量”，容差以参数为单位。

        参数越多“质量”越高 → 任何剪枝都降质；给一个较大的容差，
        应选中网格中最大的、参数下降不超过容差的剪枝率。
        """
        calib = _calib()
        model, _ = create_mini_mlp()
        model.eval()
        p0 = sum(p.numel() for p in model.parameters())

        def quality(m):
            # 归一化到 [0,1]，越接近原始参数量越“好”
            return sum(p.numel() for p in m.parameters()) / p0

        # 允许最多损失 0.2（即参数最多降到原来的 80%）
        res = accuracy_based_auto_prune(model, DENSE_CFG, data=calib,
                                 tolerance=0.2, quality_fn=quality)
        self.assertIsNotNone(res.chosen_ratio)
        self.assertLessEqual(res.quality_drop, 0.2 + 1e-9)
        # 等价地：权重削减 ≤ 容差
        self.assertLessEqual(res.weight_reduction, 0.2 + 1e-9)


class TestFacadeToleranceMode(unittest.TestCase):
    """主入口 prune(model, data=, tolerance=) 的容差模式 —— 最贴近“一键”用法。"""

    def test_facade_tolerance_applies_and_records_autotune(self):
        calib = _calib()
        model, _ = create_mini_mlp()
        model.eval()
        p_before = sum(p.numel() for p in model.parameters())

        # 用确定性的“参数量”质量（随机初始化模型的 top-1 保真度是噪声，不适合做断言）；
        # 只说“最多掉 30%”，验证容差模式：改了模型 + 记录 AutoTuneResult + 满足容差。
        def quality(m):
            return sum(p.numel() for p in m.parameters()) / p_before
        prune(model, data=calib, tolerance=0.30, quality_fn=quality)

        res = get_last_auto_prune()
        self.assertIsNotNone(res)
        self.assertTrue(res.applied)
        p_after = sum(p.numel() for p in model.parameters())
        self.assertLess(p_after, p_before)
        self.assertLessEqual(res.quality_drop, 0.30 + 1e-9)
        self.assertLessEqual(res.weight_reduction, 0.30 + 1e-9)
        with torch.no_grad():
            out = model(torch.randint(0, 1000, (2, 20)))
        self.assertTrue(torch.isfinite(out).all())

    def test_facade_fixed_ratio_mode_does_not_set_autotune(self):
        calib = _calib()
        model, _ = create_mini_mlp()
        model.eval()
        prune(model, DENSE_CFG, data=calib)  # 无 tolerance → 固定率模式
        self.assertIsNone(get_last_auto_prune())


@unittest.skipUnless(_HAS_TF and hasattr(transformers, "Qwen3ForCausalLM"),
                     "transformers/Qwen3 unavailable")
class TestToleranceRealModel(unittest.TestCase):
    """真实 Qwen3 上的端到端容差剪枝。"""

    def _model(self):
        torch.manual_seed(0)
        return transformers.Qwen3ForCausalLM(transformers.Qwen3Config(
            vocab_size=128, hidden_size=64, intermediate_size=256, num_hidden_layers=4,
            num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64,
        )).eval()

    def test_loose_vs_tight_tolerance(self):
        calib = [torch.randint(0, 128, (2, 16)) for _ in range(6)]
        eval_data = [torch.randint(0, 128, (4, 24)) for _ in range(4)]
        cfg = {"methods": {"dense": {"name": "low_variance"}},
               "missing_data_policy": "warn_skip"}

        tight = accuracy_based_auto_prune(self._model(), cfg, data=calib,
                                   tolerance=0.05, eval_data=eval_data)
        loose = accuracy_based_auto_prune(self._model(), cfg, data=calib,
                                   tolerance=0.20, eval_data=eval_data)

        # 宽容差应允许不小于紧容差的剪枝率与权重削减
        self.assertGreaterEqual(loose.chosen_ratio or 0.0, tight.chosen_ratio or 0.0)
        self.assertGreaterEqual(loose.weight_reduction, tight.weight_reduction)
        # 选中点都满足各自容差
        for res in (tight, loose):
            if res.chosen_ratio is not None:
                self.assertLessEqual(res.quality_drop, res.tolerance + 1e-9)

    @unittest.skipUnless(_HAS_TF and hasattr(transformers, "LlamaForCausalLM"),
                         "Llama unavailable")
    def test_llama_auto_skips_attention_no_crash(self):
        """不熟悉剪枝的用户对 Llama 直接用容差模式：safe_skip_attention 默认开启，
        注意力投影应被保护、不被误剪，最终模型可正常前向（不崩溃）。"""
        torch.manual_seed(0)
        model = transformers.LlamaForCausalLM(transformers.LlamaConfig(
            vocab_size=128, hidden_size=64, intermediate_size=256, num_hidden_layers=4,
            num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64,
        )).eval()
        q_before = model.model.layers[0].self_attn.q_proj.out_features
        calib = [torch.randint(0, 128, (2, 16)) for _ in range(6)]
        eval_data = [torch.randint(0, 128, (4, 24)) for _ in range(4)]

        # 一键：不传 config，只给容差
        res = accuracy_based_auto_prune(model, data=calib, tolerance=0.15, eval_data=eval_data)

        # 注意力投影维度不变（被 safe_skip_attention 保护）
        self.assertEqual(model.model.layers[0].self_attn.q_proj.out_features, q_before)
        # 模型可正常前向
        with torch.no_grad():
            out = model(eval_data[0])
        self.assertTrue(torch.isfinite(out.logits).all())
        # 选中点满足容差
        if res.chosen_ratio is not None:
            self.assertLessEqual(res.quality_drop, 0.15 + 1e-9)


if __name__ == "__main__":
    sys.exit(unittest.main(verbosity=2))
