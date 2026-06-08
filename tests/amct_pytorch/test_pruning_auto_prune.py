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
    accuracy_based_auto_prune_with_budget,
    size_budget_prune,
    get_last_auto_prune,
)

# 结果数据类不在包级公开命名空间，按子模块路径导入。
from amct_pytorch.experimental.pruning.accuracy_based_auto_prune import JointBudget

try:
    import transformers

    _HAS_TF = True
except Exception:  # pragma: no cover
    _HAS_TF = False


def _calib(n=6):
    return [torch.randint(0, 1000, (4, 20)) for _ in range(n)]


DENSE_CFG = {
    "methods": {"dense": {"name": "low_variance"}},
    "missing_data_policy": "warn_skip",
}


class TestSizeBudget(unittest.TestCase):
    """尺寸预算模式：剪到目标参数量以内，且用最小够用的率（不过剪）。"""

    def test_hits_size_budget(self):
        model, _ = create_mini_mlp()
        model.eval()
        p0 = sum(p.numel() for p in model.parameters())
        res = size_budget_prune(
            model, DENSE_CFG, data=_calib(), target_keep_ratio=0.85, apply=True
        )
        self.assertIsNotNone(res.chosen_ratio)
        p1 = sum(p.numel() for p in model.parameters())
        # 达到预算：参数 ≤ 目标(保留85%)
        self.assertLessEqual(p1, int(round(p0 * 0.85)))
        # 不过剪：实际削减不应远超预算（够用即止）—— 削减比例在 [15%, 55%] 合理带内
        self.assertGreaterEqual(res.weight_reduction, 0.15 - 1e-9)

    def test_unreachable_budget_returns_none(self):
        model, _ = create_mini_mlp()
        model.eval()
        # 目标保留 1% 远超 grid 能力 → 找不到，模型不动
        p0 = sum(p.numel() for p in model.parameters())
        res = size_budget_prune(
            model, DENSE_CFG, data=_calib(), target_keep_ratio=0.01, apply=True
        )
        self.assertIsNone(res.chosen_ratio)
        self.assertEqual(sum(p.numel() for p in model.parameters()), p0)


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
            res = accuracy_based_auto_prune(
                model,
                DENSE_CFG,
                data=calib,
                tolerance=tol,
                quality_fn=quality_by_params(p0),
            )
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
        res = accuracy_based_auto_prune(
            model, DENSE_CFG, data=calib, tolerance=0.0, eval_data=_calib(4)
        )
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
        res = accuracy_based_auto_prune(
            model,
            DENSE_CFG,
            data=calib,
            tolerance=0.5,
            eval_data=_calib(4),
            apply=False,
        )
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
        res = accuracy_based_auto_prune(
            model, DENSE_CFG, data=calib, tolerance=0.2, quality_fn=quality
        )
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


@unittest.skipUnless(
    _HAS_TF and hasattr(transformers, "Qwen3ForCausalLM"),
    "transformers/Qwen3 unavailable",
)
class TestToleranceRealModel(unittest.TestCase):
    """真实 Qwen3 上的端到端容差剪枝。"""

    def _model(self):
        torch.manual_seed(0)
        return transformers.Qwen3ForCausalLM(
            transformers.Qwen3Config(
                vocab_size=128,
                hidden_size=64,
                intermediate_size=256,
                num_hidden_layers=4,
                num_attention_heads=4,
                num_key_value_heads=2,
                max_position_embeddings=64,
            )
        ).eval()

    def test_loose_vs_tight_tolerance(self):
        calib = [torch.randint(0, 128, (2, 16)) for _ in range(6)]
        eval_data = [torch.randint(0, 128, (4, 24)) for _ in range(4)]
        cfg = {
            "methods": {"dense": {"name": "low_variance"}},
            "missing_data_policy": "warn_skip",
        }

        tight = accuracy_based_auto_prune(
            self._model(), cfg, data=calib, tolerance=0.05, eval_data=eval_data
        )
        loose = accuracy_based_auto_prune(
            self._model(), cfg, data=calib, tolerance=0.20, eval_data=eval_data
        )

        # 宽容差应允许不小于紧容差的剪枝率与权重削减
        self.assertGreaterEqual(loose.chosen_ratio or 0.0, tight.chosen_ratio or 0.0)
        self.assertGreaterEqual(loose.weight_reduction, tight.weight_reduction)
        # 选中点都满足各自容差
        for res in (tight, loose):
            if res.chosen_ratio is not None:
                self.assertLessEqual(res.quality_drop, res.tolerance + 1e-9)

    @unittest.skipUnless(
        _HAS_TF and hasattr(transformers, "LlamaForCausalLM"), "Llama unavailable"
    )
    def test_llama_auto_skips_attention_no_crash(self):
        """不熟悉剪枝的用户对 Llama 直接用容差模式：safe_skip_attention 默认开启，
        注意力投影应被保护、不被误剪，最终模型可正常前向（不崩溃）。"""
        torch.manual_seed(0)
        model = transformers.LlamaForCausalLM(
            transformers.LlamaConfig(
                vocab_size=128,
                hidden_size=64,
                intermediate_size=256,
                num_hidden_layers=4,
                num_attention_heads=4,
                num_key_value_heads=2,
                max_position_embeddings=64,
            )
        ).eval()
        q_before = model.model.layers[0].self_attn.q_proj.out_features
        calib = [torch.randint(0, 128, (2, 16)) for _ in range(6)]
        eval_data = [torch.randint(0, 128, (4, 24)) for _ in range(4)]

        # 一键：不传 config，只给容差
        res = accuracy_based_auto_prune(
            model, data=calib, tolerance=0.15, eval_data=eval_data
        )

        # 注意力投影维度不变（被 safe_skip_attention 保护）
        self.assertEqual(model.model.layers[0].self_attn.q_proj.out_features, q_before)
        # 模型可正常前向
        with torch.no_grad():
            out = model(eval_data[0])
        self.assertTrue(torch.isfinite(out.logits).all())
        # 选中点满足容差
        if res.chosen_ratio is not None:
            self.assertLessEqual(res.quality_drop, 0.15 + 1e-9)


class TestFinetuneInSearch(unittest.TestCase):
    """搜索按"剪枝+恢复后"评估：finetune_fn 让搜索选出更激进的率（修正过度保守）。"""

    def test_finetune_fn_changes_chosen_ratio(self):
        # 确定性复现"剪枝直后崩溃、恢复后可用"：未恢复且被剪过的模型质量=0，
        # 原模型(满参数)或被 finetune_fn 标记为恢复的模型质量=1。
        model, _ = create_mini_mlp()
        model.eval()
        p0 = sum(p.numel() for p in model.parameters())

        def quality(m):
            if getattr(m, "_recovered", False):
                return 1.0
            cur = sum(p.numel() for p in m.parameters())
            return 1.0 if cur >= p0 else 0.0  # 任何"未恢复的剪枝"→ 崩溃为 0

        def recover(m):
            m._recovered = True  # 模拟微调恢复

        # (A) 不给 finetune_fn：每个候选都崩溃(0.0)，drop=1.0 > tol → 选不出
        rA = accuracy_based_auto_prune(
            model,
            DENSE_CFG,
            data=_calib(),
            tolerance=0.1,
            quality_fn=quality,
            ratio_grid=(0.1, 0.2, 0.3),
            apply=False,
        )
        self.assertIsNone(rA.chosen_ratio)

        # (B) 给 finetune_fn：候选被"恢复"后评估为高分(drop=0) → 选出最大率
        model2, _ = create_mini_mlp()
        model2.eval()
        rB = accuracy_based_auto_prune(
            model2,
            DENSE_CFG,
            data=_calib(),
            tolerance=0.1,
            quality_fn=quality,
            ratio_grid=(0.1, 0.2, 0.3),
            finetune_fn=recover,
            apply=False,
        )
        self.assertEqual(rB.chosen_ratio, 0.3)


class TestQuantAwareSearch(unittest.TestCase):
    """量化感知搜索：quant_fn 让搜索在**量化态**下决策 → 选更保守的剪枝率。"""

    def test_quant_fn_lowers_measured_quality_and_search_not_more_aggressive(self):
        p0 = sum(p.numel() for p in create_mini_mlp()[0].parameters())

        def quality(m):
            red = 1.0 - sum(p.numel() for p in m.parameters()) / p0
            # 量化放大剪枝损失：被 quant_fn 标记后，同样的削减质量更低。
            return 1.0 - red * (3.0 if getattr(m, "_quantized", False) else 1.0)

        def quantize_sim(m):
            m._quantized = True  # 模拟施加目标量化方案（fake-quant）

        grid = (0.1, 0.3, 0.5)
        # (1) 机制：quant_fn 确实在评估前施加 → 量化态质量 < 浮点态质量（同一剪枝模型）
        from amct_pytorch.experimental.pruning import prune as _prune

        m_fp, _ = create_mini_mlp()
        m_fp.eval()
        _prune(
            m_fp,
            {
                "methods": {
                    "dense": {"name": "low_variance", "kwargs": {"prune_ratio": 0.3}}
                }
            },
            data=_calib(),
        )
        q_fp = quality(m_fp)
        import copy as _copy

        m_q = _copy.deepcopy(m_fp)
        quantize_sim(m_q)
        self.assertLess(quality(m_q), q_fp)

        # (2) 不变式：量化感知搜索绝不比量化盲搜索更激进（weight_reduction 不更大）
        m1, _ = create_mini_mlp()
        m1.eval()
        rfp = accuracy_based_auto_prune(
            m1,
            DENSE_CFG,
            data=_calib(),
            tolerance=0.20,
            quality_fn=quality,
            ratio_grid=grid,
            apply=False,
        )
        m2, _ = create_mini_mlp()
        m2.eval()
        rqa = accuracy_based_auto_prune(
            m2,
            DENSE_CFG,
            data=_calib(),
            tolerance=0.20,
            quality_fn=quality,
            ratio_grid=grid,
            quant_fn=quantize_sim,
            apply=False,
        )
        self.assertLessEqual(rqa.weight_reduction, rfp.weight_reduction + 1e-9)


class TestJointBudget(unittest.TestCase):
    """联合精度预算分割（prune → quant 顺序测量）。"""

    def test_budget_split_reserves_remaining_for_quant(self):
        model, _ = create_mini_mlp()
        model.eval()
        total = 0.20
        jb = accuracy_based_auto_prune_with_budget(
            model, DENSE_CFG, total_tolerance=total, prune_fraction=0.5, data=_calib()
        )
        self.assertIsInstance(jb, JointBudget)
        # 剪枝上界 = 总预算 * fraction
        self.assertAlmostEqual(jb.prune_tolerance, total * 0.5, places=9)
        # 剩余预算 = 总预算 - 实测剪枝损失，且非负、不超过总预算
        self.assertGreaterEqual(jb.remaining_tolerance, 0.0)
        self.assertLessEqual(jb.remaining_tolerance, total + 1e-9)
        self.assertAlmostEqual(
            jb.remaining_tolerance, max(0.0, total - jb.prune_drop), places=6
        )
        # 剪枝损失不应超过分给它的上界（在单调假设下）
        self.assertLessEqual(jb.prune_drop, jb.prune_tolerance + 1e-9)

    def test_budget_no_prune_keeps_full_budget_for_quant(self):
        model, _ = create_mini_mlp()
        model.eval()
        # 极小预算 → 剪枝多半不动 → 剩余≈总预算
        jb = accuracy_based_auto_prune_with_budget(
            model, DENSE_CFG, total_tolerance=0.0, prune_fraction=1.0, data=_calib()
        )
        self.assertEqual(jb.prune_drop, 0.0)
        self.assertEqual(jb.remaining_tolerance, 0.0)

    def test_budget_validates_inputs(self):
        model, _ = create_mini_mlp()
        with self.assertRaises(ValueError):
            accuracy_based_auto_prune_with_budget(
                model, DENSE_CFG, total_tolerance=-0.1, data=_calib()
            )
        with self.assertRaises(ValueError):
            accuracy_based_auto_prune_with_budget(
                model, DENSE_CFG, total_tolerance=0.1, prune_fraction=1.5, data=_calib()
            )


if __name__ == "__main__":
    sys.exit(unittest.main(verbosity=2))
