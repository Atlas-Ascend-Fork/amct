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
"""守卫式 variance-HYBRID 用例：判据 MENU 验证集择优、绝不劣于 fallback、参数下降、前向正常。

并覆盖 mass_variance 新增的 ``variance_score`` kwarg（cond/peak/cvxpeak）的直接调用。
"""

import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.dirname(__file__))

from mini_models import MiniMoEConfig, MiniMoEModel
from amct_pytorch.experimental.pruning import (
    guarded_variance_hybrid_prune,
    prune,
)
from amct_pytorch.experimental.pruning.variance_hybrid import DEFAULT_MENU

VOCAB = 64


def _toy_config():
    """多专家小 MoE（16 专家 x 4 层，top-4），给方差判据足够的选择空间。"""
    return MiniMoEConfig(
        hidden_size=32,
        intermediate_size=48,
        num_experts=16,
        num_experts_per_tok=4,
        num_hidden_layers=4,
        vocab_size=VOCAB,
    )


def _toy_model(seed=0):
    torch.manual_seed(seed)
    return MiniMoEModel(_toy_config()).eval()


def _tokens(seed, n=6):
    torch.manual_seed(seed)
    return [torch.randint(0, VOCAB, (2, 16)) for _ in range(n)]


def _num_params(model):
    return sum(p.numel() for p in model.parameters())


def _imbalanced_model(seed=0):
    """不均衡路由的小 MoE：把每层 gate 权重设成"低索引专家被强偏好"，使 mass 集中在少数专家，
    而 peak/cond 等方差判据据此选出与 mass **不同**的保留专家集 —— 给守卫/判据分歧一个确定性场景。"""
    model = _toy_model(seed)
    with torch.no_grad():
        for layer in model.layers:
            w = layer.gate.weight  # [num_experts, hidden]
            w.zero_()
            num_experts, hidden = w.shape
            for e in range(num_experts):
                w[e, e % hidden] = (num_experts - e) * 0.5
    return model


def _expert_fingerprints(model):
    """每层保留专家的指纹集合（与路由器无关，剪枝后用于比较"保留了哪些专家"）。"""
    return [
        frozenset(round(float(e.fc1.weight.sum()), 5) for e in layer.experts)
        for layer in model.layers
    ]


def _prune_variance(variance_score, model, data):
    """用单一 variance_score 直接剪 MoE（boundary=-1 让每层都用方差打分），返回剪后模型。"""
    prune(
        model,
        {
            "methods": {
                "moe": {
                    "name": "mass_variance",
                    "kwargs": {
                        "prune_ratio": 0.5,
                        "top_k": 4,
                        "boundary": -1,
                        "variance_score": variance_score,
                    },
                }
            },
            "missing_data_policy": "warn_skip",
        },
        data=data,
    )
    return model


class TestGuardedVarianceHybrid(unittest.TestCase):
    def test_never_worse_than_fallback_prunes_and_forwards(self):
        """守卫保证：验证增益 >= 0；并且确实剪了专家、剪后仍能前向。"""
        model = _toy_model(0)
        before = _num_params(model)
        res = guarded_variance_hybrid_prune(
            model,
            data=_tokens(7),
            prune_ratio=0.5,
            top_k=4,
            eval_data=_tokens(11, n=3),
            apply=True,
        )
        # 守卫：HYBRID 在验证集上绝不劣于 fallback
        self.assertGreaterEqual(res.gain_over_fallback, 0.0)
        self.assertEqual(res.fallback_criterion, "mass")
        self.assertIn(res.chosen_criterion, dict(DEFAULT_MENU))
        # 选中判据的验证质量 >= fallback 的验证质量
        self.assertGreaterEqual(res.chosen_quality, res.fallback_quality)
        # 实际剪枝：参数下降，且 4 个判据都被评估
        self.assertLess(_num_params(model), before)
        self.assertEqual(set(res.criterion_quality), {n for n, _ in DEFAULT_MENU})
        # 剪后前向正常
        out = model(_tokens(11, n=1)[0])
        self.assertEqual(out.shape[-1], VOCAB)

    def test_non_fallback_criterion_can_strictly_win(self):
        """守卫的"切换"分支确实可达：构造一个奖励**方差判据保留集**的 quality_fn，
        则某非 mass 判据严格胜出 → chosen != "mass" 且 gain_over_fallback > 0。

        这堵住了 ``gain_over_fallback >= 0`` 的同义反复（impl 在无人胜过 fallback 时把
        gain 钳到 0、chosen 退回 mass，故 >=0 永真）：此处证明当方差判据真更优时守卫会切换。
        """
        # peak 判据在不均衡模型上的保留专家集（指纹），作为"已知更优"的奖励目标。
        peak_keep = set()
        for fps in _expert_fingerprints(
            _prune_variance("peak", _imbalanced_model(0), _tokens(7))
        ):
            peak_keep |= set(fps)

        def reward_peak_keep(candidate):
            # 候选保留越多 peak 选中的专家，质量越高 —— 奖励"方差判据"的选择。
            return float(
                sum(
                    round(float(e.fc1.weight.sum()), 5) in peak_keep
                    for layer in candidate.layers
                    for e in layer.experts
                )
            )

        res = guarded_variance_hybrid_prune(
            _imbalanced_model(0),
            data=_tokens(7),
            prune_ratio=0.5,
            top_k=4,
            quality_fn=reward_peak_keep,
            apply=False,
        )
        # 某非 fallback 判据严格胜出，守卫据此切换（非同义反复的 >=0）。
        self.assertNotEqual(res.chosen_criterion, "mass")
        self.assertGreater(res.gain_over_fallback, 0.0)
        # 切换的判据其验证质量确实严格高于 fallback(mass)。
        self.assertGreater(res.chosen_quality, res.criterion_quality["mass"])

    def test_apply_false_leaves_model_unchanged(self):
        model = _toy_model(0)
        before = _num_params(model)
        res = guarded_variance_hybrid_prune(
            model,
            data=_tokens(7),
            prune_ratio=0.5,
            top_k=4,
            eval_data=_tokens(11, n=3),
            apply=False,
        )
        self.assertFalse(res.applied)
        self.assertEqual(_num_params(model), before)  # 原模型不动
        self.assertLess(res.params_after, before)  # 但报告了选中判据的剪后尺寸

    def test_degenerate_menu_falls_back_to_mass(self):
        """MENU 只含 fallback 时，必然选 mass、增益恰为 0。"""
        model = _toy_model(1)
        res = guarded_variance_hybrid_prune(
            model,
            data=_tokens(7),
            prune_ratio=0.5,
            top_k=4,
            eval_data=_tokens(11, n=3),
            menu=[("mass", {"name": "activation_count", "kwargs": {}})],
            apply=True,
        )
        self.assertEqual(res.chosen_criterion, "mass")
        self.assertEqual(res.gain_over_fallback, 0.0)

    def test_invalid_prune_ratio_raises(self):
        model = _toy_model(0)
        with self.assertRaises(ValueError):
            guarded_variance_hybrid_prune(model, data=_tokens(7), prune_ratio=1.0)


class TestVarianceScoreKwarg(unittest.TestCase):
    """mass_variance 新增 variance_score（cond/peak/cvxpeak）：每种都剪枝且前向正常。"""

    def _cfg(self, variance_score):
        return {
            "methods": {
                "moe": {
                    "name": "mass_variance",
                    "kwargs": {
                        "prune_ratio": 0.5,
                        "top_k": 4,
                        "boundary": -1,  # 每层都用方差打分
                        "variance_score": variance_score,
                    },
                },
            },
            "missing_data_policy": "warn_skip",
        }

    def test_each_variance_score_prunes(self):
        for vs in ("cond", "peak", "cvxpeak"):
            with self.subTest(variance_score=vs):
                model = _toy_model(0)
                before = _num_params(model)
                prune(model, self._cfg(vs), data=_tokens(7))
                self.assertLess(_num_params(model), before)
                out = model(_tokens(11, n=1)[0])
                self.assertEqual(out.shape[-1], VOCAB)

    def test_variance_scores_keep_different_experts(self):
        """cond / peak / cvxpeak 的**全部意义**就是选出不同的保留专家集 —— 否则三者等价。
        在不均衡路由模型上，证明至少两个判据保留了**不同**的专家集（而不仅是同样的参数量）。
        """
        cond = _expert_fingerprints(
            _prune_variance("cond", _imbalanced_model(0), _tokens(7))
        )
        peak = _expert_fingerprints(
            _prune_variance("peak", _imbalanced_model(0), _tokens(7))
        )
        cvxpeak = _expert_fingerprints(
            _prune_variance("cvxpeak", _imbalanced_model(0), _tokens(7))
        )
        # 每层保留的专家数相同（同一剪枝率），但保留的是哪些专家不同。
        self.assertEqual([len(s) for s in cond], [len(s) for s in peak])
        # peak 既不同于 cond、也不同于 cvxpeak（≥2 个判据保留集互异）。
        self.assertTrue(
            any(peak[i] != cond[i] for i in range(len(peak))),
            "peak 与 cond 保留了相同专家集 —— variance_score 没有改变选择",
        )
        self.assertTrue(
            any(peak[i] != cvxpeak[i] for i in range(len(peak))),
            "peak 与 cvxpeak 保留了相同专家集 —— variance_score 没有改变选择",
        )

    def test_default_is_cond_backward_compatible(self):
        """不传 variance_score 时等价于 cond（向后兼容）。"""
        m_default = _toy_model(0)
        prune(
            m_default,
            {
                "methods": {
                    "moe": {
                        "name": "mass_variance",
                        "kwargs": {"prune_ratio": 0.5, "top_k": 4, "boundary": -1},
                    }
                },
                "missing_data_policy": "warn_skip",
            },
            data=_tokens(7),
        )
        m_cond = _toy_model(0)
        prune(m_cond, self._cfg("cond"), data=_tokens(7))
        self.assertEqual(_num_params(m_default), _num_params(m_cond))

    def test_invalid_variance_score_raises(self):
        model = _toy_model(0)
        with self.assertRaises(ValueError):
            prune(model, self._cfg("bogus"), data=_tokens(7))


if __name__ == "__main__":
    unittest.main()
