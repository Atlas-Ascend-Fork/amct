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
"""补齐“有实现但此前无测试”的公开表面：默认配置路径、activation_count(MoE)、
FULL_STRUCTURED 预设、PrunePass、per_layer_sparsity、runtime 工具，以及与量化
共用精度抽象的 evaluator 路径。

之前审计发现这些代码均**可运行但无测试覆盖**。与之相对，真正“只有形态、无功能”
的项已被清理：孤立的 toy_models.py（已删除）、未被任何逻辑读取的
PruneConfig.save_dtype / backend_overrides（已从配置移除）。
"""

import os
import sys
import tempfile
import unittest

import torch

sys.path.insert(0, os.path.dirname(__file__))

from mini_models import create_mini_mlp, create_mini_moe
import torch.nn as nn

from amct_pytorch.experimental.pruning import (
    prune,
    accuracy_based_auto_prune,
    get_last_report,
    PrunePass,
    FULL_STRUCTURED_PRUNE_CFG,
    diagnose,
    finetune,
)

# 内部工具/结果类不在包级公开命名空间，按子模块路径导入。
from amct_pytorch.experimental.pruning.runtime import (
    collect_runtime_environment,
    save_runtime_environment,
    save_tokenizer,
)
from amct_pytorch.experimental.pruning.diagnostics import DiagnosisReport
import torch.nn.functional as F


def _tok(n=6):
    return [torch.randint(0, 1000, (4, 20)) for _ in range(n)]


def _params(m):
    return sum(p.numel() for p in m.parameters())


class TestFinetune(unittest.TestCase):
    """P4：剪枝后可选微调（恢复精度）。在一个可学习的合成分类任务上验证 loss 下降。"""

    def _toy_task(self, n=256, d=16, c=4):
        torch.manual_seed(0)
        W = torch.randn(d, c)
        X = torch.randn(n, d)
        Y = (X @ W).argmax(-1)  # 确定性可学习标签
        return [(X[i : i + 32], Y[i : i + 32]) for i in range(0, n, 32)], d, c

    def test_finetune_reduces_loss(self):
        data, d, c = self._toy_task()
        model = nn.Sequential(nn.Linear(d, 32), nn.ReLU(), nn.Linear(32, c))
        hist = finetune(
            model,
            data,
            loss_fn=lambda m, b: F.cross_entropy(m(b[0]), b[1]),
            steps=200,
            lr=1e-2,
        )
        self.assertEqual(hist["steps"], 200)
        # 训练应显著降低损失
        self.assertLess(hist["final_loss"], hist["initial_loss"])
        self.assertLess(hist["final_loss"], 0.5)

    def test_finetune_default_loss_xy(self):
        """默认损失能从 (x, y) 推断交叉熵，无需显式 loss_fn。"""
        data, d, c = self._toy_task()
        model = nn.Sequential(nn.Linear(d, c))
        hist = finetune(model, data, steps=50, lr=1e-2)
        self.assertEqual(hist["steps"], 50)
        self.assertIsNotNone(hist["final_loss"])

    def test_finetune_zero_steps_noop(self):
        data, d, c = self._toy_task()
        model = nn.Sequential(nn.Linear(d, c))
        before = [p.clone() for p in model.parameters()]
        hist = finetune(model, data, steps=0)
        self.assertEqual(hist["steps"], 0)
        for a, b in zip(before, model.parameters()):
            self.assertTrue(torch.equal(a, b))  # 未改动


class TestDefaultAndMoEMethods(unittest.TestCase):
    """默认配置路径 + MoE 两种方法（activation_count / mass_variance）。"""

    def test_default_config_prunes(self):
        """prune(model) 不传 config：走 PruneConfig() 默认三域方法
        （moe 默认 activation_count），应正常剪枝。"""
        model, config = create_mini_moe()
        model.eval()
        p_before = _params(model)
        prune(model, data=_tok())  # 默认配置
        self.assertLess(_params(model), p_before)
        with torch.no_grad():
            out = model(torch.randint(0, 1000, (2, 20)))
        self.assertEqual(out.shape, (2, 20, config.vocab_size))

    def test_moe_activation_count_method(self):
        """MoE activation_count（MassMoEPruningMethod，默认方法）显式调用。"""
        model, config = create_mini_moe()
        model.eval()
        cfg = {
            "methods": {
                "moe": {"name": "activation_count", "kwargs": {"prune_ratio": 0.5}}
            },
            "missing_data_policy": "warn_skip",
        }
        prune(model, cfg, data=_tok())
        report = get_last_report()
        self.assertLess(report.params_after, report.params_before)
        with torch.no_grad():
            out = model(torch.randint(0, 1000, (2, 20)))
        self.assertEqual(out.shape, (2, 20, config.vocab_size))

    def test_full_structured_preset(self):
        """FULL_STRUCTURED_PRUNE_CFG（三域齐发）在 MoE 模型上生效。"""
        model, _ = create_mini_moe()
        model.eval()
        p_before = _params(model)
        prune(model, FULL_STRUCTURED_PRUNE_CFG, data=_tok())
        self.assertLess(_params(model), p_before)


class TestReportAndPass(unittest.TestCase):
    def test_per_layer_sparsity_recorded(self):
        model, _ = create_mini_mlp()
        model.eval()
        prune(
            model,
            {
                "methods": {
                    "dense": {"name": "low_variance", "kwargs": {"prune_ratio": 0.5}}
                },
                "missing_data_policy": "warn_skip",
            },
            data=_tok(),
        )
        report = get_last_report()
        self.assertGreater(len(report.per_layer_sparsity), 0)
        for sparsity in report.per_layer_sparsity.values():
            self.assertGreater(sparsity, 0.0)
        # as_dict 可序列化包含该字段
        self.assertIn("per_layer_sparsity", report.as_dict())

    def test_prune_pass_run(self):
        """PrunePass（ModelOptimizer 风格，与量化 pass 串接的入口）可独立运行。"""
        model, _ = create_mini_mlp()
        model.eval()
        p_before = _params(model)
        pass_ = PrunePass(
            {
                "methods": {
                    "dense": {"name": "low_variance", "kwargs": {"prune_ratio": 0.5}}
                },
                "missing_data_policy": "warn_skip",
            },
            data=_tok(),
        )
        pass_.run(model)
        self.assertLess(_params(model), p_before)
        self.assertIsNotNone(pass_.last_report)

    def test_prune_pass_rejects_non_module(self):
        pass_ = PrunePass()
        with self.assertRaises(TypeError):
            pass_.run("not a module")


class TestRuntimeUtils(unittest.TestCase):
    def test_collect_and_save_runtime_environment(self):
        env = collect_runtime_environment()
        self.assertIn("python_version", env)
        self.assertIn("packages", env)
        tmp = tempfile.mkdtemp(prefix="amct_env_")
        try:
            path = save_runtime_environment(os.path.join(tmp, "env.json"), env)
            self.assertTrue(os.path.exists(path))
        finally:
            import shutil

            shutil.rmtree(tmp)

    def test_save_tokenizer_none_is_graceful(self):
        ok, err = save_tokenizer(None, tempfile.mkdtemp(prefix="amct_tok_"))
        self.assertFalse(ok)
        self.assertIsInstance(err, str)


class _FakeEvaluator:
    """模拟量化侧 ModelEvaluator：evaluate(model, iterations) -> 精度(此处用参数比例)。"""

    def __init__(self, p0):
        self.p0 = p0
        self.calls = 0

    def evaluate(self, model, iterations):
        self.calls += 1
        return sum(p.numel() for p in model.parameters()) / self.p0


class TestEvaluatorUnification(unittest.TestCase):
    """accuracy_based_auto_prune 接受与量化共用的 evaluator（duck-typed .evaluate）。"""

    def test_evaluator_drives_search(self):
        model, _ = create_mini_mlp()
        model.eval()
        p0 = _params(model)
        ev = _FakeEvaluator(p0)
        res = accuracy_based_auto_prune(
            model,
            {
                "methods": {"dense": {"name": "low_variance"}},
                "missing_data_policy": "warn_skip",
            },
            data=_tok(),
            tolerance=0.2,
            evaluator=ev,
        )
        # evaluator 被实际调用（基线 + 各试探）
        self.assertGreater(ev.calls, 1)
        self.assertIsNotNone(res.chosen_ratio)
        # 选中点满足容差（参数比例度量下，等价于权重削减 <= 容差）
        self.assertLessEqual(res.quality_drop, 0.2 + 1e-9)
        self.assertLessEqual(res.weight_reduction, 0.2 + 1e-9)

    def test_bad_evaluator_raises(self):
        model, _ = create_mini_mlp()
        model.eval()
        with self.assertRaises(TypeError):
            accuracy_based_auto_prune(
                model, data=_tok(), tolerance=0.2, evaluator=object()
            )  # 无 .evaluate

    def test_single_arg_evaluator_works(self):
        """AutoCalibrationEvaluatorBase 风格 evaluate(model) 单参也能驱动搜索。"""
        model, _ = create_mini_mlp()
        model.eval()
        p0 = _params(model)

        class _OneArgEvaluator:  # 与量化 AutoCalibrationEvaluatorBase.evaluate(self, model) 同形
            def __init__(self):
                self.calls = 0

            def evaluate(self, model):
                self.calls += 1
                return sum(p.numel() for p in model.parameters()) / p0

        ev = _OneArgEvaluator()
        res = accuracy_based_auto_prune(
            model,
            {
                "methods": {"dense": {"name": "low_variance"}},
                "missing_data_policy": "warn_skip",
            },
            data=_tok(),
            tolerance=0.2,
            evaluator=ev,
        )
        self.assertGreater(ev.calls, 1)
        self.assertIsNotNone(res.chosen_ratio)

    def test_none_returning_evaluator_raises(self):
        """graph 版 ModelEvaluator（evaluate 返回 None 的数据馈送器）给出明确报错。"""
        model, _ = create_mini_mlp()
        model.eval()

        class _FeederEvaluator:  # 模拟 graph ModelEvaluator：返回 None
            def evaluate(self, model, iterations):
                return None

        with self.assertRaises(TypeError):
            accuracy_based_auto_prune(
                model, data=_tok(), tolerance=0.2, evaluator=_FeederEvaluator()
            )


class TestDiagnose(unittest.TestCase):
    """剪枝功能自检 diagnose()。"""

    def test_diagnose_functional_model(self):
        model, _ = create_mini_moe()
        model.eval()
        p_before = _params(model)
        rep = diagnose(model, data=_tok())
        self.assertIsInstance(rep, DiagnosisReport)
        # 在 toy MoE 上应识别到 dense + moe 目标
        self.assertTrue(rep.any_domain_detected)
        self.assertGreater(rep.targets["dense"], 0)
        self.assertGreater(rep.targets["moe"], 0)
        # 固定率剪枝真正生效且能前向
        self.assertTrue(rep.prune_works)
        self.assertGreater(rep.prune_reduction, 0.0)
        self.assertTrue(rep.prune_forward_ok)
        # search 检查执行完毕（bool；随机权重下是否选出不强求）
        self.assertIsInstance(rep.search_works, bool)
        self.assertIsInstance(rep.summary(), str)
        # dry-run 不改原模型
        self.assertEqual(_params(model), p_before)

    def test_diagnose_unsupported_model(self):
        """无可剪枝结构（仅 Embedding）→ 0 目标、剪枝无效，被如实报告。"""
        model = nn.Sequential(nn.Embedding(50, 8))
        rep = diagnose(model, data=None)
        self.assertFalse(rep.any_domain_detected)
        self.assertFalse(rep.prune_works)
        self.assertTrue(any("未在任何域" in n for n in rep.notes))


if __name__ == "__main__":
    sys.exit(unittest.main(verbosity=2))
