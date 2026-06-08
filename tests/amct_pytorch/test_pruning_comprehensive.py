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
"""AMCT 结构化剪枝综合用例：HuggingFace 风格 mini 模型（MLP / CNN / MoE）。"""

import os
import shutil
import sys
import tempfile
import unittest

import torch

sys.path.insert(0, os.path.dirname(__file__))

from mini_models import create_mini_mlp, create_mini_cnn, create_mini_moe
from amct_pytorch.experimental.pruning import (
    DENSE_LOWVAR_PRUNE_CFG,
    CNN_VARIANCE_PRUNE_CFG,
    MOE_MASSVAR_PRUNE_CFG,
    prune,
    get_last_report,
)


def _calib_tokens():
    return [torch.randint(0, 1000, (4, 20)) for _ in range(8)]


class TestPruneRatios(unittest.TestCase):
    """剪枝率越大，参数削减越多。"""

    def test_increasing_ratio_increases_reduction(self):
        calib = _calib_tokens()
        reductions = []
        for ratio in (0.3, 0.5, 0.7):
            model, _ = create_mini_mlp()
            model.eval()
            cfg = {
                "methods": {
                    "dense": {"name": "low_variance", "kwargs": {"prune_ratio": ratio}}
                },
                "missing_data_policy": "warn_skip",
            }
            prune(model, cfg, data=calib)
            report = get_last_report()
            self.assertIsNotNone(report)
            self.assertLess(report.params_after, report.params_before)
            reductions.append(1 - report.params_after / report.params_before)
        # 削减比例随剪枝率单调递增
        self.assertLess(reductions[0], reductions[1])
        self.assertLess(reductions[1], reductions[2])


class TestSaveLoadRoundtrip(unittest.TestCase):
    """剪枝模型 state_dict 保存/加载后输出一致。"""

    def test_state_dict_roundtrip_matches(self):
        calib = _calib_tokens()
        original_model, config = create_mini_mlp()
        original_model.eval()
        prune(original_model, DENSE_LOWVAR_PRUNE_CFG, data=calib)
        report = get_last_report()
        self.assertLess(report.params_after, report.params_before)

        test_input = torch.randint(0, 1000, (2, 20))
        with torch.no_grad():
            original_output = original_model(test_input)

        temp_dir = tempfile.mkdtemp(prefix="amct_mlp_")
        try:
            save_path = os.path.join(temp_dir, "pruned_mlp.pth")
            # 只保存 state_dict（禁止 pickle 整个 nn.Module）
            torch.save(
                {
                    "model_state_dict": original_model.state_dict(),
                    "config": config.__dict__,
                    "pruning_info": {
                        "params_before": report.params_before,
                        "params_after": report.params_after,
                    },
                },
                save_path,
            )

            loaded_model, _ = create_mini_mlp()
            prune(loaded_model, DENSE_LOWVAR_PRUNE_CFG, data=calib)
            checkpoint = torch.load(save_path, weights_only=True)
            loaded_model.load_state_dict(checkpoint["model_state_dict"])
            loaded_model.eval()
            with torch.no_grad():
                loaded_output = loaded_model(test_input)
        finally:
            shutil.rmtree(temp_dir)

        output_diff = torch.abs(original_output - loaded_output).max().item()
        self.assertLess(output_diff, 1e-6)


class TestDomainPruning(unittest.TestCase):
    """各模型族结构化剪枝后参数下降且可正常前向。"""

    def test_mlp_dense_pruning(self):
        model, config = create_mini_mlp()
        model.eval()
        prune(model, DENSE_LOWVAR_PRUNE_CFG, data=_calib_tokens())
        report = get_last_report()
        self.assertLess(report.params_after, report.params_before)
        with torch.no_grad():
            out = model(torch.randint(0, 1000, (2, 20)))
        self.assertEqual(out.shape, (2, 20, config.vocab_size))

    def test_reconstruct_preserves_output_better_than_low_variance(self):
        """输出保留重构剪枝：同一剪枝率下，held-out 输出误差应小于 low_variance（切片）。"""
        import copy

        self.addCleanup(
            torch.set_rng_state, torch.get_rng_state()
        )  # 不污染后续测试的全局 RNG
        torch.manual_seed(0)
        model, _ = create_mini_mlp()
        model.eval()
        # 多样校准集（token 数远多于中间维），held-out 评估批与校准不相交
        calib = [torch.randint(0, 1000, (8, 32)) for _ in range(40)]
        eval_b = torch.randint(0, 1000, (4, 24))
        with torch.no_grad():
            ref = model(eval_b)

        def rel_err(name):
            m = copy.deepcopy(model)
            prune(
                m,
                {"methods": {"dense": {"name": name, "kwargs": {"prune_ratio": 0.4}}}},
                data=calib,
            )
            with torch.no_grad():
                out = m(eval_b)
            return ((out - ref).norm() / ref.norm()).item()

        e_lv = rel_err("low_variance")
        e_rec = rel_err("reconstruct")
        # 重构吸收了被移除神经元的贡献 → 输出更接近原模型
        self.assertLess(e_rec, e_lv)

    def test_conv1d_ffn_reconstruct_prunes_and_preserves(self):
        """GPT-2 风格 Conv1D FFN：reconstruct 应**实际剪枝**(参数下降)且重构保留输出。"""
        import torch.nn as nn
        import torch.nn.functional as F

        self.addCleanup(torch.set_rng_state, torch.get_rng_state())
        torch.manual_seed(0)

        class Conv1D(nn.Module):  # GPT-2 语义：weight [in, out]，nf=out
            def __init__(self, nf, nx):
                super().__init__()
                self.nf = nf
                self.weight = nn.Parameter(torch.randn(nx, nf) * 0.02)
                self.bias = nn.Parameter(torch.zeros(nf))

            def forward(self, x):
                return x @ self.weight + self.bias

        class Block(nn.Module):
            def __init__(self, h=32):
                super().__init__()
                self.c_fc = Conv1D(4 * h, h)
                self.c_proj = Conv1D(h, 4 * h)

            def forward(self, x):
                return x + self.c_proj(F.gelu(self.c_fc(x)))

        model = Block().eval()
        calib = [
            torch.randn(8, 16, 32) for _ in range(20)
        ]  # 多样校准, 2560 token >> 128
        x = torch.randn(4, 16, 32)
        with torch.no_grad():
            ref = model(x)
        p0 = sum(p.numel() for p in model.parameters())
        prune(
            model,
            {
                "methods": {
                    "dense": {"name": "reconstruct", "kwargs": {"prune_ratio": 0.4}}
                }
            },
            data=calib,
        )
        p1 = sum(p.numel() for p in model.parameters())
        self.assertLess(p1, p0)  # Conv1D 实际被剪枝（不再是 no-op）
        with torch.no_grad():
            out = model(x)
        self.assertLess(((out - ref).norm() / ref.norm()).item(), 0.25)  # 重构保留输出

    def test_quant_aware_saliency_valid_and_distinct(self):
        """量化感知显著度(GETA)：低位时用量化权重打分 → 选择不同的保留集，且模型仍有效。"""
        import copy

        self.addCleanup(torch.set_rng_state, torch.get_rng_state())
        torch.manual_seed(0)
        model, _ = create_mini_mlp()
        model.eval()
        calib = [torch.randint(0, 1000, (8, 32)) for _ in range(40)]
        probe = torch.randint(0, 1000, (2, 20))

        def run(**kw):
            m = copy.deepcopy(model)
            prune(
                m,
                {
                    "methods": {
                        "dense": {
                            "name": "reconstruct",
                            "kwargs": {"prune_ratio": 0.5, **kw},
                        }
                    }
                },
                data=calib,
            )
            with torch.no_grad():
                return m, m(probe)

        p_before = sum(p.numel() for p in model.parameters())
        m_blind, o_blind = run()
        m_aware, o_aware = run(quant_aware=True, quant_aware_bits=2)
        self.assertLess(
            sum(p.numel() for p in m_aware.parameters()), p_before
        )  # 仍真剪枝
        # 2-bit 量化扭曲显著度 → 保留集不同 → 输出与 quant-blind 不同（证明该选项真起作用）
        self.assertGreater((o_blind - o_aware).abs().max().item(), 1e-5)

    def test_cnn_channel_pruning(self):
        model, config = create_mini_cnn()
        model.eval()
        calib = [torch.randn(2, 3, 32, 32) for _ in range(8)]
        prune(model, CNN_VARIANCE_PRUNE_CFG, data=calib)
        report = get_last_report()
        self.assertLess(report.params_after, report.params_before)
        with torch.no_grad():
            out = model(torch.randn(2, 3, 32, 32))
        self.assertEqual(out.shape, (2, config.num_classes))

    def test_cnn_reconstruct_prunes_and_forwards(self):
        """自研 CNN reconstruct(无第三方依赖)：实际剪枝 + 前向可用 + 与 variance 同等结构稀疏。

        注：重构的*精度优势*只在**已训练**模型上有意义（随机权重无法泛化），故此处只做结构正确性断言；
        精度优势在 examples 的 MNIST 端到端基准里实测（trained: variance 0.343 vs reconstruct 0.574）。
        """
        import copy

        self.addCleanup(torch.set_rng_state, torch.get_rng_state())
        torch.manual_seed(0)
        model, config = create_mini_cnn()
        model.eval()
        calib = [torch.randn(4, 3, 32, 32) for _ in range(8)]
        p0 = sum(p.numel() for p in model.parameters())

        def pruned_params(name):
            m = copy.deepcopy(model)
            prune(
                m,
                {"methods": {"cnn": {"name": name, "kwargs": {"prune_ratio": 0.4}}}},
                data=calib,
            )
            with torch.no_grad():
                out = m(torch.randn(2, 3, 32, 32))
            self.assertEqual(out.shape, (2, config.num_classes))  # 前向可用
            return sum(p.numel() for p in m.parameters())

        n_rc = pruned_params("reconstruct")
        n_var = pruned_params("variance_channel")
        self.assertLess(n_rc, p0)  # 实际剪枝（非静默 no-op）
        self.assertEqual(n_rc, n_var)  # 与 variance 同等结构稀疏

    def test_moe_expert_pruning(self):
        model, config = create_mini_moe()
        model.eval()
        calib = [torch.randint(0, 1000, (4, 20)) for _ in range(8)]
        prune(model, MOE_MASSVAR_PRUNE_CFG, data=calib)
        report = get_last_report()
        self.assertLess(report.params_after, report.params_before)
        with torch.no_grad():
            out = model(torch.randint(0, 1000, (2, 20)))
        self.assertEqual(out.shape, (2, 20, config.vocab_size))


if __name__ == "__main__":
    unittest.main()
