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
                'methods': {'dense': {'name': 'low_variance', 'kwargs': {'prune_ratio': ratio}}},
                'missing_data_policy': 'warn_skip',
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
            torch.save({
                'model_state_dict': original_model.state_dict(),
                'config': config.__dict__,
                'pruning_info': {
                    'params_before': report.params_before,
                    'params_after': report.params_after,
                },
            }, save_path)

            loaded_model, _ = create_mini_mlp()
            prune(loaded_model, DENSE_LOWVAR_PRUNE_CFG, data=calib)
            checkpoint = torch.load(save_path, weights_only=True)
            loaded_model.load_state_dict(checkpoint['model_state_dict'])
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
