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
"""MoE 输出空间专家合并（output_merge）用例：深层 toy MoE 上选择器择优、参数下降。"""

import copy
import os
import sys
import unittest
import warnings
from unittest import mock

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(__file__))

from mini_models import MiniMoEConfig, MiniMoEModel
from amct_pytorch.experimental.pruning import (
    MOE_OUTPUT_MERGE_PRUNE_CFG,
    prune,
    get_last_report,
)
from amct_pytorch.experimental.pruning.methods import moe_output_merge
from amct_pytorch.experimental.pruning.methods.moe_output_merge import (
    _batch_token_ids,
    _fit_gate_row,
)

VOCAB = 64


def _toy_config():
    """深层小 MoE（6 专家 x 6 层），保证用例快速且合并空间充分。"""
    return MiniMoEConfig(
        hidden_size=32,
        intermediate_size=48,
        num_experts=6,
        num_experts_per_tok=2,
        num_hidden_layers=6,
        vocab_size=VOCAB,
    )


def _toy_model(seed):
    torch.manual_seed(seed)
    return MiniMoEModel(_toy_config()).eval()


def _calib_tokens():
    torch.manual_seed(123)
    return [torch.randint(0, VOCAB, (2, 16)) for _ in range(6)]


def _num_params(model):
    return sum(p.numel() for p in model.parameters())


@torch.no_grad()
def _calib_nll(model, calib):
    # 校准集 next-token NLL（与方法内选择器同口径）
    total = 0.0
    for batch in calib:
        logits = model(batch)
        total += float(
            nn.functional.cross_entropy(
                logits[:, :-1].reshape(-1, logits.shape[-1]), batch[:, 1:].reshape(-1)
            )
        )
    return total / len(calib)


class _FusedExperts(nn.Module):
    """现代融合专家张量结构（专家为第 0 维批参数）。"""

    def __init__(self, num_experts=8, hidden=16):
        super().__init__()
        self.num_experts = num_experts
        self.weight = nn.Parameter(torch.randn(num_experts, hidden, hidden))

    def forward(self, x):
        return x


class _FusedMoEModel(nn.Module):
    def __init__(self, num_experts=8, hidden=16):
        super().__init__()
        self.embeddings = nn.Embedding(VOCAB, hidden)
        self.gate = nn.Linear(hidden, num_experts, bias=False)
        self.experts = _FusedExperts(num_experts, hidden)
        self.classifier = nn.Linear(hidden, VOCAB)

    def forward(self, input_ids):
        return self.classifier(self.experts(self.embeddings(input_ids)))


class TestOutputMergeSelector(unittest.TestCase):
    """selector=calib_nll：参数下降，且校准 NLL 不劣于 activation_count 基线。"""

    def test_selector_reduces_params_and_never_worse(self):
        calib = _calib_tokens()
        merged = _toy_model(0)
        prune(merged, MOE_OUTPUT_MERGE_PRUNE_CFG, data=calib)
        report = get_last_report()
        self.assertLess(report.params_after, report.params_before)
        # 选择结果必须显式记录（绝不静默）
        events = [e.detail for e in report.events if "calib_nll selector" in e.detail]
        self.assertEqual(len(events), 1)
        self.assertIn("selected", events[0])

        act = _toy_model(0)
        act_cfg = {
            "methods": {
                "moe": {"name": "activation_count", "kwargs": {"prune_ratio": 0.5}}
            },
            "missing_data_policy": "warn_skip",
        }
        prune(act, act_cfg, data=calib)
        # 专家压缩幅度一致（FFN 通道剪枝细节允许少量差异，不比较总参数）
        for layer_m, layer_a in zip(merged.layers, act.layers):
            self.assertEqual(len(layer_m.experts), len(layer_a.experts))
            self.assertEqual(layer_m.gate.out_features, layer_a.gate.out_features)
        # 选择器逐字面取低者 → 不会劣于 activation_count
        self.assertLessEqual(_calib_nll(merged, calib), _calib_nll(act, calib) + 1e-6)


class TestOutputMergeOnly(unittest.TestCase):
    """selector=none：纯合并路径剪枝后可前向、专家数减半。"""

    def test_merge_only_prunes_and_forwards(self):
        model = _toy_model(1)
        cfg = {
            "methods": {
                "moe": {
                    "name": "output_merge",
                    "kwargs": {"keep_ratio": 0.5, "selector": "none"},
                }
            },
            "missing_data_policy": "warn_skip",
        }
        prune(model, cfg, data=_calib_tokens())
        report = get_last_report()
        self.assertLess(report.params_after, report.params_before)
        for layer in model.layers:
            self.assertEqual(len(layer.experts), 3)
            self.assertEqual(layer.gate.out_features, 3)
        details = [e.detail for e in report.events]
        self.assertTrue(any("Output-merge experts 6 -> 3" in d for d in details))
        with torch.no_grad():
            logits = model(torch.randint(0, VOCAB, (2, 8)))
        self.assertEqual(logits.shape, (2, 8, VOCAB))
        self.assertTrue(torch.isfinite(logits).all())


class TestOutputMergeFusedSkip(unittest.TestCase):
    """融合专家结构：响亮警告并跳过，模型不变。"""

    def test_fused_target_loud_skip(self):
        torch.manual_seed(2)
        model = _FusedMoEModel().eval()
        before = _num_params(model)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            prune(model, MOE_OUTPUT_MERGE_PRUNE_CFG, data=_calib_tokens())
        self.assertEqual(_num_params(model), before)
        messages = [str(w.message) for w in caught]
        self.assertTrue(any("skipping fused" in m for m in messages))
        report = get_last_report()
        self.assertTrue(any("skipping fused" in w for w in report.warnings))


class TestRouterMergeMath(unittest.TestCase):
    """路由合并数学：校准 LS 拟合的概率误差须远低于逐元素 logaddexp 基线。"""

    def test_ls_fit_beats_logaddexp(self):
        torch.manual_seed(7)
        hidden, e_n, grp = 32, 6, [0, 1]
        x = torch.randn(512, hidden)
        for use_bias in (False, True):
            gate = nn.Linear(hidden, e_n, bias=use_bias)
            probs = torch.softmax(gate(x).float(), -1)
            target = probs[:, grp].sum(1)

            gate_la = copy.deepcopy(gate)
            gate_la.weight.data[0] = torch.logaddexp(
                gate.weight.data[0], gate.weight.data[1]
            )
            la_probs, _ = self._probs_after(gate_la, x, grp)
            err_la = float((la_probs[:, 0] - target).abs().mean())

            gate_ls = copy.deepcopy(gate)
            row, bias_val = _fit_gate_row(gate, x, grp, probs)
            gate_ls.weight.data[0] = row
            if use_bias:
                self.assertIsNotNone(bias_val)
                gate_ls.bias.data[0] = bias_val
            else:
                self.assertIsNone(bias_val)
            ls_probs, _ = self._probs_after(gate_ls, x, grp)
            err_ls = float((ls_probs[:, 0] - target).abs().mean())

            # LS 拟合须明显低于 logaddexp 基线；带偏置时近乎精确
            self.assertLess(err_ls, 0.5 * err_la)
            if use_bias:
                self.assertLess(err_ls, 0.1 * err_la)
                self.assertLess(err_ls, 0.03)

    def _probs_after(self, gate, x, grp):
        # 删除被并列、保留吸收行后的剩余 softmax
        keep = [i for i in range(gate.out_features) if i not in grp[1:]]
        return torch.softmax(gate(x).float()[:, keep], -1), keep


class TestSelectorWhitelist(unittest.TestCase):
    """selector 白名单：仅 calib_nll / none，其余显式 ValueError。"""

    def test_invalid_selector_raises(self):
        model = _toy_model(4)
        cfg = {
            "methods": {
                "moe": {"name": "output_merge", "kwargs": {"selector": "bogus"}}
            },
            "missing_data_policy": "warn_skip",
        }
        with self.assertRaises(ValueError):
            prune(model, cfg, data=_calib_tokens())


class TestTokenIdExtraction(unittest.TestCase):
    """token 标签提取：优先 kwargs['input_ids']，绝不取 attention_mask。"""

    def test_prefers_input_ids_never_attention_mask(self):
        ids = torch.randint(0, VOCAB, (2, 16))
        mask = torch.ones(2, 16, dtype=torch.long)
        # attention_mask 在 input_ids 之前出现也必须取 input_ids
        got = _batch_token_ids((), {"attention_mask": mask, "input_ids": ids})
        self.assertIs(got, ids)
        # 仅有 attention_mask（2D 整型）时绝不误用
        self.assertIsNone(_batch_token_ids((), {"attention_mask": mask}))
        # 无 input_ids kwargs 时取 args[0]
        self.assertIs(_batch_token_ids((ids,), {"attention_mask": mask}), ids)
        # args[0] 是浮点张量则不取
        self.assertIsNone(_batch_token_ids((torch.randn(2, 16),), {}))


class TestMergeMutation(unittest.TestCase):
    """合并落地必须改变模型输出：跳过 _bake_merged_weights 时结果应不同。"""

    def test_skipping_bake_changes_outputs(self):
        calib = _calib_tokens()
        cfg = {
            "methods": {
                "moe": {
                    "name": "output_merge",
                    "kwargs": {"keep_ratio": 0.5, "selector": "none"},
                }
            },
            "missing_data_policy": "warn_skip",
        }
        baked = _toy_model(5)
        unbaked = copy.deepcopy(baked)
        prune(baked, cfg, data=calib)
        with mock.patch.object(
            moe_output_merge, "_bake_merged_weights", lambda *args: None
        ):
            prune(unbaked, cfg, data=calib)
        # 剪枝拓扑一致，仅烘焙权重不同
        self.assertEqual(_num_params(baked), _num_params(unbaked))
        x = torch.randint(0, VOCAB, (2, 16), generator=torch.Generator().manual_seed(9))
        with torch.no_grad():
            self.assertFalse(torch.allclose(baked(x), unbaked(x)))


if __name__ == "__main__":
    unittest.main()
