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
"""剪枝与量化联合使用的端到端测试（prune → quantize → convert），并从
**量化性能保持**角度验证：在剪枝后的模型上量化，量化步骤不应显著改变
（剪枝后）模型的预测。

amct.quantize/convert 在 CPU 上对小模型可运行；若环境缺失则整组跳过。
"""
import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.dirname(__file__))

from mini_models import create_mini_mlp
import amct_pytorch.experimental.pruning as P

try:
    import amct_pytorch as amct
    _HAS_QUANT = hasattr(amct, "quantize") and hasattr(amct, "convert")
except Exception:  # pragma: no cover
    _HAS_QUANT = False


def _tok(n=6):
    return [torch.randint(0, 1000, (4, 20)) for _ in range(n)]


def _params(m):
    return sum(p.numel() for p in m.parameters())


def _top1_agree(a, b):
    return (a.argmax(-1) == b.argmax(-1)).float().mean().item()


DENSE_CFG = {
    "methods": {"dense": {"name": "low_variance", "kwargs": {"prune_ratio": 0.5}}},
    "skip_layers": ["self_attn"],
    "missing_data_policy": "warn_skip",
}


@unittest.skipUnless(_HAS_QUANT, "amct.quantize/convert unavailable")
class TestPruneThenQuantize(unittest.TestCase):

    def _fresh(self, state=None):
        m, _ = create_mini_mlp()
        if state is not None:
            m.load_state_dict(state)
        m.eval()
        return m

    def test_prune_then_quantize_pipeline_runs(self):
        """prune → quantize → convert 端到端可运行，输出形状/有限性保持。"""
        x = torch.randint(0, 1000, (4, 20))
        model = self._fresh()
        with torch.no_grad():
            shape_before = model(x).shape

        p_before = _params(model)
        P.prune(model, DENSE_CFG, data=_tok())
        self.assertLess(_params(model), p_before)         # 剪枝确实减参

        amct.quantize(model)                               # 在剪枝后的模型上量化
        with torch.no_grad():
            out_q = model(x)
        self.assertEqual(out_q.shape, shape_before)
        self.assertTrue(torch.isfinite(out_q).all())

        amct.convert(model)                                # 转换部署模型
        with torch.no_grad():
            out_c = model(x)
        self.assertEqual(out_c.shape, shape_before)
        self.assertTrue(torch.isfinite(out_c).all())

    def test_quantization_retention_on_pruned_model(self):
        """量化性能保持：量化前后（均为剪枝后模型）预测应高度一致。

        retention = top1-agree(剪枝+量化 的输出, 剪枝后未量化 的输出)。
        """
        x = torch.randint(0, 1000, (8, 20))
        state = self._fresh().state_dict()

        # 剪枝后（未量化）
        pruned = self._fresh(state)
        P.prune(pruned, DENSE_CFG, data=_tok())
        with torch.no_grad():
            out_pruned = pruned(x)

        # 同样剪枝 + 量化
        pq = self._fresh(state)
        P.prune(pq, DENSE_CFG, data=_tok())
        amct.quantize(pq)
        with torch.no_grad():
            out_pq = pq(x)

        retention = _top1_agree(out_pruned, out_pq)
        # 量化步骤应保持（剪枝后）模型的绝大多数预测
        self.assertGreaterEqual(retention, 0.9)
        self.assertTrue(torch.isfinite(out_pq).all())

    def test_tolerance_prune_then_quantize(self):
        """容差模式（acc 二分搜索）选率剪枝后再量化，端到端可运行。"""
        x = torch.randint(0, 1000, (4, 20))
        # 用参数量作确定性质量，保证选出一个剪枝率（随机权重 top-1 噪声不适合断言）
        model = self._fresh()
        p0 = _params(model)
        P.prune(model, DENSE_CFG, data=_tok(), tolerance=0.3,
                quality_fn=lambda m: _params(m) / p0)
        res = P.get_last_auto_prune()
        self.assertTrue(res.applied)

        amct.quantize(model)
        with torch.no_grad():
            out = model(x)
        self.assertTrue(torch.isfinite(out).all())


if __name__ == "__main__":
    sys.exit(unittest.main(verbosity=2))
