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

try:
    import torch_npu

    # 注意：conftest.py 会把 torch_npu mock 成 MagicMock 以便在 CPU 上 import 仓库代码。
    # MagicMock 的 is_available() 返回 truthy 的 MagicMock，会**误判为有 NPU**。
    # 故用严格 `is True` 比较，把 mock 滤掉，只在真实 NPU 上为 True。
    _HAS_NPU = (torch_npu is not None) and (torch.npu.is_available() is True)
except Exception:  # pragma: no cover
    _HAS_NPU = False


def _quant_module_count(model):
    """统计真正被替换成量化模块的层数（名字含 'Quant'）。

    关键事实：amct 量化只对 **fp16/bf16** 权重发火，且 fake-quant 需要 ``torch_npu``。
    在 CPU+fp32 上 quantize/convert 是 **no-op**（不插入任何量化模块）。本计数用于把
    "量化其实没发生" 的情形与真正量化区分开，避免测试因 no-op 而 **假性通过**。
    """
    return sum(1 for m in model.modules() if "Quant" in type(m).__name__)


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
        self.assertLess(_params(model), p_before)  # 剪枝确实减参

        amct.quantize(model)  # 在剪枝后的模型上量化
        with torch.no_grad():
            out_q = model(x)
        self.assertEqual(out_q.shape, shape_before)
        self.assertTrue(torch.isfinite(out_q).all())

        amct.convert(model)  # 转换部署模型
        with torch.no_grad():
            out_c = model(x)
        self.assertEqual(out_c.shape, shape_before)
        self.assertTrue(torch.isfinite(out_c).all())

    def test_quantization_retention_on_pruned_model(self):
        """量化性能保持：在 **真正发火** 的量化下，量化前后（均为剪枝后模型）预测高度一致。

        关键：必须先确认量化真的插入了量化模块，否则 retention 会因 no-op 而 **假性=1.0**
        给出虚假信心。本测试在 NPU + fp16 上运行，并显式断言量化已发火。
        """
        if not _HAS_NPU:
            self.skipTest("real quantization needs fp16 weights + torch_npu (NPU)")
        dev = "npu:0"
        x = torch.randint(0, 1000, (8, 20)).to(dev)
        state = self._fresh().state_dict()
        # 两份必须用**同一** calib 剪枝，否则 low_variance 会按不同随机数据剪掉不同神经元，
        # retention 就会混入"剪枝差异"而非纯"量化效果"。共用 calib 才能隔离量化的影响。
        calib = _tok()

        pruned = self._fresh(state).half().to(dev)
        P.prune(pruned, DENSE_CFG, data=calib)
        with torch.no_grad():
            out_pruned = pruned(x)

        pq = self._fresh(state).half().to(dev)
        P.prune(pq, DENSE_CFG, data=calib)
        amct.quantize(pq)
        # 反假性通过：量化必须真的插入量化模块
        self.assertGreater(
            _quant_module_count(pq),
            0,
            "quantize() did not insert any quant modules — retention would be meaningless",
        )
        with torch.no_grad():
            out_pq = pq(x)

        retention = _top1_agree(out_pruned.float(), out_pq.float())
        self.assertGreaterEqual(retention, 0.9)
        self.assertTrue(torch.isfinite(out_pq).all())

    def test_cpu_quantize_is_noop_documented(self):
        """文档化既知行为：CPU+fp32 上 amct 量化是 no-op（不插入量化模块）。

        这把 "量化没发生" 显式钉成断言，防止他人误以为 CPU 上跑了真量化（历史假性通过的根因）。
        """
        m = self._fresh()  # fp32 / CPU
        amct.quantize(m)
        self.assertEqual(
            _quant_module_count(m),
            0,
            "CPU+fp32 quantize unexpectedly inserted quant modules — assumption changed",
        )

    def test_quantization_actually_fires_and_perturbs(self):
        """量化在 NPU+fp16 上 **真正发火**：插入量化模块且改变输出（区别于 CPU no-op）。"""
        if not _HAS_NPU:
            self.skipTest("needs torch_npu (NPU)")
        dev = "npu:0"
        m = self._fresh().half().to(dev)
        x = torch.randint(0, 1000, (4, 20)).to(dev)
        with torch.no_grad():
            y0 = m(x).float().cpu()
        amct.quantize(m)
        self.assertGreater(_quant_module_count(m), 0)
        amct.convert(m)
        deploy_types = {
            type(mm).__name__ for mm in m.modules() if "Quant" in type(mm).__name__
        }
        self.assertTrue(deploy_types, "no deploy quant modules after convert")
        with torch.no_grad():
            y2 = m(x).float().cpu()
        # INT8 权重量化应带来非零（但有界）的输出扰动
        self.assertGreater((y2 - y0).abs().max().item(), 0.0)
        self.assertTrue(torch.isfinite(y2).all())

    def test_tolerance_prune_then_quantize(self):
        """容差模式（acc 二分搜索）选率剪枝后再量化，端到端可运行。"""
        x = torch.randint(0, 1000, (4, 20))
        # 用参数量作确定性质量，保证选出一个剪枝率（随机权重 top-1 噪声不适合断言）
        model = self._fresh()
        p0 = _params(model)
        P.prune(
            model,
            DENSE_CFG,
            data=_tok(),
            tolerance=0.3,
            quality_fn=lambda m: _params(m) / p0,
        )
        res = P.get_last_auto_prune()
        self.assertTrue(res.applied)

        amct.quantize(model)
        with torch.no_grad():
            out = model(x)
        self.assertTrue(torch.isfinite(out).all())


if __name__ == "__main__":
    sys.exit(unittest.main(verbosity=2))
