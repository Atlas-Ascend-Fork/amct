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
"""依赖图(DepGraph)CNN 通道剪枝的 E2E / 剪枝 / 集成测试。

覆盖简单“兄弟卷积”启发式无法处理的现代结构：把卷积包进独立 block、带残差/
下采样捷径的网络（手写 residual 网 + transformers 的 ResNet/ConvNext/
MobileNetV2/RegNet）。依赖图实现 static 内置于 _vendor（改编自 MIT 的
Torch-Pruning）。
"""

import os
import sys
import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))

import amct_pytorch.experimental.pruning as P
from amct_pytorch.experimental.pruning import prune, CNN_DEPGRAPH_PRUNE_CFG

try:
    import transformers

    _HAS_TF = True
except Exception:  # pragma: no cover
    _HAS_TF = False


def _params(m):
    return sum(p.numel() for p in m.parameters())


def _img_calib(n=4, b=2, hw=32):
    return [torch.randn(b, 3, hw, hw) for _ in range(n)]


# --- 手写 residual 网：每个 conv 自成子模块（兄弟启发式必然 0 目标） ---
class _ConvLayer(nn.Module):
    def __init__(self, ci, co, act=True, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(ci, co, 3, stride=stride, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(co)
        self.act = act

    def forward(self, x):
        x = self.bn(self.conv(x))
        return F.relu(x) if self.act else x


class _Block(nn.Module):
    def __init__(self, ci, co, stride=1):
        super().__init__()
        self.layer0 = _ConvLayer(ci, co, True, stride)
        self.layer1 = _ConvLayer(co, co, False)
        self.short = (
            nn.Sequential()
            if (stride == 1 and ci == co)
            else _ConvLayer(ci, co, False, stride)
        )

    def forward(self, x):
        return F.relu(self.layer1(self.layer0(x)) + self.short(x))


class _ResidualNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.stem = _ConvLayer(3, 16)
        self.b1 = _Block(16, 16)
        self.b2 = _Block(16, 32, stride=2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(32, num_classes)

    def forward(self, x):
        x = self.b2(self.b1(self.stem(x)))
        return self.fc(torch.flatten(self.pool(x), 1))


class TestDepGraphCore(unittest.TestCase):
    """无需 transformers：手写残差网（卷积分散在独立子模块）。"""

    def test_residual_net_prunes_via_facade(self):
        torch.manual_seed(0)
        model = _ResidualNet().eval()
        x = torch.randn(2, 3, 32, 32)
        with torch.no_grad():
            out_before = model(x)
        p_before = _params(model)
        stem_before = model.stem.conv.out_channels

        prune(model, CNN_DEPGRAPH_PRUNE_CFG, data=_img_calib())

        # 真正减参 + 卷积通道下降 + 分类输出维不变 + 前向正常
        self.assertLess(_params(model), p_before)
        self.assertLess(model.stem.conv.out_channels, stem_before)
        self.assertEqual(model.fc.out_features, 10)
        with torch.no_grad():
            out_after = model(x)
        self.assertEqual(out_after.shape, out_before.shape)
        self.assertTrue(torch.isfinite(out_after).all())

    def test_depgraph_advanced_options(self):
        """P3 进阶选项：hessian 重要度 / global_pruning / round_to(HW对齐) / iterative_steps。"""
        cases = [
            {"importance": "hessian", "prune_ratio": 0.3},
            {"importance": "magnitude", "prune_ratio": 0.3, "global_pruning": True},
            {"importance": "magnitude", "prune_ratio": 0.3, "round_to": 8},
            {"importance": "magnitude", "prune_ratio": 0.3, "iterative_steps": 3},
        ]
        for kw in cases:
            torch.manual_seed(0)
            model = _ResidualNet().eval()
            p0 = _params(model)
            cfg = {
                "methods": {"cnn": {"name": "depgraph", "kwargs": kw}},
                "missing_data_policy": "warn_skip",
            }
            prune(model, cfg, data=_img_calib())
            self.assertLess(_params(model), p0, f"{kw} did not prune")
            with torch.no_grad():
                out = model(torch.randn(2, 3, 32, 32))
            self.assertTrue(
                torch.isfinite(out).all(), f"{kw} produced non-finite output"
            )

    def test_depgraph_importance_options(self):
        # taylor 需要梯度（无标签 surrogate 反传）；fpgm 仅前向。均应成功剪枝。
        for imp in ("magnitude", "l1", "lamp", "random", "fpgm", "taylor"):
            torch.manual_seed(0)
            model = _ResidualNet().eval()
            p0 = _params(model)
            cfg = {
                "methods": {
                    "cnn": {
                        "name": "depgraph",
                        "kwargs": {"prune_ratio": 0.3, "importance": imp},
                    }
                },
                "missing_data_policy": "warn_skip",
            }
            prune(model, cfg, data=_img_calib())
            self.assertLess(_params(model), p0, f"importance={imp} did not prune")
            with torch.no_grad():
                out = model(torch.randn(2, 3, 32, 32))
            self.assertTrue(torch.isfinite(out).all())

    def test_tolerance_mode_with_depgraph(self):
        """容差(acc 二分搜索)模式 + 依赖图 CNN 方法集成。"""
        torch.manual_seed(0)
        model = _ResidualNet().eval()
        p0 = _params(model)
        cfg = {
            "methods": {"cnn": {"name": "depgraph"}},
            "missing_data_policy": "warn_skip",
        }
        # 用确定性参数量质量；最多掉 40%
        res = P.accuracy_based_auto_prune(
            model,
            cfg,
            data=_img_calib(),
            tolerance=0.4,
            quality_fn=lambda m: _params(m) / p0,
        )
        self.assertTrue(res.applied)
        self.assertLessEqual(res.weight_reduction, 0.4 + 1e-9)
        with torch.no_grad():
            out = model(torch.randn(2, 3, 32, 32))
        self.assertTrue(torch.isfinite(out).all())

    def test_vendored_torch_pruning_present_with_license(self):
        from amct_pytorch.experimental.pruning._vendor import torch_pruning as tp

        self.assertTrue(hasattr(tp, "DependencyGraph"))
        base = os.path.dirname(tp.__file__)
        self.assertTrue(os.path.exists(os.path.join(base, "LICENSE")))
        self.assertTrue(os.path.exists(os.path.join(base, "NOTICE.md")))


@unittest.skipUnless(_HAS_TF, "transformers not installed")
class TestDepGraphRealHF(unittest.TestCase):
    """transformers 的现代视觉网（残差/下采样/depthwise/独立 block 包裹）。"""

    def _check(self, model, x, num_labels=10):
        model.eval()
        with torch.no_grad():
            out_before = model(x).logits
        p_before = _params(model)
        prune(model, CNN_DEPGRAPH_PRUNE_CFG, data=[x.clone() for _ in range(4)])
        self.assertLess(_params(model), p_before)
        with torch.no_grad():
            out_after = model(x).logits
        self.assertEqual(out_after.shape, out_before.shape)  # 输出类别数不变
        self.assertEqual(out_after.shape[-1], num_labels)
        self.assertTrue(torch.isfinite(out_after).all())

    @unittest.skipUnless(
        _HAS_TF and hasattr(transformers, "ResNetForImageClassification"),
        "ResNet unavailable",
    )
    def test_resnet(self):
        torch.manual_seed(0)
        m = transformers.ResNetForImageClassification(
            transformers.ResNetConfig(
                num_channels=3,
                embedding_size=16,
                hidden_sizes=[32, 64],
                depths=[1, 1],
                layer_type="basic",
                num_labels=10,
            )
        )
        self._check(m, torch.randn(2, 3, 64, 64))

    @unittest.skipUnless(
        _HAS_TF and hasattr(transformers, "ConvNextForImageClassification"),
        "ConvNext unavailable",
    )
    def test_convnext(self):
        torch.manual_seed(0)
        m = transformers.ConvNextForImageClassification(
            transformers.ConvNextConfig(num_labels=10)
        )
        self._check(m, torch.randn(2, 3, 64, 64))

    @unittest.skipUnless(
        _HAS_TF and hasattr(transformers, "MobileNetV2ForImageClassification"),
        "MobileNetV2 unavailable",
    )
    def test_mobilenet_v2(self):
        torch.manual_seed(0)
        m = transformers.MobileNetV2ForImageClassification(
            transformers.MobileNetV2Config(num_channels=3, num_labels=10)
        )
        self._check(m, torch.randn(2, 3, 64, 64))

    @unittest.skipUnless(
        _HAS_TF and hasattr(transformers, "RegNetForImageClassification"),
        "RegNet unavailable",
    )
    def test_regnet(self):
        torch.manual_seed(0)
        m = transformers.RegNetForImageClassification(
            transformers.RegNetConfig(num_channels=3, num_labels=10)
        )
        self._check(m, torch.randn(2, 3, 64, 64))


if __name__ == "__main__":
    sys.exit(unittest.main(verbosity=2))
