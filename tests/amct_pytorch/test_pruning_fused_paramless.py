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
"""无参模块透明性探针 + 融合 gate_up 路径（low_variance / reconstruct）专项用例。"""

import unittest
import warnings

import torch
import torch.nn as nn

from amct_pytorch.experimental.pruning import prune
from amct_pytorch.experimental.pruning.config import PruneConfig
from amct_pytorch.experimental.pruning.domains.dense import DensePruningDomain


_HIDDEN = 32
_INTER = 48


class _GluHalver(nn.Module):
    """GLU 风格无参折半模块：末维减半 → 对通道维不透明，必须中断 two-layer 链。"""

    def forward(self, x):
        gate, up = x.chunk(2, dim=-1)
        return torch.nn.functional.silu(gate) * up


class _GluHalverMlp(nn.Module):
    """fc1 输出维与 fc2 输入维相同（48→48），但中间 GLU 折半改变通道语义。"""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(_HIDDEN, _INTER)
        self.glu = _GluHalver()
        self.fc2 = nn.Linear(_INTER, _HIDDEN)

    def forward(self, x):
        y = self.glu(self.fc1(x))
        return self.fc2(torch.cat([y, y], dim=-1))


class _TransparentAct(nn.Module):
    """形状保持的无参自定义激活：探针应判透明，链不中断。"""

    def forward(self, x):
        return torch.nn.functional.gelu(x)


class _TwoLayerActMlp(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(_HIDDEN, _INTER)
        self.act = _TransparentAct()
        self.fc2 = nn.Linear(_INTER, _HIDDEN)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class _FusedMlp(nn.Module):
    """Phi-3/GLM-4 风格融合 FFN：gate_up_proj 前半 gate、后半 up。"""

    def __init__(self):
        super().__init__()
        self.gate_up_proj = nn.Linear(_HIDDEN, 2 * _INTER)
        self.down_proj = nn.Linear(_INTER, _HIDDEN)

    def forward(self, x):
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.down_proj(torch.nn.functional.silu(gate) * up)


def _calib():
    # 多样校准：token 数（160）远大于剪后中间维（24），重构方能确定
    return [torch.randn(8, _HIDDEN) for _ in range(20)]


def _cfg(method):
    return {
        "methods": {"dense": {"name": method, "kwargs": {"prune_ratio": 0.5}}},
        "missing_data_policy": "warn_skip",
    }


def _zero_dead_channels(model):
    """同时清零 gate 半区与 up 半区的死通道（权重+偏置），其输出恒为 0、方差为 0。"""
    dead = list(range(_INTER // 2, _INTER))
    with torch.no_grad():
        for i in dead:
            for row in (i, _INTER + i):
                model.gate_up_proj.weight[row].zero_()
                model.gate_up_proj.bias[row].zero_()


class TestParamlessTransparencyProbe(unittest.TestCase):
    """P0：无参模块需形状探针验证；改变末维（GLU 折半）必须中断 FFN 链。"""

    def test_glu_halver_breaks_chain_no_target(self):
        model = _GluHalverMlp().eval()
        targets = DensePruningDomain().find_targets(model, PruneConfig())
        self.assertEqual(targets, [])  # 折半模块不透明 → 不得识别 fc1→fc2

    def test_glu_halver_model_not_pruned(self):
        torch.manual_seed(0)
        model = _GluHalverMlp().eval()
        p0 = sum(p.numel() for p in model.parameters())
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            prune(model, _cfg("low_variance"), data=_calib())
        self.assertEqual(sum(p.numel() for p in model.parameters()), p0)

    def test_shape_preserving_act_keeps_chain(self):
        model = _TwoLayerActMlp().eval()
        targets = DensePruningDomain().find_targets(model, PruneConfig())
        self.assertEqual(len(targets), 1)  # 透明激活不中断链


class TestFusedGateUpCoverage(unittest.TestCase):
    """P1：融合 gate_up 的 low_variance 分支与域内半区切片，破坏任一则断言失败。"""

    def test_fused_low_variance_exact_on_zero_channels(self):
        # 死通道贡献为 0 → 剪除后输出必须逐元素一致；
        # 打分分支（前半+后半方差）或 fused_keep 偏移错误都会破坏该等式
        ref, out = self._prune_zeroed_fused("low_variance")
        self.assertLess((out - ref).abs().max().item(), 1e-5)

    def test_fused_reconstruct_preserves_on_zero_channels(self):
        ref, out = self._prune_zeroed_fused("reconstruct")
        self.assertLess(((out - ref).norm() / ref.norm()).item(), 0.05)

    def _prune_zeroed_fused(self, method):
        torch.manual_seed(0)
        model = _FusedMlp().eval()
        _zero_dead_channels(model)
        x = torch.randn(4, _HIDDEN)
        with torch.no_grad():
            ref = model(x)
        prune(model, _cfg(method), data=_calib())
        # 半区切片正确性：gate_up 输出维必须保持 down 输入维的 2 倍
        self.assertEqual(model.down_proj.in_features, _INTER // 2)
        self.assertEqual(model.gate_up_proj.out_features, 2 * (_INTER // 2))
        with torch.no_grad():
            out = model(x)
        return ref, out


if __name__ == "__main__":
    unittest.main()
