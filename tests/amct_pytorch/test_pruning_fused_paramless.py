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


_EMBED = 24


class _OPTStyleFFNBlock(nn.Module):
    """真正的两层 FFN（应被识别并剪枝）。"""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(_HIDDEN, _INTER)
        self.fc2 = nn.Linear(_INTER, _HIDDEN)

    def forward(self, x):
        return self.fc2(torch.nn.functional.gelu(self.fc1(x)))


class _OPTStyleDecoder(nn.Module):
    """OPT-350m 风词嵌入投影：project_out(hidden→embed) 先于 project_in(embed→hidden) 注册。

    注册顺序上 (project_out→project_in) 满足 in→inter→in（hidden→embed→hidden）的两层
    FFN 形状，但前向顺序上二者被整个解码器栈隔开（project_in→layers→project_out），并非
    顺序 FFN；其“中间维”实为与词表绑定的嵌入维。误当 FFN 剪枝会破坏维度，故须按命名排除。
    """

    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(40, _EMBED)
        self.project_out = nn.Linear(_HIDDEN, _EMBED, bias=False)  # 先注册（陷阱）
        self.project_in = nn.Linear(_EMBED, _HIDDEN, bias=False)
        self.layers = nn.ModuleList([_OPTStyleFFNBlock()])
        self.head = nn.Linear(_EMBED, 40, bias=False)

    def forward(self, ids):
        x = self.project_in(self.embed(ids))
        for blk in self.layers:
            x = blk(x)
        return self.head(self.project_out(x))


class TestEmbedProjectionExclusion(unittest.TestCase):
    """P0：OPT-350m 风 project_in/out 不得被误配为两层 FFN（剪则维度损坏、前向崩溃）。"""

    def test_embed_projection_not_matched_as_ffn(self):
        model = _OPTStyleDecoder().eval()
        targets = DensePruningDomain().find_targets(model, PruneConfig())
        paths = set()
        for t in targets:
            paths |= {getattr(t, "producer_path", ""), getattr(t, "consumer_path", "")}
        self.assertNotIn("project_in", paths)  # 词嵌入投影不得出现在任何目标
        self.assertNotIn("project_out", paths)
        self.assertEqual(len(targets), 1)  # 仅真正的 FFN
        self.assertEqual(targets[0].producer_path, "layers.0.fc1")
        self.assertEqual(targets[0].consumer_path, "layers.0.fc2")

    def test_embed_projection_model_prunes_and_runs(self):
        torch.manual_seed(0)
        model = _OPTStyleDecoder().eval()
        ids = torch.randint(0, 40, (4, 8))
        with torch.no_grad():
            ref_shape = model(ids).shape
        calib = [torch.randint(0, 40, (8, 8)) for _ in range(4)]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            prune(model, _cfg("reconstruct"), data=calib)
        # 投影维度保持不变（未被损坏），前向仍可运行
        self.assertEqual(model.project_in.in_features, _EMBED)
        self.assertEqual(model.project_in.out_features, _HIDDEN)
        self.assertEqual(model.project_out.in_features, _HIDDEN)
        self.assertEqual(model.project_out.out_features, _EMBED)
        with torch.no_grad():
            self.assertEqual(model(ids).shape, ref_shape)


class TestRouterSliceSquareDims(unittest.TestCase):
    """[回归] _slice_router_experts 方形(多维 == orig_n)消歧：旧 len(dims)==1 守卫会**静默不切片**，
    使路由器维度与专家数不一致（前向越界/误路由）。修复后按专家轴切片。"""

    def test_square_router_weight_slices_out_axis(self):
        from amct_pytorch.experimental.pruning.domains.moe import _slice_router_experts

        router = nn.Linear(8, 8, bias=False)  # out=in=8=orig_n（方形）
        _slice_router_experts(router, torch.tensor([0, 2, 4, 6]), orig_n=8)
        self.assertEqual(tuple(router.weight.shape), (4, 8))  # 专家轴(out)→4，hidden 维保持 8
        self.assertEqual(router.out_features, 4)

    def test_square_correction_bias_slices_last_axis(self):
        from amct_pytorch.experimental.pruning.domains.moe import _slice_router_experts

        mod = nn.Module()
        mod.register_buffer("e_score_correction_bias", torch.zeros(8, 8))  # (n_group, n_experts) 方形
        _slice_router_experts(mod, torch.tensor([0, 1, 2, 3]), orig_n=8)
        self.assertEqual(tuple(mod.e_score_correction_bias.shape), (8, 4))  # 专家轴=最后一维 → 4


if __name__ == "__main__":
    unittest.main()
