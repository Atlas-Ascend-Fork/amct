# -*- coding: UTF-8 -*-
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
"""样例公用的微型模型与数据（随机初始化，**不下载任何权重**，纯 CPU 可跑）。

这些结构刻意覆盖三种剪枝域，便于在没有联网/没有 NPU 的环境演示 amct 剪枝：
 - dense FFN（``MiniMLP``）        → dense 域：剪中间维
 - CNN 残差结构（``MiniResCNN``）  → cnn 域：依赖图剪通道
 - MoE（``MiniMoE``）              → moe 域：剪专家
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------
# dense FFN
# --------------------------------------------------------------------------
class MiniMLP(nn.Module):
    """两层 FFN 堆叠（hidden→inter→hidden），dense 域剪中间维。"""

    def __init__(self, vocab=256, hidden=64, inter=256, layers=2):
        super().__init__()
        self.embed = nn.Embedding(vocab, hidden)
        self.blocks = nn.ModuleList(
            [
                nn.ModuleDict(
                    {"fc1": nn.Linear(hidden, inter), "fc2": nn.Linear(inter, hidden)}
                )
                for _ in range(layers)
            ]
        )
        self.head = nn.Linear(hidden, vocab)

    def forward(self, input_ids):
        h = self.embed(input_ids)
        for b in self.blocks:
            h = h + b["fc2"](F.gelu(b["fc1"](h)))
        return self.head(h)


def make_mlp():
    torch.manual_seed(0)
    return MiniMLP().eval()


def mlp_data(n=6, batch=4, seq=16, vocab=256):
    torch.manual_seed(1)
    return [torch.randint(0, vocab, (batch, seq)) for _ in range(n)]


# --------------------------------------------------------------------------
# CNN（带残差/下采样捷径 → 必须用依赖图剪枝）
# --------------------------------------------------------------------------
class ResBlock(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.short = (
            nn.Sequential()
            if stride == 1 and cin == cout
            else nn.Sequential(
                nn.Conv2d(cin, cout, 1, stride=stride, bias=False), nn.BatchNorm2d(cout)
            )
        )

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return F.relu(y + self.short(x))


class MiniResCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.stem = nn.Conv2d(3, 16, 3, padding=1, bias=False)
        self.b1 = ResBlock(16, 32, stride=2)
        self.b2 = ResBlock(32, 64, stride=2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(64, num_classes)

    def forward(self, x):
        x = F.relu(self.stem(x))
        x = self.b2(self.b1(x))
        x = self.pool(x).flatten(1)
        return self.head(x)


def make_cnn():
    torch.manual_seed(0)
    return MiniResCNN().eval()


def cnn_data(n=4, batch=4, size=16):
    torch.manual_seed(2)
    return [torch.randn(batch, 3, size, size) for _ in range(n)]


# --------------------------------------------------------------------------
# MoE（gate + ModuleList experts）→ moe 域剪专家
# --------------------------------------------------------------------------
class Expert(nn.Module):
    def __init__(self, hidden, inter):
        super().__init__()
        self.fc1 = nn.Linear(hidden, inter)
        self.fc2 = nn.Linear(inter, hidden)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class MoELayer(nn.Module):
    def __init__(self, hidden=64, inter=128, num_experts=8, topk=2):
        super().__init__()
        self.num_experts, self.topk = num_experts, topk
        self.gate = nn.Linear(hidden, num_experts, bias=False)
        self.experts = nn.ModuleList(
            [Expert(hidden, inter) for _ in range(num_experts)]
        )

    def forward(self, h):
        b, s, d = h.shape
        flat = h.reshape(-1, d)
        probs = F.softmax(self.gate(flat), dim=-1)
        w, idx = torch.topk(probs, self.topk, dim=-1)
        w = w / w.sum(-1, keepdim=True)
        out = torch.zeros_like(flat)
        for k in range(self.topk):
            sel = idx[:, k]
            for e in range(self.num_experts):
                m = sel == e
                if m.any():
                    out[m] += w[m, k] * self.experts[e](flat[m])
        return out.reshape(b, s, d)


class MiniMoE(nn.Module):
    def __init__(self, vocab=256, hidden=64, layers=2):
        super().__init__()
        self.embed = nn.Embedding(vocab, hidden)
        self.layers = nn.ModuleList([MoELayer(hidden=hidden) for _ in range(layers)])
        self.head = nn.Linear(hidden, vocab)

    def forward(self, input_ids):
        h = self.embed(input_ids)
        for layer in self.layers:
            h = h + layer(h)
        return self.head(h)


def make_moe():
    torch.manual_seed(0)
    return MiniMoE().eval()


def moe_data(n=6, batch=4, seq=16, vocab=256):
    torch.manual_seed(3)
    return [torch.randint(0, vocab, (batch, seq)) for _ in range(n)]


def count_params(m):
    return sum(p.numel() for p in m.parameters())
