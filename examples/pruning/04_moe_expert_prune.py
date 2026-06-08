# -*- coding: UTF-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
"""样例 04：MoE 专家剪枝（gate + ModuleList experts 老式结构）。

演示要点：
 - moe 域支持两类方法：activation_count（按激活频次）/ mass_variance（按质量方差）；
 - 按校准数据上的专家利用率裁掉贡献最小的专家，并同步收缩 gate。
运行：python examples/pruning/04_moe_expert_prune.py
"""

import copy

import amct_pytorch.experimental.pruning as P
from _models import make_moe, moe_data, count_params


def _num_experts(m):
    for layer in m.layers:
        return layer.num_experts
    return None


def main():
    base = make_moe()
    calib = moe_data()
    before = count_params(base)
    print(f"[04] 原始：参数 {before:,}，每层专家数 {_num_experts(base)}")

    for method in ("activation_count", "mass_variance"):
        model = copy.deepcopy(base)
        kwargs = {"prune_ratio": 0.5}
        if method == "mass_variance":
            kwargs["boundary"] = 10
        cfg = {"methods": {"moe": {"name": method, "kwargs": kwargs}}}
        P.prune(model, cfg, data=calib)
        _ = model(calib[0])
        print(
            f"[04] method={method:16s} 剪 50% 专家：参数 → {count_params(model):,}，"
            f"每层专家数 {_num_experts(model)} 前向✅"
        )


if __name__ == "__main__":
    main()
