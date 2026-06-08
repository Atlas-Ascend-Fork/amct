# -*- coding: UTF-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
"""样例 03：CNN 通道剪枝 —— 自研输出保留(重构)方法 vs 朴素切片(variance)。

演示要点：
 - cnn 域识别 (producer conv → bn → consumer) 顺序通道目标（无任何第三方依赖）；
 - ``reconstruct``：剪掉低显著度通道后用最小二乘重构 consumer 权重补偿，低破坏；
 - ``variance_channel``：按激活方差朴素切片，作为对照；
 - 自动避开输出头，保持类别数不变。
运行：python examples/pruning/03_cnn_depgraph_importance.py
"""

import copy

import amct_pytorch.experimental.pruning as P
from _models import make_cnn, cnn_data, count_params


def main():
    base = make_cnn()
    calib = cnn_data()
    before = count_params(base)

    for name in ("variance_channel", "reconstruct"):
        model = copy.deepcopy(base)
        cfg = {"methods": {"cnn": {"name": name, "kwargs": {"prune_ratio": 0.3}}}}
        P.prune(model, cfg, data=calib)
        after = count_params(model)
        _ = model(calib[0])  # 前向校验
        print(
            f"[03] cnn.{name:16s} 通道剪枝 0.3：参数 {before:,} → {after:,} "
            f"(削减 {100 * (1 - after / before):.1f}%) 前向✅"
        )

    # 也可直接用预设（重构方法）
    model = copy.deepcopy(base)
    P.prune(model, P.CNN_RECONSTRUCT_PRUNE_CFG, data=calib)
    print(f"[03] 预设 CNN_RECONSTRUCT_PRUNE_CFG：参数 → {count_params(model):,}")


if __name__ == "__main__":
    main()
