# -*- coding: UTF-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
"""样例 03：现代 CNN（残差/下采样）的依赖图通道剪枝，并遍历多种重要度准则。

演示要点：
 - 残差/下采样结构必须用 depgraph（依赖图）方法，自动成组裁剪耦合层；
 - importance 可选 {magnitude, l1, lamp, bn, random}（来自内置 Torch-Pruning）；
 - 自动忽略输出头，保持类别数不变。
运行：python examples/pruning/03_cnn_depgraph_importance.py
"""

import copy

import amct_pytorch.experimental.pruning as P
from _models import make_cnn, cnn_data, count_params


def main():
    base = make_cnn()
    calib = cnn_data()
    before = count_params(base)

    # taylor 需要校准数据反传梯度（无标签，用输出均方 surrogate）；其余仅前向
    for imp in ("magnitude", "l1", "lamp", "bn", "random", "fpgm", "taylor"):
        model = copy.deepcopy(base)
        cfg = {
            "methods": {
                "cnn": {
                    "name": "depgraph",
                    "kwargs": {"prune_ratio": 0.3, "importance": imp},
                }
            }
        }
        P.prune(model, cfg, data=calib)
        after = count_params(model)
        _ = model(calib[0])  # 前向校验
        print(
            f"[03] importance={imp:9s} 通道剪枝 0.3：参数 {before:,} → {after:,} "
            f"(削减 {100 * (1 - after / before):.1f}%) 前向✅"
        )

    # 也可直接用预设
    model = copy.deepcopy(base)
    P.prune(model, P.CNN_DEPGRAPH_PRUNE_CFG, data=calib)
    print(f"[03] 预设 CNN_DEPGRAPH_PRUNE_CFG：参数 → {count_params(model):,}")


if __name__ == "__main__":
    main()
