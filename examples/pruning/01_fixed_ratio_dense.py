# -*- coding: UTF-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
"""样例 01：固定剪枝率的 dense FFN 结构化剪枝（in-place，与 amct.quantize 同风格）。

演示要点：
 - dict 配置直接传入 prune()，无需 .cfg 文件；
 - dense 域自动避开注意力投影（此处为纯 FFN，无需手动 skip）；
 - 统计经 get_last_report() 获取。
运行：python examples/pruning/01_fixed_ratio_dense.py
"""

from _models import make_mlp, mlp_data, count_params
import amct_pytorch.experimental.pruning as P


def main():
    model = make_mlp()
    calib = mlp_data()
    before = count_params(model)

    cfg = {
        "methods": {"dense": {"name": "low_variance", "kwargs": {"prune_ratio": 0.5}}}
    }
    P.prune(model, cfg, data=calib)  # in-place

    after = count_params(model)
    print(
        f"[01] dense FFN 固定率 0.5：参数 {before:,} → {after:,} "
        f"(削减 {100 * (1 - after / before):.1f}%)"
    )
    print("[01] report:", P.get_last_report().as_dict())
    # 前向仍可用
    _ = model(calib[0])
    print("[01] 剪枝后前向通过 ✅")


if __name__ == "__main__":
    main()
