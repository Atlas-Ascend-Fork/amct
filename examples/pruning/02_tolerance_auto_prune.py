# -*- coding: UTF-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
"""样例 02：容差驱动的自动剪枝（面向非专家，"只说能接受多少精度损失"）。

演示要点：
 - 只给 tolerance，自动在 ratio_grid 上**二分查找**满足容差的最大剪枝率；
 - 默认保真度指标无需标签（以原模型 top-1 预测为参考）；
 - safe_skip_attention 默认开，自动保护注意力；
 - 结果经 get_last_auto_prune().summary() 获取，含每个候选率的 trial。
运行：python examples/pruning/02_tolerance_auto_prune.py
"""

import amct_pytorch.experimental.pruning as P
from _models import make_mlp, mlp_data


def _run(tol):
    model = make_mlp()
    calib = mlp_data()
    P.prune(model, data=calib, tolerance=tol)  # 仅给容差，其余全自动
    res = P.get_last_auto_prune()
    print(f"\n[02] tolerance={tol}")
    print("     ", res.summary())
    print(
        f"      baseline={res.baseline_quality:.3f}  chosen_ratio={res.chosen_ratio}  "
        f"applied={res.applied}"
    )
    print("      二分查找触及的候选率 trial：")
    for t in res.trials:
        flag = "✓接受" if t.accepted else "✗超容差"
        print(
            f"        ratio={t.prune_ratio:.2f}  quality={t.quality:.3f}  "
            f"drop={t.quality_drop:.3f}  {flag}"
        )


def main():
    # 紧容差：可能一个都不满足 → 保守地不剪（applied=False，安全方向）
    _run(0.05)
    # 放宽容差：二分查找选出满足容差的最大剪枝率并就地应用
    _run(0.20)


if __name__ == "__main__":
    main()
