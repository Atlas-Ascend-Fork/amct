# -*- coding: UTF-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
"""样例 12：守卫式 variance-HYBRID 的 MoE 专家剪枝（部署式"实际应用"）。

演示要点：
 - 一次校准 → 在判据 MENU {mass, cond_var, peak, cvxpeak} 上各剪一份副本；
 - 用一个**独立的小验证集**实测择优（默认 top-1 保真度，无需标签）；
 - 仅当某方差判据*严格优于* fallback(mass) 时才切换 —— 守卫保证不劣于 mass；
 - 把选中判据就地应用，得到可部署的剪枝模型 + 真实指标（res.summary()）。
运行：python examples/pruning/moe_variance_hybrid.py
"""

from _models import make_moe, moe_data, count_params
from amct_pytorch.experimental.pruning import guarded_variance_hybrid_prune


def _num_experts(m):
    for layer in m.layers:
        return layer.num_experts
    return None


def main():
    model = make_moe()
    calib = moe_data(n=8)  # 校准集
    val = moe_data(n=3, batch=2)  # 守卫用的验证集（与校准不同）
    before = count_params(model)
    print(f"[12] 原始：参数 {before:,}，每层专家数 {_num_experts(model)}")

    # 实际应用：跑 HYBRID 并就地应用最优判据，得到可部署的剪枝模型。
    res = guarded_variance_hybrid_prune(
        model, data=calib, prune_ratio=0.5, top_k=2, eval_data=val, apply=True
    )
    _ = model(val[0])  # 剪后前向正常
    print("[12]", res.summary())
    print(
        f"[12] 选中判据={res.chosen_criterion}（fallback={res.fallback_criterion}），"
        f"每层专家数 {_num_experts(model)} 前向✅"
    )
    # 守卫不变量：验证集上绝不劣于 fallback。
    assert res.gain_over_fallback >= 0.0


if __name__ == "__main__":
    main()
