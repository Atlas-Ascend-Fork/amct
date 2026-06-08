# -*- coding: UTF-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
"""样例 05：剪枝 → 量化 → 转换 联合压缩（两条等价路径）。

剪枝（减神经元/专家）与量化（降比特）正交，体积收益相乘。推荐顺序：
**先剪枝，再量化，最后 convert** —— 量化须在剪枝后的权重上 calibration。

本样例给出两种串接方式：
 A) 函数式：prune() → amct.quantize() → amct.convert()（推荐，最简单）
 B) Pass 式：在 ModelOptimizer 中以 PrunePass 完成剪枝，再量化。
    注意：InsertQuantizeModulePass 需要在**剪枝后**的模型上解析 layer_config，
    因此不能在同一个 optimizer 里把两者无脑串到一起 —— 必须先让 PrunePass 改完形状，
    再基于剪枝后的模型解析量化配置。
运行：python examples/pruning/05_prune_then_quantize.py
"""

import amct_pytorch.experimental.pruning as P
from _models import make_mlp, mlp_data, count_params

try:
    import amct_pytorch as amct

    HAS_QUANT = hasattr(amct, "quantize") and hasattr(amct, "convert")
except Exception:
    HAS_QUANT = False


def path_a():
    model = make_mlp()
    calib = mlp_data()
    before = count_params(model)
    P.prune(
        model,
        {
            "methods": {
                "dense": {"name": "low_variance", "kwargs": {"prune_ratio": 0.5}}
            }
        },
        data=calib,
    )
    print(f"[05-A] 剪枝后参数 {before:,} → {count_params(model):,}")
    if HAS_QUANT:
        amct.quantize(model)  # 在剪枝后的模型上量化
        amct.convert(model)
        _ = model(calib[0])
        print("[05-A] quantize→convert 完成，前向✅")
    else:
        print("[05-A] 环境无 amct.quantize/convert，跳过量化（剪枝部分已验证）")


def path_b():
    from amct_pytorch.classic.optimizer import ModelOptimizer

    model = make_mlp()
    calib = mlp_data()

    # 1) 用 ModelOptimizer 跑 PrunePass（验证它符合 .run(model) 的 Pass 协议）
    opt = ModelOptimizer()
    opt.add_pass(
        P.PrunePass(
            {
                "methods": {
                    "dense": {"name": "low_variance", "kwargs": {"prune_ratio": 0.5}}
                }
            },
            data=calib,
        )
    )
    opt.do_optimizer(model)
    print(f"[05-B] ModelOptimizer(PrunePass) 完成，剪枝后参数 {count_params(model):,}")

    # 2) 在剪枝后的模型上量化（layer_config 必须基于剪枝后形状解析）
    if HAS_QUANT:
        amct.quantize(model)
        amct.convert(model)
        _ = model(calib[0])
        print("[05-B] 剪枝后 quantize→convert 完成，前向✅")
    else:
        print("[05-B] 无量化环境，PrunePass 协议已验证")


def main():
    path_a()
    path_b()


if __name__ == "__main__":
    main()
