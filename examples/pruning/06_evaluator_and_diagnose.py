# -*- coding: UTF-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
"""样例 06：自定义精度抽象（evaluator / quality_fn）与剪枝可行性诊断。

演示要点：
 - evaluator：与量化侧 accuracy_based_auto_calibration 共用同一精度抽象 ——
   任意带 .evaluate(model, iterations) -> float 的对象即可驱动剪枝精度搜索；
   （注意：量化内置 graph 版 ModelEvaluator.evaluate 不返回值，需用返回 metric 的实现）
 - quality_fn：直接传自定义精度函数（优先级 quality_fn > evaluator > 默认 top-1 保真度）；
 - diagnose()：在不改模型的前提下，报告各域可剪目标，提前发现"无目标/不支持"。
运行：python examples/pruning/06_evaluator_and_diagnose.py
"""

import torch

from _models import make_mlp, mlp_data
import amct_pytorch.experimental.pruning as P


class RatioEvaluator:
    """最小可用 evaluator：返回一个随剪枝单调下降的代理精度（这里用参数留存比例）。

    真实场景里把它换成"在验证集上跑 model 并返回准确率/mAP"即可，同一个对象
    也能传给 amct.accuracy_based_auto_calibration 的 evaluator，两侧复用。
    """

    def __init__(self, full_params):
        self.full = full_params

    def evaluate(self, model, iterations=1):
        cur = sum(p.numel() for p in model.parameters())
        return cur / self.full


def demo_evaluator():
    model = make_mlp()
    calib = mlp_data()
    full = sum(p.numel() for p in model.parameters())
    res = P.accuracy_based_auto_prune(
        model,
        data=calib,
        tolerance=0.1,
        evaluator=RatioEvaluator(full),
        apply=True,
    )
    print("[06-evaluator]", res.summary())


def demo_quality_fn():
    model = make_mlp()
    calib = mlp_data()

    ref = model(calib[0]).argmax(-1)

    def quality_fn(m):  # 越大越好；这里用与原模型 top-1 一致率
        with torch.no_grad():
            pred = m(calib[0]).argmax(-1)
        return (pred == ref).float().mean().item()

    res = P.accuracy_based_auto_prune(
        model, data=calib, tolerance=0.05, quality_fn=quality_fn
    )
    print("[06-quality_fn]", res.summary())


def demo_diagnose():
    model = make_mlp()
    calib = mlp_data()
    rep = P.diagnose(model, data=calib)
    print("[06-diagnose]", rep.as_dict() if hasattr(rep, "as_dict") else rep)


def main():
    demo_evaluator()
    demo_quality_fn()
    demo_diagnose()


if __name__ == "__main__":
    main()
