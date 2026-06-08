# -*- coding: UTF-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
"""演示：acc 二分搜索 **是否考虑剪枝后微调恢复**，会选出截然不同的剪枝率。

要点（用户提出的关键交互）：一次性结构剪枝会让精度崩到接近随机，若搜索按"剪枝直后"
评估，则几乎选不出任何率（过度保守）；给 ``finetune_fn`` 后，搜索按"剪枝+恢复后"评估，
就能选到真正可用的更激进的率。本脚本在 MNIST 上用**真实 test accuracy** 对比两者。

运行：python examples/pruning/e2e_search_finetune_compare.py
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(__file__))
import torch, torch.nn.functional as F
from e2e_mnist_real_acc import SmallCNN, loaders, accuracy, train

import amct_pytorch.experimental.pruning as P

RESULT = os.path.join(os.environ.get("AMCT_PRUNE_RESULT_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")), "search_finetune_compare.json")
RATIOS = (0.1, 0.2, 0.3, 0.4, 0.5)


def main():
    torch.manual_seed(0)
    trl, tel = loaders()
    model = SmallCNN()
    train(model, trl, epochs=3)
    acc0 = accuracy(model, tel)
    print(f"[CMP] baseline test acc = {acc0:.4f}")

    calib = [x for x, _ in [next(iter(trl))]]
    ft_data = [(x, y) for i, (x, y) in zip(range(15), trl)]

    # 真实任务质量 = test accuracy（越大越好）；tolerance = 允许的 acc 下降
    def acc_quality(m):
        return accuracy(m, tel)

    def do_finetune(m):
        P.finetune(m, ft_data, loss_fn=lambda mm, b: F.cross_entropy(mm(b[0]), b[1]),
                   steps=150, lr=5e-4)

    cfg = {"methods": {"cnn": {"name": "depgraph", "kwargs": {"importance": "taylor"}}},
           "missing_data_policy": "warn_skip"}
    tol = 0.05   # 最多接受 5% acc 下降

    # (A) 不考虑微调：按剪枝直后评估
    import copy
    t0 = time.time()
    rA = P.accuracy_based_auto_prune(copy.deepcopy(model), cfg, data=calib, tolerance=tol,
                                     quality_fn=acc_quality, ratio_grid=RATIOS, apply=True)
    print(f"[CMP] (A) 无微调搜索: chosen={rA.chosen_ratio} applied={rA.applied} "
          f"acc_drop={rA.quality_drop}  {time.time()-t0:.0f}s")
    for tr in rA.trials:
        print(f"[CMP]     ratio={tr.prune_ratio:.2f} acc={tr.quality:.3f} drop={tr.quality_drop:.3f} {'OK' if tr.accepted else 'x'}")

    # (B) 考虑微调：每个候选剪枝后先恢复再评估
    t0 = time.time()
    rB = P.accuracy_based_auto_prune(copy.deepcopy(model), cfg, data=calib, tolerance=tol,
                                     quality_fn=acc_quality, ratio_grid=RATIOS,
                                     finetune_fn=do_finetune, apply=True)
    print(f"[CMP] (B) 微调后搜索: chosen={rB.chosen_ratio} applied={rB.applied} "
          f"acc_drop={rB.quality_drop}  {time.time()-t0:.0f}s")
    for tr in rB.trials:
        print(f"[CMP]     ratio={tr.prune_ratio:.2f} acc(recovered)={tr.quality:.3f} drop={tr.quality_drop:.3f} {'OK' if tr.accepted else 'x'}")

    rec = {"baseline_acc": acc0, "tolerance": tol,
           "no_finetune": {"chosen": rA.chosen_ratio, "trials": [(t.prune_ratio, t.quality, t.accepted) for t in rA.trials]},
           "with_finetune": {"chosen": rB.chosen_ratio, "trials": [(t.prune_ratio, t.quality, t.accepted) for t in rB.trials]}}
    os.makedirs(os.path.dirname(RESULT), exist_ok=True)
    json.dump(rec, open(RESULT, "w"), indent=2)
    print(f"[CMP] saved {RESULT}")
    print(f"[CMP] 结论: 无微调选 {rA.chosen_ratio}，微调后可选 {rB.chosen_ratio}（更激进且仍满足容差）")


if __name__ == "__main__":
    main()
