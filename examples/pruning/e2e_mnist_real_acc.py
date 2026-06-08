# -*- coding: UTF-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
"""真·E2E（带标签的真实任务精度）：在 MNIST 上训练一个小 CNN，然后
   训练精度 → 剪枝(掉精度) → 微调(恢复) → 量化(NPU) 全程用**真实 test accuracy** 衡量。

这回答了 "是否做了 acc 级别的真实决策 E2E"：模型真训练、标签真有、精度真测。

运行：ASCEND_RT_VISIBLE_DEVICES=3 python examples/pruning/e2e_mnist_real_acc.py
"""
import json, os, time
import torch, torch.nn as nn, torch.nn.functional as F
import torchvision, torchvision.transforms as T

import amct_pytorch.experimental.pruning as P

DATA = os.environ.get("MNIST_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "mnist_data"))
RESULT = os.path.join(os.environ.get("AMCT_PRUNE_RESULT_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")), "e2e_mnist_real_acc.json")
try:
    import torch_npu  # noqa: F401
    HAS_NPU = (torch.npu.is_available() is True)
except Exception:
    HAS_NPU = False


class SmallCNN(nn.Module):
    def __init__(self, n=10):
        super().__init__()
        self.c1 = nn.Conv2d(1, 16, 3, padding=1); self.c2 = nn.Conv2d(16, 32, 3, padding=1)
        self.c3 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2); self.gap = nn.AdaptiveAvgPool2d(1); self.fc = nn.Linear(64, n)

    def forward(self, x):
        x = self.pool(F.relu(self.c1(x))); x = self.pool(F.relu(self.c2(x)))
        x = F.relu(self.c3(x)); x = self.gap(x).flatten(1)
        return self.fc(x)


def loaders():
    tf = T.Compose([T.ToTensor(), T.Normalize((0.1307,), (0.3081,))])
    tr = torchvision.datasets.MNIST(DATA, train=True, download=True, transform=tf)
    te = torchvision.datasets.MNIST(DATA, train=False, download=True, transform=tf)
    tr = torch.utils.data.Subset(tr, range(12000))   # 子集，CPU 训练快
    trl = torch.utils.data.DataLoader(tr, batch_size=128, shuffle=True)
    tel = torch.utils.data.DataLoader(te, batch_size=512)
    return trl, tel


@torch.no_grad()
def accuracy(model, loader, dev="cpu"):
    model.eval(); ok = tot = 0
    for x, y in loader:
        x, y = x.to(dev), y.to(dev)
        if x.dtype != next(model.parameters()).dtype:
            x = x.to(next(model.parameters()).dtype)
        p = model(x).float().argmax(-1).cpu()
        ok += (p == y.cpu()).sum().item(); tot += y.numel()
    return ok / tot


def train(model, trl, epochs=3, lr=1e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr); model.train()
    for _ in range(epochs):
        for x, y in trl:
            opt.zero_grad(); F.cross_entropy(model(x), y).backward(); opt.step()
    model.eval()


def main():
    torch.manual_seed(0)
    trl, tel = loaders()
    model = SmallCNN()
    t0 = time.time(); train(model, trl, epochs=3); print(f"[MNIST-E2E] 训练完成 {time.time()-t0:.1f}s")
    base_params = sum(p.numel() for p in model.parameters())
    acc0 = accuracy(model, tel)
    print(f"[MNIST-E2E] baseline test acc = {acc0:.4f}  params={base_params:,}")
    rec = {"baseline_acc": acc0, "baseline_params": base_params, "stages": []}

    import copy
    # calib（依赖图剪枝的样例输入 + 微调数据）
    calib = [x for x, _ in [next(iter(trl))]]
    ft_data = [(x, y) for i, (x, y) in zip(range(20), trl)]   # 微调用带标签批

    # 1) 剪枝（依赖图, taylor）40% 通道
    m = copy.deepcopy(model)
    P.prune(m, {"methods": {"cnn": {"name": "depgraph",
              "kwargs": {"prune_ratio": 0.4, "importance": "taylor", "round_to": 8}}}}, data=calib)
    acc_p = accuracy(m, tel); pp = sum(p.numel() for p in m.parameters())
    print(f"[MNIST-E2E] 剪枝(taylor40%+round8): acc {acc0:.4f}→{acc_p:.4f} (Δ={acc_p-acc0:+.4f}) params={pp:,}({100*(1-pp/base_params):.0f}%↓)")
    rec["stages"].append({"stage": "prune_taylor0.4", "acc": acc_p, "params": pp})

    # 2) 微调恢复（P4: optional finetune）
    ft = P.finetune(m, ft_data, loss_fn=lambda mm, b: F.cross_entropy(mm(b[0]), b[1]), steps=300, lr=5e-4)
    acc_ft = accuracy(m, tel)
    print(f"[MNIST-E2E] 微调300步: acc {acc_p:.4f}→{acc_ft:.4f} (恢复 {acc_ft-acc_p:+.4f}; final_loss={ft['final_loss']:.4f})")
    rec["stages"].append({"stage": "finetune300", "acc": acc_ft, "recovered": acc_ft - acc_p})

    # 3) 量化（NPU3, fp16），真实 acc
    if HAS_NPU:
        dev = "npu:0"
        import amct_pytorch as amct
        mq = copy.deepcopy(m).half().to(dev)
        amct.quantize(mq); amct.convert(mq)
        qmods = sorted({type(z).__name__ for z in mq.modules() if "Quant" in type(z).__name__})
        acc_q = accuracy(mq, tel, dev=dev)
        print(f"[MNIST-E2E] 量化(INT8 weight,NPU3): acc {acc_ft:.4f}→{acc_q:.4f} (Δ={acc_q-acc_ft:+.4f}) quant_modules={qmods}")
        rec["stages"].append({"stage": "quantize_npu", "acc": acc_q, "quant_modules": qmods})
    else:
        print("[MNIST-E2E] 无 NPU → 跳过量化阶段")
        rec["stages"].append({"stage": "quantize_npu", "skipped": "no torch_npu/NPU"})

    os.makedirs(os.path.dirname(RESULT), exist_ok=True)
    json.dump(rec, open(RESULT, "w"), indent=2, ensure_ascii=False)
    print(f"[MNIST-E2E] 结果已保存: {RESULT}")


if __name__ == "__main__":
    main()
