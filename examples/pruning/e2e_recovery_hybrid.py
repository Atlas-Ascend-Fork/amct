# -*- coding: UTF-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
"""真·E2E（带标签的合成分类任务）：复现 dense FFN 剪枝的"recovery 阶梯"结论。

要复现的论断
------------
对训练充分的 dense FFN，**免微调**剪枝的精度杠杆是剪枝后的 **RECOVERY（补偿）**，而非选神经元
的判据。在激进剪枝率、不微调时，recovery 阶梯为：

    none(朴素切片) 崩塌  <<  bias(均值折叠)  <  ls(最小二乘重构)

守卫式 recovery-HYBRID（``guarded_recovery_hybrid_prune``）用小验证集自动择优，**绝不劣于 none**。

为何要"训练过"的模型
--------------------
随机未训练模型上 FFN 输出饱和（各 recovery 几乎一致、阶梯不显现）。故本脚本：
 1) 用固定随机 teacher 造一个**非线性**的合成监督分类任务（tokens → C 类，需要 FFN 隐藏容量）；
 2) 搭一个 **非残差** 的 dense 模型（FFN 输出直接前传，使中间维成为信息瓶颈，
    dense 域识别 dense1→gelu→dense2 两层 FFN 并剪中间维、重构 dense2）；
 3) 训到高精度；
 4) 不微调，按 {0.5, 0.75} 激进剪枝，分别用 recovery none/bias/ls 测 **held-out test 精度**；
 5) 跑 ``guarded_recovery_hybrid_prune`` 证明它自动选最优 recovery 并报告相对 none 的增益。

确定性、纯 CPU、不联网、无 NPU。
运行：python examples/pruning/e2e_recovery_hybrid.py
"""

from __future__ import annotations

import copy
import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

import amct_pytorch.experimental.pruning as P

SEED = 0
HIDDEN = 64
INTER = 384
LAYERS = 1
VOCAB = 96
SEQ = 12
NUM_CLASSES = 10
RATIOS = (0.5, 0.75)

RESULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
JSON_PATH = os.path.join(RESULT_DIR, "recovery_hybrid.json")
MD_PATH = os.path.join(RESULT_DIR, "recovery_hybrid.md")


# --------------------------------------------------------------------------
# 合成监督任务：固定随机 teacher（深 + 非线性）给 token 序列打类别标签。
# teacher 故意有较大隐藏容量，使 student 必须用满 FFN 中间维才能拟合 → 剪中间维真的伤精度。
# --------------------------------------------------------------------------
class Teacher(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, HIDDEN)
        self.net = nn.Sequential(
            nn.Linear(HIDDEN, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, NUM_CLASSES),
        )

    @torch.no_grad()
    def forward(self, ids):
        h = self.embed(ids).mean(dim=1)  # 句级均值池化
        return self.net(h)


def make_task(n_train=4096, n_test=2048, n_calib=24, n_val=8, batch=64):
    """返回 (train_loader, test_set, calib_batches, val_batches, val_labels)。

    calib/val 为剪枝/择优用的"输入张量列表"（dense 域用 hook 收集 dense2 的输入激活）。
    """
    g = torch.Generator().manual_seed(SEED + 7)
    teacher = Teacher().eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    def sample(n):
        ids = torch.randint(0, VOCAB, (n, SEQ), generator=g)
        labels = teacher(ids).argmax(dim=-1)
        return ids, labels

    tr_ids, tr_y = sample(n_train)
    te_ids, te_y = sample(n_test)
    va_ids, va_y = sample(n_val * batch)  # 独立第三抽样：val 与 test 严格 disjoint（守卫择优不偷看 test）
    train_batches = [
        (tr_ids[i : i + batch], tr_y[i : i + batch]) for i in range(0, n_train, batch)
    ]
    # 校准/验证集：仅需输入（dense reconstruct 用前向激活，无需标签）；
    # 但 hybrid 的 quality_fn 用 val 精度，故另存 val 标签。
    calib = [tr_ids[i : i + batch][:batch] for i in range(0, n_calib * batch, batch)]
    val_ids = va_ids.reshape(n_val, batch, SEQ)
    val_y = va_y.reshape(n_val, batch)
    val_batches = [val_ids[i] for i in range(n_val)]
    val_labels = [val_y[i] for i in range(n_val)]
    return train_batches, (te_ids, te_y), calib, val_batches, val_labels


# --------------------------------------------------------------------------
# 非残差 dense student：FFN 输出直接前传，中间维=信息瓶颈（剪它真的掉精度）。
# 每个 block 的 fc1/fc2 是同一 ModuleDict 的兄弟 Linear → dense 域识别为两层 FFN。
# --------------------------------------------------------------------------
class DenseStudent(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, HIDDEN)
        self.blocks = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "fc1": nn.Linear(HIDDEN, INTER),
                        "fc2": nn.Linear(INTER, HIDDEN),
                    }
                )
                for _ in range(LAYERS)
            ]
        )
        self.head = nn.Linear(HIDDEN, NUM_CLASSES)

    def forward(self, ids):
        h = self.embed(ids)
        for b in self.blocks:
            # 非残差、无 post-norm：FFN 输出直接替换隐藏态 → 中间维是唯一信息瓶颈。
            # 朴素切片(none)会同时丢掉被剪神经元的 **均值贡献**（无 norm 重置 → 直接掉精度）。
            h = b["fc2"](F.gelu(b["fc1"](h)))
        return self.head(h.mean(dim=1))


@torch.no_grad()
def accuracy(model, ids, labels, batch=512):
    model.eval()
    ok = tot = 0
    for i in range(0, ids.shape[0], batch):
        pred = model(ids[i : i + batch]).argmax(dim=-1)
        ok += int((pred == labels[i : i + batch]).sum())
        tot += int(labels[i : i + batch].numel())
    return ok / tot


def train(model, train_batches, epochs=12, lr=2e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for _ in range(epochs):
        for ids, y in train_batches:
            opt.zero_grad()
            F.cross_entropy(model(ids), y).backward()
            opt.step()
    model.eval()


def val_accuracy_fn(val_batches, val_labels):
    """返回 quality_fn(model)->float：val 分类精度（越大越好），传给 hybrid 守卫择优。"""

    @torch.no_grad()
    def quality(model):
        model.eval()
        ok = tot = 0
        for ids, y in zip(val_batches, val_labels):
            pred = model(ids).argmax(dim=-1)
            ok += int((pred == y).sum())
            tot += int(y.numel())
        return ok / max(tot, 1)

    return quality


def prune_recovery(model, ratio, recovery, calib):
    """在副本上按单一 recovery 剪枝（dense reconstruct），返回剪枝后的模型。"""
    m = copy.deepcopy(model)
    P.prune(
        m,
        {
            "methods": {
                "dense": {
                    "name": "reconstruct",
                    "kwargs": {"prune_ratio": ratio, "recovery": recovery},
                }
            },
            "missing_data_policy": "warn_skip",
        },
        data=calib,
    )
    return m


def count_params(m):
    return sum(p.numel() for p in m.parameters())


def run_ratio(model, ratio, base_acc, base_params, te, calib, val_batches, val_labels):
    te_ids, te_y = te
    quality_fn = val_accuracy_fn(val_batches, val_labels)
    row = {"ratio": ratio, "base_acc": round(base_acc, 4)}

    # 1) 三种 recovery 各剪一份，测 held-out test 精度（阶梯应为 none << bias < ls）。
    for rec in ("none", "bias", "ls"):
        pruned = prune_recovery(model, ratio, rec, calib)
        acc = accuracy(pruned, te_ids, te_y)
        row[rec] = round(acc, 4)
        row[f"{rec}_params"] = count_params(pruned)

    # 2) 守卫式 hybrid：自动择优 + 报告相对 none 的增益（用自定义 val 精度作 quality_fn）。
    hyb_model = copy.deepcopy(model)
    res = P.guarded_recovery_hybrid_prune(
        hyb_model,
        data=calib,
        prune_ratio=ratio,
        domain="dense",
        eval_data=val_batches,
        quality_fn=quality_fn,
        apply=True,
    )
    hybrid_test_acc = accuracy(hyb_model, te_ids, te_y)
    weight_reduction = 1.0 - count_params(hyb_model) / base_params
    row["hybrid_pick"] = res.chosen_recovery
    row["gain_over_fallback"] = round(res.gain_over_fallback, 4)
    row["hybrid_test_acc"] = round(hybrid_test_acc, 4)
    row["weight_reduction"] = round(weight_reduction, 4)
    row["recovery_quality"] = {k: round(v, 4) for k, v in res.recovery_quality.items()}

    print(
        f"[recovery-E2E] ratio={ratio:.2f}  base={base_acc:.4f}  "
        f"none={row['none']:.4f}  bias={row['bias']:.4f}  ls={row['ls']:.4f}  "
        f"| hybrid_pick={res.chosen_recovery} gain={res.gain_over_fallback:+.4f} "
        f"test={hybrid_test_acc:.4f}  weight-{weight_reduction * 100:.1f}%"
    )
    return row


def write_md(rows, base_acc, base_params, chance):
    lines = [
        "# Dense FFN 剪枝 recovery 阶梯 — 守卫式 recovery-HYBRID (合成分类, 免微调, 纯CPU)",
        "",
        f"训练后 base test acc={base_acc:.4f}（{NUM_CLASSES} 类, chance≈{chance:.3f}, "
        f"params={base_params:,}）；激进剪 FFN 中间维、**不微调**，对比三种 recovery 的 "
        f"held-out test 精度。阶梯 none(崩塌) << bias < ls；hybrid 用 val 精度自动择优、绝不劣于 none。",
        "",
        "| ratio | base | none | bias | ls | hybrid_pick | gain(val) | hybrid_test | weight↓% |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['ratio']:.2f} | {r['base_acc']:.4f} | {r['none']:.4f} | "
            f"{r['bias']:.4f} | {r['ls']:.4f} | {r['hybrid_pick']} | "
            f"{r['gain_over_fallback']:+.4f} | {r['hybrid_test_acc']:.4f} | "
            f"{r['weight_reduction'] * 100:.1f}% |"
        )
    lines += ["", "复现: python examples/pruning/e2e_recovery_hybrid.py", ""]
    with open(MD_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    torch.manual_seed(SEED)
    train_batches, te, calib, val_batches, val_labels = make_task()
    chance = 1.0 / NUM_CLASSES

    torch.manual_seed(SEED)  # 模型初值确定性
    model = DenseStudent()
    train(model, train_batches)
    base_params = count_params(model)
    base_acc = accuracy(model, te[0], te[1])
    print(
        f"[recovery-E2E] 训练完成: base test acc={base_acc:.4f} "
        f"(chance={chance:.3f}, params={base_params:,}, inter={INTER})"
    )

    rows = [
        run_ratio(model, r, base_acc, base_params, te, calib, val_batches, val_labels)
        for r in RATIOS
    ]

    os.makedirs(RESULT_DIR, exist_ok=True)
    payload = {
        "task": "synthetic_token_classification",
        "num_classes": NUM_CLASSES,
        "chance": round(chance, 4),
        "base_test_acc": round(base_acc, 4),
        "base_params": base_params,
        "hidden": HIDDEN,
        "intermediate": INTER,
        "layers": LAYERS,
        "seed": SEED,
        "ratios": list(RATIOS),
        "rows": rows,
    }
    with open(JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    write_md(rows, base_acc, base_params, chance)
    print(f"[recovery-E2E] 已保存: {JSON_PATH}")
    print(f"[recovery-E2E] 已保存: {MD_PATH}")

    # 阶梯断言（守卫不变量 + 已知阶梯方向）：hybrid 绝不劣于 none。
    for r in rows:
        assert r["gain_over_fallback"] >= 0.0, r
        assert r["ls"] >= r["none"] - 1e-6, ("ls should not be below none", r)


if __name__ == "__main__":
    main()
