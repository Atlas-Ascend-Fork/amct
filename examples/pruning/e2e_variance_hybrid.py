# -*- coding: UTF-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
"""样例 13：守卫式 variance-HYBRID 的 MoE 专家剪枝 —— **可复现的 E2E 实测**。

复现包 README 的核心论断（在合成不均衡 mini-suite 上 variance 判据 +~6pp）：

  在**专精路由 + 专家使用不均衡 + 尾敏感指标（balanced accuracy）** 的 regime 上，
  守卫式 variance-HYBRID **击败** mass(activation_count) fallback；
  在**均衡 regime** 上守卫回退到 mass —— 绝不更差（gain≈0）。

测试床：一个可控的 MoE **分类器**（每类有专属 token 子词表 → 真实专精路由），
带类不均衡旋钮（imb=20 时后半类 20× 更稀有）。指标是 balanced accuracy（每类召回均值，
压尾类）。这正是 prune_bench/moe_cls 已确证的 regime（imb20/ratio0.5：
mass~0.85 vs variance~0.97，+~11pp）。本脚本在 repo 内、CPU、确定性、无下载地复现其裁剪版。

对每个 (imbalance, ratio)：
 (a) MASS fallback —— prune(copy, activation_count) → TEST balanced acc；
 (b) 守卫 HYBRID —— guarded_variance_hybrid_prune(自定义 quality_fn=VAL balanced acc) → TEST。
守卫只用 VAL 择优，故 HYBRID 在 VAL 上绝不劣于 mass；TEST 上展示真实增益。

运行：python examples/pruning/e2e_variance_hybrid.py
"""

from __future__ import annotations

import copy
import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from amct_pytorch.experimental.pruning import guarded_variance_hybrid_prune, prune

# --- 测试床常量（裁剪自 prune_bench/moe_cls，小到 CPU<3min 可跑） ------------------
V, C, SEQ = 160, 16, 16  # 词表 / 类数 / 序列长（与 moe_cls 一致）
SIG = 6  # 每类专属 token 数；前 C*SIG 个 token 为"signature"，其余为 filler
FILLER = V - C * SIG
DIM, LAYERS, E, TOPK = 48, 3, 16, 2  # 隐藏维 / 层数 / 每层专家数 / top-k
SIG_FRAC = 0.4  # 序列中 signature token 占比（专精强度）
N_TRAIN, N_TEST, N_VAL = 4000, 1500, 1000
EPOCHS, BS, LR = 9, 128, 2e-3
RATIO = 0.5
IMBALANCES = (1, 20)  # imb=1 均衡（守卫回退）；imb=20 尾重（variance 取胜）

_HERE = os.path.dirname(os.path.abspath(__file__))
_OUT_DIR = os.path.join(_HERE, "results")
_JSON = os.path.join(_OUT_DIR, "variance_hybrid.json")
_MD = os.path.join(_OUT_DIR, "variance_hybrid.md")


# --- 合成数据：类专属 signature 子词表 → 真实专家专精；imbalance 控制尾类稀有度 --------
def gen_data(n, imbalance, seed):
    g = torch.Generator().manual_seed(seed)
    freq = torch.ones(C)
    freq[C // 2 :] = 1.0 / imbalance  # 后半类更稀有
    freq = freq / freq.sum()
    y = torch.multinomial(freq, n, replacement=True, generator=g)
    x = torch.empty(n, SEQ, dtype=torch.long)
    filler_lo = C * SIG
    for i in range(n):
        c = int(y[i])
        is_sig = torch.rand(SEQ, generator=g) < SIG_FRAC
        sig_tok = c * SIG + torch.randint(0, SIG, (SEQ,), generator=g)
        fil_tok = filler_lo + torch.randint(0, max(1, FILLER), (SEQ,), generator=g)
        x[i] = torch.where(is_sig, sig_tok, fil_tok)
    return x, y


# --- MoE 分类器：每层 .gate(nn.Linear)+.experts(nn.ModuleList)，moe 域可直接检出 -----
# 结构镜像 tests/amct_pytorch/mini_models.py 的 MiniMoELayer（gate+experts+num_experts/topk）。
class Expert(nn.Module):
    def __init__(self, hidden, inter):
        super().__init__()
        self.fc1 = nn.Linear(hidden, inter)
        self.fc2 = nn.Linear(inter, hidden)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class MoELayer(nn.Module):
    def __init__(self, hidden, num_experts, topk):
        super().__init__()
        self.num_experts, self.topk = num_experts, topk
        self.gate = nn.Linear(hidden, num_experts, bias=False)
        self.experts = nn.ModuleList(
            [Expert(hidden, hidden * 2) for _ in range(num_experts)]
        )

    def forward(self, h):
        b, s, d = h.shape
        flat = h.reshape(-1, d)
        probs = F.softmax(self.gate(flat), dim=-1)
        w, idx = torch.topk(probs, self.topk, dim=-1)
        w = w / w.sum(-1, keepdim=True)
        out = torch.zeros_like(flat)
        for k in range(self.topk):
            sel = idx[:, k]
            for e in range(self.num_experts):
                m = sel == e
                if m.any():
                    out[m] += w[m, k, None] * self.experts[e](flat[m])
        return out.reshape(b, s, d)


class MoEClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok = nn.Embedding(V, DIM)
        self.pos = nn.Embedding(SEQ, DIM)
        self.layers = nn.ModuleList([MoELayer(DIM, E, TOPK) for _ in range(LAYERS)])
        self.norm = nn.LayerNorm(DIM)
        self.head = nn.Linear(DIM, C)

    def forward(self, idx):
        t = idx.shape[1]
        h = self.tok(idx) + self.pos(torch.arange(t, device=idx.device))[None]
        for layer in self.layers:
            h = h + layer(h)
        return self.head(self.norm(h).mean(1))


def train(model, x_tr, y_tr):
    model = model.float().train()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    n = x_tr.shape[0]
    for _ in range(EPOCHS):
        perm = torch.randperm(n)
        for i in range(0, n, BS):
            idx = perm[i : i + BS]
            opt.zero_grad()
            F.cross_entropy(model(x_tr[idx]), y_tr[idx]).backward()
            opt.step()
    return model.eval()


@torch.no_grad()
def _predict(model, x, bs=256):
    model = model.eval()
    preds = [model(x[i : i + bs]).argmax(-1) for i in range(0, x.shape[0], bs)]
    return torch.cat(preds)


@torch.no_grad()
def balanced_acc(model, x, y):
    """每类召回均值（尾敏感）。mass 丢稀有类专家 → 该指标暴跌；variance 保住。"""
    p = _predict(model, x)
    recalls = [
        float((p[y == c] == c).float().mean()) for c in range(C) if (y == c).any()
    ]
    return sum(recalls) / len(recalls)


def _num_experts(model):
    for layer in model.layers:
        return layer.num_experts
    return None


def mass_prune_test(model, calib, x_te, y_te, ratio):
    """纯 MASS 基线：deepcopy → activation_count 剪枝 → TEST balanced acc。
    这与 HYBRID 的 fallback 判据完全一致。"""
    mc = copy.deepcopy(model).float()
    cfg = {
        "methods": {
            # 与守卫 HYBRID 的 _candidate_config 一致：cnn/dense 置 0，只剪 MoE → 公平同域对比
            # （否则默认会连专家内部 dense FFN 一起剪，basline 双重受损、虚高 hybrid 增益）。
            "cnn": {"name": "variance_channel", "kwargs": {"prune_ratio": 0.0}},
            "dense": {"name": "low_variance", "kwargs": {"prune_ratio": 0.0}},
            "moe": {
                "name": "activation_count",
                "kwargs": {"prune_ratio": ratio, "top_k": TOPK},
            },
        },
        "missing_data_policy": "warn_skip",
    }
    prune(mc, cfg, data=calib)
    return balanced_acc(mc, x_te, y_te), _num_experts(mc)


def run_point(imbalance):
    print(f"\n=== imbalance={imbalance} (ratio={RATIO}) ===", flush=True)
    x_tr, y_tr = gen_data(N_TRAIN, imbalance, seed=0)
    x_te, y_te = gen_data(N_TEST, imbalance, seed=1)
    x_va, y_va = gen_data(N_VAL, imbalance, seed=2)
    calib = [x_tr[:512]]  # 校准批（list of model.forward 输入）
    val = [x_va[i : i + 256] for i in range(0, N_VAL, 256)]

    torch.manual_seed(0)
    model = train(MoEClassifier(), x_tr, y_tr)
    base = balanced_acc(model, x_te, y_te)
    n_exp_before = _num_experts(model)
    print(
        f"  base balanced-acc(TEST)={base:.4f}  experts/layer={n_exp_before}",
        flush=True,
    )

    # (a) MASS fallback
    mass_test, n_exp_after = mass_prune_test(model, calib, x_te, y_te, RATIO)

    # (a') 单判据 TEST（无菜单择优）：把"判据增益"(mass→单判据) 与"菜单选择增益"(单判据→hybrid)分开，
    # 否则 hybrid−mass 会把 argmax-over-menu 的选择红利误计入判据本身。
    def _single_criterion_test(vs):
        mc = copy.deepcopy(model).float()
        prune(
            mc,
            {"methods": {
                "cnn": {"name": "variance_channel", "kwargs": {"prune_ratio": 0.0}},
                "dense": {"name": "low_variance", "kwargs": {"prune_ratio": 0.0}},
                "moe": {"name": "mass_variance", "kwargs": {
                    "prune_ratio": RATIO, "top_k": TOPK, "boundary": -1, "variance_score": vs}}},
             "missing_data_policy": "warn_skip"},
            data=calib,
        )
        return round(balanced_acc(mc, x_te, y_te), 4)

    per_criterion_test = {vs: _single_criterion_test(vs) for vs in ("cond", "peak", "cvxpeak")}
    best_single = max(per_criterion_test.values())

    # (b) 守卫 HYBRID：自定义 quality_fn = VAL balanced acc（尾敏感，默认保真度不敏感）。
    def quality_fn(m):
        return balanced_acc(m, x_va, y_va)

    hyb_model = copy.deepcopy(model).float()
    res = guarded_variance_hybrid_prune(
        hyb_model,
        data=calib,
        prune_ratio=RATIO,
        top_k=TOPK,
        eval_data=val,
        quality_fn=quality_fn,
        apply=True,
    )
    hybrid_test = balanced_acc(hyb_model, x_te, y_te)
    gain = hybrid_test - mass_test

    print(
        f"  mass(fallback) TEST={mass_test:.4f} | "
        f"hybrid TEST={hybrid_test:.4f} | pick={res.chosen_criterion} | "
        f"gain(hybrid-mass)={gain:+.4f}",
        flush=True,
    )
    print(
        "  VAL balanced-acc the guard saw: "
        + ", ".join(f"{k}={v:.4f}" for k, v in res.trials),
        flush=True,
    )
    print(
        "  per-criterion TEST(no menu): "
        + ", ".join(f"{k}={v:.4f}" for k, v in per_criterion_test.items())
        + f"  → 判据增益(best single−mass)={best_single - mass_test:+.4f},"
        f" 选择增益(hybrid−best single)={hybrid_test - best_single:+.4f}",
        flush=True,
    )
    # 守卫不变量：VAL 上 HYBRID 绝不劣于 fallback。
    assert res.gain_over_fallback >= 0.0

    return {
        "imbalance": imbalance,
        "ratio": RATIO,
        "base_balanced_acc": round(base, 4),
        "mass_test_acc": round(mass_test, 4),
        "hybrid_test_acc": round(hybrid_test, 4),
        "hybrid_pick": res.chosen_criterion,
        "fallback": res.fallback_criterion,
        "gain": round(gain, 4),
        "per_criterion_test_acc": per_criterion_test,
        "criterion_gain": round(best_single - mass_test, 4),
        "selection_gain": round(hybrid_test - best_single, 4),
        "val_guard_gain": round(res.gain_over_fallback, 4),
        "val_quality_per_criterion": {
            k: round(v, 4) for k, v in res.criterion_quality.items()
        },
        "experts_per_layer_before": n_exp_before,
        "experts_per_layer_after": n_exp_after,
        "weight_reduction": round(res.weight_reduction, 4),
    }


def write_md(points):
    lines = [
        "# Guarded variance-HYBRID MoE expert pruning — E2E reproducible",
        "",
        "Synthetic specialized-routing MoE classifier (V=160, C=16, top-k=2, ratio=0.5); "
        "metric = balanced accuracy (mean per-class recall, tail-sensitive). "
        "Imbalanced regime → variance pick beats mass; balanced regime → guard falls back (ties).",
        "",
        "| imbalance | base | mass (fallback) | hybrid | pick | gain |",
        "|---|---|---|---|---|---|",
    ]
    for p in points:
        lines.append(
            f"| {p['imbalance']} | {p['base_balanced_acc']:.4f} | "
            f"{p['mass_test_acc']:.4f} | {p['hybrid_test_acc']:.4f} | "
            f"{p['hybrid_pick']} | {p['gain']:+.4f} |"
        )
    lines += [
        "",
        "balanced accuracy on a held-out TEST split; `pick` is the criterion the guard chose "
        "on a separate VAL split (fallback = mass = activation_count).",
        "",
        "复现: python examples/pruning/e2e_variance_hybrid.py",
        "",
    ]
    with open(_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    os.makedirs(_OUT_DIR, exist_ok=True)
    points = [run_point(imb) for imb in IMBALANCES]

    payload = {
        "task": "synthetic specialized-routing MoE classifier, balanced-accuracy metric",
        "config": {
            "vocab": V,
            "num_classes": C,
            "seq": SEQ,
            "dim": DIM,
            "layers": LAYERS,
            "experts_per_layer": E,
            "top_k": TOPK,
            "sig_frac": SIG_FRAC,
            "ratio": RATIO,
            "imbalances": list(IMBALANCES),
            "n_train": N_TRAIN,
            "n_test": N_TEST,
            "n_val": N_VAL,
            "epochs": EPOCHS,
        },
        "points": points,
    }
    with open(_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    write_md(points)

    print("\n=== HEADLINE (TEST balanced accuracy) ===")
    print(
        f"{'imbalance':>9} | {'base':>7} | {'mass':>7} | {'hybrid':>7} | {'pick':>9} | {'gain':>8}"
    )
    for p in points:
        print(
            f"{p['imbalance']:>9} | {p['base_balanced_acc']:>7.4f} | "
            f"{p['mass_test_acc']:>7.4f} | {p['hybrid_test_acc']:>7.4f} | "
            f"{p['hybrid_pick']:>9} | {p['gain']:>+8.4f}"
        )
    print(f"\nwrote: {_JSON}\nwrote: {_MD}")


if __name__ == "__main__":
    main()
