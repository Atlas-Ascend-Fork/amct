# -*- coding: UTF-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
"""结论 3 / 结论 4(dense) 的**可复现脚本**：1% 质量预算下 dense-FFN 剪枝的"恢复菜单"对比。

对每个模型、每种方法，按**结论 3 同款**协议二分搜索"在 1% 相对 held-ppl 增加内可达的最大总参数削减"：
 - 门控指标 == 上报指标：``quality = base_ppl / held_ppl``（drop = 相对 ppl 增加），容差 1%；
 - held 与 calib/finetune 严格 disjoint（train[:680] 单遍微调，held[700:714]，calib=train[:24]）；
 - noFT 与 +1-epoch FT 两种模式都测。

菜单 = ``low_variance``(基线，仅选神经元不补偿) / ``reconstruct`` 的 recovery ∈ {none, bias, ls}。
**HYBRID** = 守卫在 val 上从 {none,bias,ls} 自动择优、绝不劣于 none（预算意义下即取菜单中最大可达削减）。

要点：训练充分的预训练 LM，*免微调*几乎无空间(≈0)；+1-epoch 微调后 recovery=ls ≈ low_variance 的 2–3 倍，
且在 Conv1D-FFN(distilgpt2) 上 low_variance 无目标(0%) 而 reconstruct 仍可剪——这是恢复菜单的覆盖面优势。

运行（默认 4 模型面板，复现 README 表；CPU 可跑但慢，NPU 快）：
  HF_HUB_DISABLE_XET=1 python examples/pruning/e2e_dense_budget_menu.py
  # 子集 / 自定义：--only SmolLM-135M,distilgpt2   --quick(更少 FT 步，仅冒烟)
结果 json 写入 examples/pruning/results/（已 gitignore，不入库）；表格直接打印到 stdout。
预训练权重不可达时退化为随机权重(仅演示流程，数值无意义)。
"""
import argparse
import json
import math
import os
import time

import torch
import torch.nn.functional as F

try:
    import torch_npu  # noqa: F401  (registers the NPU backend when present)
except Exception:
    torch_npu = None

import amct_pytorch.experimental.pruning as P
from amct_pytorch.experimental.pruning import accuracy_based_auto_prune

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
HERE = os.path.dirname(os.path.abspath(__file__))
DEV = "npu:0" if (hasattr(torch, "npu") and torch.npu.is_available()) else "cpu"
RATIOS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
TOL = 0.01  # 1% 质量预算（相对 held-ppl 增加 <= 1%）

# 与结论 3 同源的小型预训练 LM 面板（含 distilgpt2 = Conv1D-FFN，low_variance 无目标的覆盖面反例）
PANEL = [
    ("pythia-160m", "EleutherAI/pythia-160m"),
    ("gpt-neo-125M", "EleutherAI/gpt-neo-125M"),
    ("SmolLM-135M", "HuggingFaceTB/SmolLM-135M"),
    ("distilgpt2", "distilgpt2"),
]
METHODS = [
    ("low_variance", {"name": "low_variance", "kwargs": {}}),
    ("recon_none", {"name": "reconstruct", "kwargs": {"recovery": "none"}}),
    ("recon_bias", {"name": "reconstruct", "kwargs": {"recovery": "bias"}}),
    ("recon_ls", {"name": "reconstruct", "kwargs": {"recovery": "ls"}}),
]


def _logits(out):
    if torch.is_tensor(out):
        return out
    return out.logits if hasattr(out, "logits") else out[0]


def _wikitext_batches(tok, seq=128, bs=8, n=720):
    """WikiText-2 → [n] 个 [bs, seq] token 批（不足时回退随机，仅演示流程）。"""
    try:
        from huggingface_hub import hf_hub_download
        import pyarrow.parquet as pq

        path = hf_hub_download(
            repo_id="Salesforce/wikitext", repo_type="dataset",
            filename="wikitext-2-raw-v1/train-00000-of-00001.parquet",
        )
        txt = " ".join(t for t in pq.read_table(path).column("text").to_pylist() if t and t.strip())
        ids = tok(txt, return_tensors="pt").input_ids[0]
        usable = (ids.numel() // (bs * seq)) * (bs * seq)
        chunks = ids[:usable].view(-1, bs, seq)
        return [chunks[i].to(DEV) for i in range(min(n, chunks.shape[0]))]
    except Exception as exc:
        print(f"[budget] WikiText 不可达({type(exc).__name__})，回退随机 token（仅演示流程）", flush=True)
        return [torch.randint(0, tok.vocab_size, (bs, seq)).to(DEV) for _ in range(n)]


def _ppl(model, held):
    model.eval()
    tot_loss = tot_tok = 0
    with torch.no_grad():
        for b in held:
            lg = _logits(model(b))
            tot_loss += float(F.cross_entropy(
                lg[:, :-1].reshape(-1, lg.shape[-1]), b[:, 1:].reshape(-1), reduction="sum"))
            tot_tok += b[:, 1:].numel()
    return math.exp(tot_loss / max(tot_tok, 1))


def _ce_loss(m, b):
    lg = _logits(m(b))
    return F.cross_entropy(lg[:, :-1].reshape(-1, lg.shape[-1]), b[:, 1:].reshape(-1))


def run_model(mid, repo, ft_steps, cells, save):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(repo)
    batches = _wikitext_batches(tok)
    train, held = batches[:680], batches[700:714]  # 单遍微调 vs held，严格 disjoint（结论 3 同款）
    recon = train[:24]  # ≈24k token > FFN intermediate
    base = AutoModelForCausalLM.from_pretrained(repo, torch_dtype=torch.float32).to(DEV).eval()
    base_ppl = _ppl(base, held)
    print(f"[{mid}] baseline held-ppl = {base_ppl:.3f}", flush=True)

    def quality(m):  # 门控 == 上报：相对 held-ppl 增加
        return base_ppl / _ppl(m, held)

    def do_ft(m):
        P.finetune(m, train[:ft_steps], loss_fn=_ce_loss, steps=ft_steps, lr=1e-5, warmup=20, grad_clip=1.0)

    for mname, mspec in METHODS:
        for ft_mode, ftfn in (("noFT", None), ("FT", do_ft)):
            key = f"{mid}|{mname}|{ft_mode}"
            if key in cells:
                continue
            cfg = {"methods": {"dense": mspec},
                   "skip_layers": ["self_attn", "attn", "attention"],
                   "missing_data_policy": "warn_skip"}
            t0 = time.time()
            try:
                res = accuracy_based_auto_prune(
                    base, cfg, data=recon, tolerance=TOL, quality_fn=quality,
                    ratio_grid=RATIOS, finetune_fn=ftfn, apply=False)
                cell = {"weight_reduction": round(res.weight_reduction, 4),
                        "chosen_ratio": res.chosen_ratio,
                        "ppl_loss": round(res.quality_drop, 4) if res.quality_drop is not None else None,
                        "secs": round(time.time() - t0, 1)}
            except Exception as exc:
                cell = {"error": f"{type(exc).__name__}: {exc}"[:120]}
            cells[key] = cell
            print(f"  [{key}] cut={cell.get('weight_reduction')} ppl_loss={cell.get('ppl_loss')} "
                  f"({cell.get('secs')}s) {cell.get('error', '')}", flush=True)
            save()
    del base
    if DEV.startswith("npu"):
        torch.npu.empty_cache()


def _print_table(cells, panel):
    """打印 README 同款 markdown 表：每模型 noFT/FT × {low_variance,none,bias,ls,HYBRID=守卫择优}。"""
    order = ["low_variance", "recon_none", "recon_bias", "recon_ls"]
    print("\n| model | mode | low_variance | none | bias | ls | HYBRID(guard) |")
    print("|---|---|---|---|---|---|---|")
    for mid, _ in panel:
        for mode in ("noFT", "FT"):
            vals = {m: cells.get(f"{mid}|{m}|{mode}", {}).get("weight_reduction") for m in order}
            menu = [vals[m] for m in ("recon_none", "recon_bias", "recon_ls") if vals[m] is not None]
            hybrid = max(menu) if menu else None  # 守卫从 {none,bias,ls} 择优，绝不劣于 none
            fmt = lambda v: "-" if v is None else f"{v:.3f}"
            print(f"| {mid} | {mode} | {fmt(vals['low_variance'])} | {fmt(vals['recon_none'])} | "
                  f"{fmt(vals['recon_bias'])} | {fmt(vals['recon_ls'])} | {fmt(hybrid)} |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="逗号分隔的模型子集（默认全 4 个）")
    ap.add_argument("--out", default="dense_budget_menu.json")
    ap.add_argument("--quick", action="store_true", help="少量 FT 步，仅冒烟（数值不代表 README 表）")
    args = ap.parse_args()
    ft_steps = 40 if args.quick else 680  # 默认 680 = 结论 3 单遍；--quick 仅验证流程
    outdir = os.path.join(HERE, "results")
    os.makedirs(outdir, exist_ok=True)
    out = os.path.join(outdir, args.out)
    cells = json.load(open(out)).get("cells", {}) if os.path.exists(out) else {}

    def save():
        json.dump({"cells": cells}, open(out, "w"), indent=1)

    only = [s.strip() for s in args.only.split(",") if s.strip()]
    panel = [(m, r) for m, r in PANEL if (not only or m in only)]
    print(f"=== dense budget menu (1% rel-ppl) {[m for m, _ in panel]} | DEV={DEV} "
          f"| ft_steps={ft_steps}{' QUICK' if args.quick else ''} ===", flush=True)
    for mid, repo in panel:
        try:
            run_model(mid, repo, ft_steps, cells, save)
        except Exception as exc:
            print(f"!! model {mid} FAILED: {type(exc).__name__}: {exc}", flush=True)
        save()
    _print_table(cells, panel)
    print(f"\n[DONE] {out}", flush=True)


if __name__ == "__main__":
    main()
