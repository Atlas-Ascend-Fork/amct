# -*- coding: UTF-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
"""模型支持矩阵：用**真实预训练权重**逐个小模型实测剪枝是否成立，并逐条记录。

为每个模型记录：架构族 / 参数量 / 检出的可剪域与目标数 / 剪枝是否生效 / 剪枝后能否前向 /
真实指标变化（LM=perplexity，CNN=与原模型 top-1 决策一致率）/ 结论 / 备注。
结果增量写入 markdown + json，便于以后给用户参考"哪些模型/结构支持到什么程度"。

运行：HF_HUB_DISABLE_XET=1 python examples/pruning/support_matrix.py
"""
import amct_pytorch.experimental.pruning as P
import torch
import os
import json
import time
import traceback
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

OUT_MD = os.path.join(
    os.environ.get(
        "AMCT_PRUNE_RESULT_DIR",
        os.path.join(
            os.path.dirname(
                os.path.abspath(__file__)),
            "results")),
    "support_matrix.md")
OUT_JSON = OUT_MD.replace(".md", ".json")
RESULTS = []


def _nparams(m):
    return sum(p.numel() for p in m.parameters())


# ---- LLM (dense FFN / MoE) ----
LLM_MODELS = [
    ("SmolLM-135M", "HuggingFaceTB/SmolLM-135M", "Llama"),
    ("opt-125m", "facebook/opt-125m", "OPT"),
    ("pythia-160m", "EleutherAI/pythia-160m", "GPTNeoX"),
    ("Qwen2.5-0.5B", "Qwen/Qwen2.5-0.5B", "Qwen2"),
    ("gpt2", "gpt2", "GPT2(Conv1D)"),          # 预期负例：FFN 是 Conv1D 非 nn.Linear
    ("distilgpt2", "distilgpt2", "GPT2(Conv1D)"),
]

# ---- CNN (torchvision, 真 ImageNet 权重) ----
CNN_MODELS = ["resnet18", "mobilenet_v2", "regnet_y_400mf", "convnext_tiny", "vgg11_bn", "squeezenet1_1"]

LLM_TEXT = ("Model compression via pruning and quantization makes deep neural networks cheaper to "
            "deploy on edge devices and accelerators while keeping most of their accuracy. "
            "Structured pruning removes whole channels neurons or experts so the result stays dense. " * 8)


@torch.no_grad()
def _ppl(model, ids):
    return float(torch.exp(model(ids, labels=ids).loss.float()))


def run_llm(label, repo, family):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import copy
    rec = {"group": "LLM", "model": label, "family": family}
    try:
        tok = AutoTokenizer.from_pretrained(repo)
        m = AutoModelForCausalLM.from_pretrained(repo, dtype=torch.float32).eval()
        ids = tok(LLM_TEXT, return_tensors="pt").input_ids[:, :192]
        # 只取非空切片做 calib（短文本时尾部切片可能为空 → 注意力 reshape 崩溃）
        calib = [ids[:, i:i + 48] for i in range(0, ids.shape[1], 48) if ids[:, i:i + 48].shape[1] > 0]
        rec["params"] = _nparams(m)
        diag = P.diagnose(m, data=calib)
        rec["targets"] = dict(diag.targets)
        ppl0 = _ppl(m, ids)
        mp = copy.deepcopy(m)
        P.prune(mp, {"methods": {"dense": {"name": "low_variance", "kwargs": {"prune_ratio": 0.2}}}},
                data=calib)
        red = 1 - _nparams(mp) / rec["params"]
        ppl1 = _ppl(mp, ids)
        rec.update(reduction=round(red, 4), metric_before=round(ppl0, 3), metric_after=round(ppl1, 3),
                   metric="perplexity", forward_ok=True)
        rec["status"] = "SUPPORTED" if (diag.targets.get("dense", 0) > 0 and red > 0.001) else "UNSUPPORTED"
        rec["note"] = "" if rec["status"] == "SUPPORTED" else "no dense FFN nn.Linear targets detected"
    except Exception as e:
        rec.update(status="ERROR", note=f"{type(e).__name__}: {str(e)[:120]}", error=traceback.format_exc()[-300:])
    RESULTS.append(rec)
    _flush()
    _print(rec)


def run_cnn(name):
    import torchvision.models as tvm
    import copy
    rec = {"group": "CNN", "model": name, "family": "torchvision"}
    try:
        ctor = getattr(tvm, name)
        try:
            m = ctor(weights="IMAGENET1K_V1").eval()
        except Exception:
            m = ctor(weights="DEFAULT").eval()
        rec["params"] = _nparams(m)
        calib = [torch.randn(2, 3, 224, 224) for _ in range(3)]
        with torch.no_grad():
            ref = torch.cat([m(b).argmax(-1) for b in calib])
        mp = copy.deepcopy(m)
        P.prune(mp, {"methods": {"cnn": {"name": "depgraph",
                "kwargs": {"prune_ratio": 0.3, "importance": "magnitude"}}}}, data=calib)
        red = 1 - _nparams(mp) / rec["params"]
        with torch.no_grad():
            pred = torch.cat([mp(b).argmax(-1) for b in calib])
        agree = (pred == ref).float().mean().item()
        rec.update(reduction=round(red, 4), metric_before=1.0, metric_after=round(agree, 3),
                   metric="top1-agreement", forward_ok=True,
                   status="SUPPORTED" if red > 0.001 else "UNSUPPORTED", note="")
    except Exception as e:
        rec.update(status="ERROR", note=f"{type(e).__name__}: {str(e)[:120]}", error=traceback.format_exc()[-300:])
    RESULTS.append(rec)
    _flush()
    _print(rec)


def _print(r):
    print(f"[MATRIX] {r['group']:3s} {r['model']:16s} {r.get('family',''):14s} "
          f"status={r['status']:11s} params={r.get('params','?')} "
          f"targets={r.get('targets','')} red={r.get('reduction','')} "
          f"{r.get('metric','')}:{r.get('metric_before','')}->{r.get('metric_after','')} {r.get('note','')}")


def _flush():
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    json.dump(RESULTS, open(OUT_JSON, "w"), indent=2, ensure_ascii=False)
    lines = ["# AMCT experimental.pruning — 模型支持矩阵 (真实权重实测)", "",
             "| 组 | 模型 | 族 | 状态 | 参数 | 检出域(targets) | 剪枝率减 | 指标 | 前→后 | 备注 |",
             "|----|------|----|------|------|------------------|----------|------|-------|------|"]
    for r in RESULTS:
        ba = f"{r.get('metric_before','')}->{r.get('metric_after','')}"
        lines.append(f"| {r['group']} | {r['model']} | {r.get('family','')} | {r['status']} | "
                     f"{r.get('params','')} | {r.get('targets','')} | {r.get('reduction','')} | "
                     f"{r.get('metric','')} | {ba} | {r.get('note','')} |")
    open(OUT_MD, "w").write("\n".join(lines) + "\n")


def main():
    t0 = time.time()
    for label, repo, fam in LLM_MODELS:
        run_llm(label, repo, fam)
    for name in CNN_MODELS:
        run_cnn(name)
    print(f"[MATRIX] done {len(RESULTS)} models in {time.time()-t0:.0f}s → {OUT_MD}")


if __name__ == "__main__":
    main()
