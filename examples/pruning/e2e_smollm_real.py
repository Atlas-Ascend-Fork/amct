# -*- coding: UTF-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
"""真·E2E：用**预训练实权重** SmolLM-135M（Llama 架构）测真实任务指标(perplexity)。

流程：baseline ppl → 容差自动剪枝(实测 ppl) → 联合预算分割 → 量化→转换(实测 ppl)。
与单测不同：这里是**真实权重 + 真实指标**，而非随机权重 + forward 充实度。

运行：HF_HUB_DISABLE_XET=1 python examples/pruning/e2e_smollm_real.py
"""
import json
import os
import time

import torch

import amct_pytorch.experimental.pruning as P

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
MODEL = "HuggingFaceTB/SmolLM-135M"
RESULT = os.path.join(
    os.environ.get(
        "AMCT_PRUNE_RESULT_DIR",
        os.path.join(
            os.path.dirname(
                os.path.abspath(__file__)),
            "results")),
    "e2e_smollm135m.json")

TEXT = (
    "Machine learning is a field of study in artificial intelligence concerned with the "
    "development of statistical algorithms that can learn from data and generalize to unseen "
    "examples. Deep neural networks have driven much of the recent progress, achieving strong "
    "results in vision, language, and speech. Model compression techniques such as pruning and "
    "quantization reduce the size and computational cost of these networks, making them practical "
    "to deploy on edge devices and accelerators while preserving most of their accuracy. "
    "Structured pruning removes whole channels, neurons, or experts so the resulting model is "
    "dense and runs efficiently on standard hardware."
) * 3


def load():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()
    return model, tok


@torch.no_grad()
def perplexity(model, ids):
    """真实任务指标：因果语言建模困惑度（越低越好），无需标签。"""
    out = model(ids, labels=ids)
    return float(torch.exp(out.loss))


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def main():
    model, tok = load()
    ids = tok(TEXT, return_tensors="pt").input_ids[:, :256]
    # 校准批（方差类剪枝 + 默认充实度评估都用它）
    calib = [ids[:, i:i + 64] for i in range(0, 256, 64)]

    import copy
    base_ppl = perplexity(model, ids)
    base_params = n_params(model)
    print(f"[E2E] baseline: params={base_params:,}  perplexity={base_ppl:.3f}")
    rec = {"model": MODEL, "baseline": {"params": base_params, "ppl": base_ppl}, "stages": []}

    # 真实任务指标驱动搜索：quality = -perplexity（越大越好）。tolerance 单位=ppl 增量。
    def neg_ppl(m):
        return -perplexity(m, ids)

    # ---- 0) 真实 prune→ppl 曲线（固定率，实测）----
    curve = {}
    for r in (0.05, 0.1, 0.15, 0.2, 0.3):
        mc = copy.deepcopy(model)
        P.prune(mc, {"methods": {"dense": {"name": "low_variance", "kwargs": {"prune_ratio": r}}}}, data=calib)
        curve[r] = perplexity(mc, ids)
        print(f"[E2E] fixed ratio {r:.2f}: ppl={curve[r]:.3f} (Δ={curve[r] - base_ppl:+.3f}), params={n_params(mc):,}")
    rec["prune_ppl_curve"] = {str(k): v for k, v in curve.items()}

    # ---- 1) 真·accuracy-based 自动剪枝：用 perplexity 驱动二分（tol = 允许的 ppl 增量）----
    m1 = copy.deepcopy(model)
    t0 = time.time()
    res = P.accuracy_based_auto_prune(m1, data=calib, tolerance=1.0, quality_fn=neg_ppl,
                                      ratio_grid=(0.05, 0.1, 0.15, 0.2, 0.3))
    ppl1 = perplexity(m1, ids)
    print(f"[E2E] auto-prune(perplexity, tol=+1.0ppl): {res.summary()}")
    print(f"      → params={n_params(m1):,}  ppl={ppl1:.3f} (Δ={ppl1 - base_ppl:+.3f})  {time.time() - t0:.1f}s")
    rec["stages"].append({"stage": "auto_prune_ppl_tol1.0", "chosen_ratio": res.chosen_ratio,
                          "applied": res.applied, "params": n_params(m1), "ppl": ppl1,
                          "measured_ppl_increase": res.quality_drop})

    # ---- 2) 联合精度预算分割（perplexity 驱动；prune 用掉实测一部分，剩余留给量化）----
    m2 = copy.deepcopy(model)
    jb = P.accuracy_based_auto_prune_with_budget(m2, data=calib, total_tolerance=2.0,
                                                 prune_fraction=0.5, quality_fn=neg_ppl,
                                                 ratio_grid=(0.05, 0.1, 0.15, 0.2, 0.3))
    ppl2 = perplexity(m2, ids)
    print(f"[E2E] budget(总+2.0ppl): {jb.summary()}")
    print(f"      → params={n_params(m2):,}  ppl={ppl2:.3f}  量化可用剩余预算={jb.remaining_tolerance:.3f}ppl")
    rec["stages"].append({"stage": "budget_total2.0ppl", "prune_tol": jb.prune_tolerance,
                          "prune_drop": jb.prune_drop, "remaining_for_quant": jb.remaining_tolerance,
                          "params": n_params(m2), "ppl": ppl2})

    # ---- 3) 剪枝 → 量化 → 转换，实测 ppl ----
    try:
        import amct_pytorch as amct
        # (a) 量化单独（不剪枝），隔离量化对 ppl 的影响
        mq = copy.deepcopy(model)
        amct.quantize(mq)
        amct.convert(mq)
        ppl_qonly = perplexity(mq, ids)
        # (b) 剪枝 → 量化 → 转换
        m3 = copy.deepcopy(model)
        P.prune(m3, {"methods": {"dense": {"name": "low_variance", "kwargs": {"prune_ratio": 0.1}}}},
                data=calib)
        ppl_pruned = perplexity(m3, ids)
        amct.quantize(m3)
        amct.convert(m3)
        ppl_q = perplexity(m3, ids)
        print(f"[E2E] quant-only: ppl {base_ppl:.3f} → {ppl_qonly:.3f} (Δ={ppl_qonly - base_ppl:+.3f})")
        print(f"[E2E] prune(0.1)→quant→convert: ppl {base_ppl:.3f} → 剪枝 {ppl_pruned:.3f} → 量化 {ppl_q:.3f}")
        rec["stages"].append({"stage": "quant_only", "ppl": ppl_qonly})
        rec["stages"].append({"stage": "prune0.1_then_quant", "ppl_pruned": ppl_pruned,
                              "ppl_quantized": ppl_q, "params": n_params(m3)})
    except Exception as e:
        print(f"[E2E] quantize 阶段跳过/失败: {type(e).__name__}: {str(e)[:160]}")
        rec["stages"].append({"stage": "prune_then_quant", "error": f"{type(e).__name__}: {str(e)[:160]}"})

    os.makedirs(os.path.dirname(RESULT), exist_ok=True)
    with open(RESULT, "w") as f:
        json.dump(rec, f, indent=2, ensure_ascii=False)
    print(f"[E2E] 结果已保存: {RESULT}")


if __name__ == "__main__":
    main()
