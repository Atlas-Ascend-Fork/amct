# -*- coding: UTF-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
"""真·量化 E2E（NPU）：SmolLM-135M(fp16) 上量化**实际发火**，测真实 perplexity。

CPU 上 amct 量化是 no-op（fp32 被 dtype 约束跳过；且 fake-quant 需 torch_npu）。
本脚本在 NPU + fp16 上跑，量化真正把 Linear 换成 MinMaxQuant→NpuWeightQuantizedLinear，
从而隔离并测量 **量化对 perplexity 的真实影响**，以及 剪枝+量化 的联合影响。

运行：ASCEND_RT_VISIBLE_DEVICES=3 HF_HUB_DISABLE_XET=1 python examples/pruning/e2e_smollm_npu_quant.py
"""

import json
import os
import time
import torch
import torch_npu
import amct_pytorch as amct
import amct_pytorch.experimental.pruning as P

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
# torch_npu 提供 NPU 后端（导入即注册）；引用以表明其被使用。
_NPU_AVAILABLE = torch_npu is not None and torch.npu.is_available()
MODEL = "HuggingFaceTB/SmolLM-135M"
RESULT = os.path.join(
    os.environ.get(
        "AMCT_PRUNE_RESULT_DIR",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"),
    ),
    "e2e_smollm135m_npu_quant.json",
)
DEV = "npu:0"  # ASCEND_RT_VISIBLE_DEVICES=3 → 物理 NPU3

TEXT = (
    "Machine learning is a field of study in artificial intelligence concerned with the development "
    "of statistical algorithms that can learn from data and generalize to unseen examples. Deep neural "
    "networks have driven much of the recent progress in vision, language, and speech. Model compression "
    "techniques such as pruning and quantization reduce the size and cost of these networks."
) * 3


@torch.no_grad()
def ppl(model, ids):
    out = model(ids, labels=ids)
    return float(torch.exp(out.loss.float()))


def nparams(m):
    return sum(p.numel() for p in m.parameters())


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import copy

    tok = AutoTokenizer.from_pretrained(MODEL)
    base = (
        AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16).to(DEV).eval()
    )
    ids = tok(TEXT, return_tensors="pt").input_ids[:, :256].to(DEV)
    calib = [ids[:, i : i + 64] for i in range(0, 256, 64)]

    base_ppl = ppl(base, ids)
    print(f"[NPU-E2E] baseline(fp16): params={nparams(base):,}  ppl={base_ppl:.4f}")
    rec = {
        "model": MODEL,
        "device": "NPU3/fp16",
        "baseline_ppl": base_ppl,
        "stages": [],
    }

    # (1) 量化单独：真实发火，隔离量化对 ppl 的影响
    mq = copy.deepcopy(base)
    t0 = time.time()
    amct.quantize(mq)
    amct.convert(mq)
    types = sorted(
        {type(m).__name__ for m in mq.modules() if "Quant" in type(m).__name__}
    )
    ppl_q = ppl(mq, ids)
    print(
        f"[NPU-E2E] quant-only: ppl {base_ppl:.4f} → {ppl_q:.4f} (Δ={ppl_q - base_ppl:+.4f})  "
        f"quant_modules={types}  {time.time() - t0:.1f}s"
    )
    rec["stages"].append(
        {
            "stage": "quant_only",
            "ppl": ppl_q,
            "delta": ppl_q - base_ppl,
            "quant_module_types": types,
        }
    )

    # (2) 剪枝 → 量化 → 转换：联合压缩，真实 ppl
    mp = copy.deepcopy(base)
    P.prune(
        mp,
        {
            "methods": {
                "dense": {"name": "low_variance", "kwargs": {"prune_ratio": 0.1}}
            }
        },
        data=calib,
    )
    ppl_pruned = ppl(mp, ids)
    amct.quantize(mp)
    amct.convert(mp)
    ppl_pq = ppl(mp, ids)
    print(
        f"[NPU-E2E] prune(0.1)→quant→convert: base {base_ppl:.4f} → 剪枝 {ppl_pruned:.4f} → +量化 {ppl_pq:.4f}"
    )
    rec["stages"].append(
        {
            "stage": "prune0.1_then_quant",
            "ppl_pruned": ppl_pruned,
            "ppl_prune_quant": ppl_pq,
            "params": nparams(mp),
        }
    )

    os.makedirs(os.path.dirname(RESULT), exist_ok=True)
    with open(RESULT, "w") as f:
        json.dump(rec, f, indent=2, ensure_ascii=False)
    print(f"[NPU-E2E] 结果已保存: {RESULT}")


if __name__ == "__main__":
    main()
