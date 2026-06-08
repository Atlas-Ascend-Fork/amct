# -*- coding: UTF-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
"""样例：输出保留(重构)剪枝 vs 均匀比例剪枝 —— 在 held-out 上的 (尺寸, 精度) 对比。

Llama 式 FFN:  h = silu(gate(x)) * up(x);  y = down(h)        (中间维 M)
``dense.reconstruct`` 方法剪掉低显著度中间神经元后，用最小二乘**重构 down_proj 权重**补偿，
使块输出尽量不变 —— 低损失、**无需微调**。对比基线 ``dense.low_variance``（朴素切片）。

要点（重要）：校准集必须**有代表性且多样**，且 token 数远多于中间维。重复文本会让激活秩亏、
重构过拟合，在 held-out 上发散。这里用多样的 WikiText，评估用**不相交**的 held-out 片段，
报告 (参数下降, 困惑度) 对。

运行：HF_HUB_DISABLE_XET=1 python examples/pruning/reconstruct_vs_uniform.py
"""

import os
import copy

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
import torch
import amct_pytorch.experimental.pruning as P

MODEL = "HuggingFaceTB/SmolLM-135M"


def _wikitext_ids(tok, n_text=8000):
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq

    path = hf_hub_download(
        repo_id="Salesforce/wikitext",
        repo_type="dataset",
        filename="wikitext-2-raw-v1/train-00000-of-00001.parquet",
    )
    texts = pq.read_table(path).column("text").to_pylist()[:n_text]
    joined = " ".join(t for t in texts if t and t.strip())
    return tok(joined, return_tensors="pt").input_ids[0]


@torch.no_grad()
def perplexity(model, ids):
    return float(torch.exp(model(ids, labels=ids).loss.float()))


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    base = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()
    ids = _wikitext_ids(tok)
    calib = [
        ids[i * 512 : (i + 1) * 512].unsqueeze(0) for i in range(8)
    ]  # 4096 diverse tokens
    eval_ids = ids[6000:6512].unsqueeze(0)  # held-out, disjoint
    p0 = n_params(base)
    print(
        f"[REC] baseline: params={p0:,}  held-out ppl={perplexity(base, eval_ids):.2f}  "
        f"(calib=4096 diverse tokens)"
    )
    print(
        f"{'ratio':>6} | {'low_variance (size, ppl)':>26} | {'reconstruct (size, ppl)':>26}"
    )
    for ratio in (0.2, 0.3, 0.4, 0.5):
        rows = {}
        for name in ("low_variance", "reconstruct"):
            m = copy.deepcopy(base)
            P.prune(
                m,
                {
                    "methods": {
                        "dense": {"name": name, "kwargs": {"prune_ratio": ratio}}
                    }
                },
                data=calib,
            )
            rows[name] = (1 - n_params(m) / p0, perplexity(m, eval_ids))
        (su, pu), (sr, pr) = rows["low_variance"], rows["reconstruct"]
        print(
            f"{ratio:>6.2f} | {su * 100:6.1f}% off  ppl {pu:8.2f} | {sr * 100:6.1f}% off  ppl {pr:8.2f}"
        )


if __name__ == "__main__":
    main()
