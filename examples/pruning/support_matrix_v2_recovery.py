# -*- coding: UTF-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
"""Support matrix v2 — measured WITH recovery (no caveats, all label-free).

- LLM: prune(dense 0.2) -> perplexity -> self-supervised fine-tune (labels=input_ids) -> recovered ppl
- CNN: prune(depgraph 0.3) with magnitude vs taylor -> top-1 decision agreement;
       then label-free self-distillation (match original logits) -> recovered agreement

Run: HF_HUB_DISABLE_XET=1 python examples/pruning/support_matrix_v2_recovery.py
"""

import os
import json
import copy
import time

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
import torch
import torch.nn.functional as F
import amct_pytorch.experimental.pruning as P
from amct_pytorch.experimental.pruning import finetune

OUT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "results", "support_matrix_v2.json"
)
LLM = [
    ("SmolLM-135M", "HuggingFaceTB/SmolLM-135M"),
    ("opt-125m", "facebook/opt-125m"),
    ("pythia-160m", "EleutherAI/pythia-160m"),
    ("Qwen2.5-0.5B", "Qwen/Qwen2.5-0.5B"),
]
CNN = [
    "resnet18",
    "mobilenet_v2",
    "regnet_y_400mf",
    "convnext_tiny",
    "vgg11_bn",
    "squeezenet1_1",
]
TEXT = (
    "Model compression via pruning and quantization makes deep neural networks cheaper to deploy "
    "on edge devices while keeping most of their accuracy. Structured pruning removes whole "
    "channels neurons or experts so the result stays dense and efficient. " * 8
)
res = {"llm": [], "cnn": []}


@torch.no_grad()
def ppl(m, ids):
    return float(torch.exp(m(ids, labels=ids).loss.float()))


def run_llm(label, repo):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(repo)
    m = AutoModelForCausalLM.from_pretrained(repo, dtype=torch.float32).eval()
    ids = tok(TEXT, return_tensors="pt").input_ids[:, :192]
    calib = [
        ids[:, i : i + 48]
        for i in range(0, ids.shape[1], 48)
        if ids[:, i : i + 48].shape[1] > 0
    ]
    base = ppl(m, ids)
    P.prune(
        m,
        {
            "methods": {
                "dense": {"name": "low_variance", "kwargs": {"prune_ratio": 0.2}}
            }
        },
        data=calib,
    )
    pruned = ppl(m, ids)
    finetune(m, calib, loss_fn=lambda mm, b: mm(b, labels=b).loss, steps=60, lr=2e-5)
    recovered = ppl(m, ids)
    row = {
        "model": label,
        "ppl_base": round(base, 2),
        "ppl_pruned": round(pruned, 2),
        "ppl_recovered": round(recovered, 2),
    }
    res["llm"].append(row)
    print(
        f"[V2-LLM] {label:14s} base {base:.2f} -> pruned {pruned:.2f} -> recovered {recovered:.2f}"
    )


def agree(m, ref, calib):
    with torch.no_grad():
        pred = torch.cat([m(b).argmax(-1) for b in calib])
    return (pred == ref).float().mean().item()


def run_cnn(name):
    import torchvision.models as tvm

    base = tvm.__dict__[name](weights="IMAGENET1K_V1").eval()
    calib = [torch.randn(2, 3, 224, 224) for _ in range(4)]
    with torch.no_grad():
        ref = torch.cat([base(b).argmax(-1) for b in calib])
    out = {"model": name}
    for imp in ("magnitude", "taylor"):
        m = copy.deepcopy(base)
        P.prune(
            m,
            {
                "methods": {
                    "cnn": {
                        "name": "depgraph",
                        "kwargs": {"prune_ratio": 0.3, "importance": imp},
                    }
                }
            },
            data=calib,
        )
        out[f"agree_{imp}"] = round(agree(m, ref, calib), 3)
        if imp == "taylor":
            # label-free recovery: self-distillation to match the original model's logits
            orig = base
            finetune(
                m,
                calib,
                loss_fn=lambda mm, x: F.mse_loss(mm(x), orig(x).detach()),
                steps=150,
                lr=1e-3,
            )
            out["agree_taylor_recovered"] = round(agree(m, ref, calib), 3)
    res["cnn"].append(out)
    print(
        f"[V2-CNN] {name:16s} magnitude {out['agree_magnitude']:.3f} | "
        f"taylor {out['agree_taylor']:.3f} | taylor+distill {out['agree_taylor_recovered']:.3f}"
    )


def main():
    t0 = time.time()
    for label, repo in LLM:
        run_llm(label, repo)
    for name in CNN:
        run_cnn(name)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=2)
    print(f"[V2] done in {time.time() - t0:.0f}s -> {OUT}")


if __name__ == "__main__":
    main()
