# -*- coding: UTF-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
"""模型动物园 E2E 流水线：训练一批小型**真实**模型，逐阶段记录 (精度, 参数量, 量化级别)。

不下载大模型权重 —— 而是用 *缩小架构*（减层 / 减专家 / 减维）的小模型，在**小型真实数据集**上
**从头训练**到有意义的精度，再走 baseline → 剪枝(可量化感知) → AMCT 量化 流水线。每过一个阶段，
就把 (metric, params, quant_level) 追加写入可续跑的 JSON + Markdown 表（results/ 下）。

三族：
  - CNN  @ MNIST           （图像分类，metric=top-1 acc）—— 变深度/宽度
  - Dense-LM @ WikiText     （因果 LM，metric=ppl & 下一 token top-1）—— 变层数/维度
  - MoE-LM @ WikiText       （MoE 因果 LM，metric=ppl）—— 变专家数/层数

量化在 NPU 上**真实发火**（amct.quantize→convert，int8）。CNN 仅 Linear 头被量化（AMCT 量化 Linear，
不量化 Conv2d），LM/MoE 全 Linear 被量化。剪枝用本仓库 reconstruct(输出保留) / variance_channel / moe 方法。

运行（NPU 真量化）：
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
  ASCEND_RT_VISIBLE_DEVICES=3 HF_HUB_DISABLE_XET=1 \
    .venv_npu/bin/python examples/pruning/e2e_model_zoo_pipeline.py [--only cnn|lm|moe] [--limit N]
可续跑：再次运行会跳过 JSON 中已完成的 (model_id)。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = os.environ.get("AMCT_PRUNE_RESULT_DIR", os.path.join(HERE, "results"))
JSON_PATH = os.path.join(RESULT_DIR, "model_zoo_pipeline.json")
MD_PATH = os.path.join(RESULT_DIR, "model_zoo_pipeline.md")

try:
    import torch_npu

    _ = torch_npu  # 副作用导入（注册 torch.npu 后端）；显式引用以避免“未使用”告警
    _HAS_NPU = torch.npu.is_available()
    DEV = "npu:0" if _HAS_NPU else "cpu"
except Exception:
    _HAS_NPU = False
    DEV = "cpu"


def nparams(m):
    return sum(p.numel() for p in m.parameters())


# ===================================================================== data
def mnist_loaders(batch=256, root="/tmp/mnist"):
    from torchvision import datasets, transforms
    import torch.utils.data as D

    tf = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    tr = datasets.MNIST(root, train=True, download=True, transform=tf)
    te = datasets.MNIST(root, train=False, download=True, transform=tf)
    return (
        D.DataLoader(tr, batch_size=batch, shuffle=True, num_workers=0),
        D.DataLoader(te, batch_size=512, num_workers=0),
    )


_WIKI_IDS = None


def wikitext_ids():
    global _WIKI_IDS
    if _WIKI_IDS is None:
        from transformers import AutoTokenizer
        from huggingface_hub import hf_hub_download
        import pyarrow.parquet as pq

        tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M")
        path = hf_hub_download(
            repo_id="Salesforce/wikitext",
            repo_type="dataset",
            filename="wikitext-2-raw-v1/train-00000-of-00001.parquet",
        )
        txt = " ".join(
            t
            for t in pq.read_table(path).column("text").to_pylist()[:12000]
            if t and t.strip()
        )
        ids = tok(txt, return_tensors="pt").input_ids[0]
        _WIKI_IDS = (ids, tok.vocab_size)
    return _WIKI_IDS


# ===================================================================== models
class TinyCNN(nn.Module):
    """MNIST 用小 CNN：depth 个 (conv-bn-relu) 残差段 + 线性头。"""

    def __init__(self, width=16, depth=3, num_classes=10):
        super().__init__()
        chs = [1] + [width * (2**i) for i in range(depth)]
        layers = []
        for i in range(depth):
            stride = 2 if i > 0 else 1
            layers += [
                nn.Conv2d(chs[i], chs[i + 1], 3, padding=1, stride=stride, bias=False),
                nn.BatchNorm2d(chs[i + 1]),
                nn.ReLU(inplace=True),
            ]
        self.features = nn.Sequential(*layers)
        self.head = nn.Linear(chs[-1], num_classes)

    def forward(self, x):
        x = self.features(x)
        return self.head(F.adaptive_avg_pool2d(x, 1).flatten(1))


class TinyLM(nn.Module):
    """缩小的因果 Transformer LM（减层/减维），WikiText 上从头训练。"""

    def __init__(self, vocab, dim=128, layers=4, heads=4, seq=128, moe=0):
        super().__init__()
        self.seq = seq
        self.tok = nn.Embedding(vocab, dim)
        self.pos = nn.Embedding(seq, dim)
        self.blocks = nn.ModuleList([_Block(dim, heads, moe) for _ in range(layers)])
        self.norm = nn.LayerNorm(dim)
        self.headw = nn.Linear(dim, vocab, bias=False)

    def forward(self, idx, labels=None):
        b, t = idx.shape
        pos = torch.arange(t, device=idx.device)
        h = self.tok(idx) + self.pos(pos)[None]
        for blk in self.blocks:
            h = blk(h)
        logits = self.headw(self.norm(h))
        out = type("O", (), {})()
        out.logits = logits
        if labels is not None:
            out.loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)), labels[:, 1:].reshape(-1)
            )
        return out


def _moe_route(flat, gate, experts, topk, n_moe):
    """简易 top-k MoE 路由（每 token 路由到对应专家），从 _Block.forward 提出以降深度。"""
    probs = F.softmax(gate(flat), -1)
    w, idx = torch.topk(probs, topk, dim=-1)
    w = w / w.sum(-1, keepdim=True)
    out = torch.zeros_like(flat)
    for k in range(topk):
        sel = idx[:, k]
        for e in range(n_moe):
            m = sel == e
            if m.any():
                out[m] += w[m, k] * experts[e](flat[m])
    return out


class _Block(nn.Module):
    def __init__(self, dim, heads, moe=0):
        super().__init__()
        self.n1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.n2 = nn.LayerNorm(dim)
        self.moe = moe
        if moe and moe > 0:
            self.gate = nn.Linear(dim, moe, bias=False)
            self.experts = nn.ModuleList([_FFN(dim) for _ in range(moe)])
            self.topk = 2
        else:
            self.mlp = _FFN(dim)

    def forward(self, x):
        t = x.size(1)
        mask = torch.triu(torch.ones(t, t, device=x.device, dtype=torch.bool), 1)
        a, _ = self.attn(
            self.n1(x), self.n1(x), self.n1(x), attn_mask=mask, need_weights=False
        )
        x = x + a
        h = self.n2(x)
        if self.moe:
            flat = h.reshape(-1, h.size(-1))
            out = _moe_route(flat, self.gate, self.experts, self.topk, self.moe)
            x = x + out.reshape_as(h)
        else:
            x = x + self.mlp(h)
        return x


class _FFN(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.gate_proj = nn.Linear(dim, dim * mult)
        self.up_proj = nn.Linear(dim, dim * mult)
        self.down_proj = nn.Linear(dim * mult, dim)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# ===================================================================== train/eval
def train_cnn(model, epochs=2):
    tr, te = mnist_loaders()
    model.to(DEV).train()
    opt = torch.optim.Adam(model.parameters(), 1e-3)
    for _ in range(epochs):
        for x, y in tr:
            x, y = x.to(DEV), y.to(DEV)
            opt.zero_grad()
            F.cross_entropy(model(x), y).backward()
            opt.step()
    calib = [next(iter(tr))[0][:64].cpu() for _ in range(4)]
    return calib, te


@torch.no_grad()
def acc_cnn(model, te):
    model.eval()
    c = t = 0
    for x, y in te:
        p = model(
            x.to(DEV).half()
            if next(model.parameters()).dtype == torch.float16
            else x.to(DEV)
        )
        c += (p.float().argmax(1).cpu() == y).sum().item()
        t += len(y)
    return c / t


def lm_batches(ids, seq, bs, n):
    out = []
    step = seq * bs
    for i in range(n):
        start = i * step
        end = start + step
        chunk = ids[start:end]
        if len(chunk) < step:
            break
        out.append(chunk.reshape(bs, seq))
    return out


def train_lm(model, seq=128, steps=400):
    ids, _ = wikitext_ids()
    model.to(DEV).train()
    opt = torch.optim.AdamW(model.parameters(), 3e-4)
    batches = lm_batches(ids, seq, 8, steps + 40)
    held = batches[-20:]
    train_b = batches[:-20]
    step = 0
    while step < steps:
        for b in train_b:
            if step >= steps:
                break
            b = b.to(DEV)
            opt.zero_grad()
            model(b, labels=b).loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1
    calib = [b for b in train_b[:6]]
    return calib, held


@torch.no_grad()
def ppl_acc_lm(model, held):
    model.eval()
    tot_loss = tot_tok = correct = 0
    for b in held:
        b = b.to(DEV)
        out = model(b, labels=b)
        logits = out.logits[:, :-1].reshape(-1, out.logits.size(-1))
        tgt = b[:, 1:].reshape(-1)
        tot_loss += F.cross_entropy(logits.float(), tgt, reduction="sum").item()
        correct += (logits.float().argmax(-1) == tgt).sum().item()
        tot_tok += tgt.numel()
    return math.exp(tot_loss / tot_tok), correct / tot_tok


# ===================================================================== zoo spec
def _prune_cfg(domain, name, ratio):
    return {"methods": {domain: {"name": name, "kwargs": {"prune_ratio": ratio}}}}


def _cnn_specs():
    """CNN @ MNIST：宽度×深度（最小也要够准，避免 w8/d2 这种欠拟合点）。"""
    return [
        {
            "id": f"cnn_w{width}_d{depth}",
            "family": "cnn",
            "build": (lambda w=width, d=depth: TinyCNN(w, d)),
            "prune": _prune_cfg("cnn", "reconstruct", 0.3),
        }
        for width in (16, 24, 32)
        for depth in (3, 4)
    ]


def _lm_specs():
    """Dense-LM @ WikiText：层数×维度。"""
    return [
        {
            "id": f"lm_L{layers}_d{dim}",
            "family": "lm",
            "build": (
                lambda L=layers, D_=dim: TinyLM(
                    wikitext_ids()[1], dim=D_, layers=L, heads=4
                )
            ),
            "prune": _prune_cfg("dense", "reconstruct", 0.3),
        }
        for layers in (2, 4, 6)
        for dim in (96, 160)
    ]


def _moe_specs():
    """MoE-LM @ WikiText：专家数×层数。"""
    return [
        {
            "id": f"moe_E{experts}_L{layers}",
            "family": "moe",
            "build": (
                lambda E=experts, L=layers: TinyLM(
                    wikitext_ids()[1], dim=128, layers=L, heads=4, moe=E
                )
            ),
            "prune": _prune_cfg("moe", "activation_count", 0.5),
        }
        for experts in (4, 8)
        for layers in (2, 4)
    ]


def _hf_specs():
    """真实 HuggingFace 小模型（预训练，全 Linear → 量化全发火），可续扩展。"""
    return [
        {
            "id": "hf_" + hf_id.split("/")[-1],
            "family": "hf",
            "hf_id": hf_id,
            "prune": _prune_cfg("dense", "reconstruct", 0.2),
        }
        for hf_id in hf_model_ids()
    ]


def build_zoo():
    return _cnn_specs() + _lm_specs() + _moe_specs() + _hf_specs()


# 小型 HF 因果 LM 种子表（下载快、可跑 prune+量化）；用 results/hf_model_queue.txt 持续追加更多。
_HF_SEED = [
    "HuggingFaceTB/SmolLM-135M",
    "distilgpt2",
    "gpt2",
    "EleutherAI/pythia-70m",
    "EleutherAI/pythia-160m",
]


def _queue_extra_ids(qf, existing):
    """从队列文件读取额外 HF model id（跳过注释/空行/重复）。"""
    extra = []
    with open(qf) as f:
        for line in f:
            s = line.strip()
            if s and not s.startswith("#") and s not in existing:
                extra.append(s)
    return extra


def hf_model_ids():
    ids = list(_HF_SEED)
    qf = os.path.join(RESULT_DIR, "hf_model_queue.txt")
    if os.path.exists(qf):
        ids.extend(_queue_extra_ids(qf, ids))
    return ids


# ===================================================================== pipeline
def _load_done():
    if not os.path.exists(JSON_PATH):
        return {"runs": []}
    with open(JSON_PATH) as f:
        return json.load(f)


def _save(db):
    os.makedirs(RESULT_DIR, exist_ok=True)
    with open(JSON_PATH, "w") as f:
        json.dump(db, f, indent=2)
    # MD 表
    lines = [
        "# Model-Zoo E2E Pipeline — (metric, params, quant) per stage",
        "",
        f"Device: {DEV} | quant: real AMCT int8 on NPU (Linear only)"
        if _HAS_NPU
        else f"Device: {DEV} | quant: skipped (no NPU)",
        "",
        "| model | family | stage | metric | value | params | MB | size↓% | quant |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for run in db["runs"]:
        mb0 = run["stages"][0].get("mb") or 1
        for s in run["stages"]:
            mb = s.get("mb", 0)
            sz = 100 * (1 - mb / mb0)
            lines.append(
                f"| {run['id']} | {run['family']} | {s['stage']} | {s['metric']} | "
                f"{s['value']:.4f} | {s['params']:,} | {mb:.2f} | {sz:.1f}% | {s['quant']} |"
            )
    with open(MD_PATH, "w") as f:
        f.write("\n".join(lines) + "\n")


def load_hf(hf_id):
    """加载真实 HF 因果 LM + 用其**自身**分词器在 WikiText 上准备 calib/held（ppl 才有意义）。"""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq

    tok = AutoTokenizer.from_pretrained(hf_id)
    dtype = torch.float16 if _HAS_NPU else torch.float32
    model = AutoModelForCausalLM.from_pretrained(hf_id, dtype=dtype)
    path = hf_hub_download(
        repo_id="Salesforce/wikitext",
        repo_type="dataset",
        filename="wikitext-2-raw-v1/train-00000-of-00001.parquet",
    )
    txt = " ".join(
        t
        for t in pq.read_table(path).column("text").to_pylist()[:8000]
        if t and t.strip()
    )
    ids = tok(txt, return_tensors="pt").input_ids[0]
    batches = lm_batches(ids, 128, 4, 24)
    return model, batches[:6], batches[6:16]


def amct_int8(model):
    import amct_pytorch as amct

    amct.quantize(model)
    amct.convert(model)


def run_one(spec, db):
    import copy
    import amct_pytorch.experimental.pruning as P

    fam = spec["family"]
    t0 = time.time()
    if fam == "hf":
        model, calib, held = load_hf(spec["hf_id"])
        metric_name = "ppl"

        def eval_fp(m):
            return ppl_acc_lm(m, held)[0]
    elif fam == "cnn":
        model = spec["build"]()
        calib, te = train_cnn(model)
        metric_name = "acc"

        def eval_fp(m):
            return acc_cnn(m, te)
    else:
        model = spec["build"]()
        calib, held = train_lm(model)
        metric_name = "ppl"

        def eval_fp(m):
            return ppl_acc_lm(m, held)[0]

    stages = []

    def record(stage, m, quant, params=None):
        # 量化不改变权重**数量**只改精度(bytes)；量化后 nparams 会漏计(权重存为 buffer)，
        # 故量化阶段沿用剪枝后的权重数，再用 bytes/权重 体现真实体积(MB)。
        val = eval_fp(m)
        pc = params if params is not None else nparams(m)
        bytes_per = (
            1 if quant == "int8" else (1 if quant == "int4" else (2 if _HAS_NPU else 4))
        )
        mb = round(pc * bytes_per / 1e6, 3)
        stages.append(
            {
                "stage": stage,
                "metric": metric_name,
                "value": float(val),
                "params": pc,
                "mb": mb,
                "quant": quant,
            }
        )
        print(
            f"    [{spec['id']}] {stage:18s} {metric_name}={val:.4f} params={pc:,} mb={mb} quant={quant}",
            flush=True,
        )

    # stage1: baseline (NPU fp16 if available)
    base = model.to(DEV).half() if _HAS_NPU else model.to(DEV).float()
    base.eval()
    record("baseline", base, "none")

    # stage2: prune (CPU fp32 拷贝做结构化剪枝，再回 NPU fp16)。某些 HF 架构(如 GPT2 的 Conv1D FFN)
    # 不被 dense 域识别 → 自然跳过（参数不变），仍记录该阶段。
    mp = copy.deepcopy(model).float().cpu()
    try:
        P.prune(mp, spec["prune"], data=[c.cpu() for c in calib])
    except Exception as e:
        print(
            f"    [{spec['id']}] prune skipped: {type(e).__name__} {str(e)[:120]}",
            flush=True,
        )
    mp = mp.to(DEV).half() if _HAS_NPU else mp.to(DEV).float()
    mp.eval()
    record("prune", mp, "none")

    # stage3: prune + AMCT int8 quant (real, NPU only)
    if _HAS_NPU:
        try:
            mq = copy.deepcopy(mp)
            pre_q_params = nparams(mp)
            amct_int8(mq)
            record("prune+int8", mq, "int8", params=pre_q_params)
        except Exception as e:
            print(
                f"    [{spec['id']}] quant FAILED: {type(e).__name__} {str(e)[:120]}",
                flush=True,
            )

    db["runs"].append(
        {
            "id": spec["id"],
            "family": fam,
            "seconds": round(time.time() - t0, 1),
            "stages": stages,
        }
    )
    _save(db)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["cnn", "lm", "moe", "hf"], default=None)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(RESULT_DIR, exist_ok=True)
    db = _load_done()
    done = {r["id"] for r in db["runs"]}
    zoo = build_zoo()
    if args.only:
        zoo = [z for z in zoo if z["family"] == args.only]
    todo = [z for z in zoo if z["id"] not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(
        f"[ZOO] device={DEV} total={len(zoo)} done={len(done)} todo={len(todo)}",
        flush=True,
    )
    for i, spec in enumerate(todo):
        print(f"[ZOO {i + 1}/{len(todo)}] {spec['id']} ({spec['family']})", flush=True)
        try:
            run_one(spec, db)
        except Exception as e:
            import traceback

            print(
                f"[ZOO] {spec['id']} ERROR: {type(e).__name__} {str(e)[:160]}",
                flush=True,
            )
            traceback.print_exc()
    print(f"[ZOO] done. results: {JSON_PATH}", flush=True)


if __name__ == "__main__":
    main()
