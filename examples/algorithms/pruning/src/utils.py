# ----------------------------------------------------------------------------
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------
"""Shared tiny models and data for the samples (random init, downloads no weights, CPU-only).

Covers the three pruning domains so amct pruning can be demonstrated without
network access or an NPU:
 - dense FFN (``MiniMLP``)  -> dense domain: prune the intermediate dim
 - CNN (``MiniCNN``)        -> cnn domain: prune channels along the dependency chain
 - MoE (``MiniMoE``)        -> moe domain: prune experts
"""

from __future__ import annotations

import json
import importlib.util
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class MiniMLP(nn.Module):
    """Two stacked FFN blocks (hidden->inter->hidden); the dense domain prunes the intermediate dim."""

    def __init__(self, vocab=256, hidden=64, inter=256, layers=2):
        super().__init__()
        self.embed = nn.Embedding(vocab, hidden)
        self.blocks = nn.ModuleList(
            [
                nn.ModuleDict(
                    {"fc1": nn.Linear(hidden, inter), "fc2": nn.Linear(inter, hidden)}
                )
                for _ in range(layers)
            ]
        )
        self.head = nn.Linear(hidden, vocab)

    def forward(self, input_ids):
        h = self.embed(input_ids)
        for b in self.blocks:
            h = h + b["fc2"](F.gelu(b["fc1"](h)))
        return self.head(h)


def make_mlp():
    torch.manual_seed(0)
    return MiniMLP().eval()


def mlp_data(n=6, batch=4, seq=16, vocab=256):
    torch.manual_seed(1)
    return [torch.randint(0, vocab, (batch, seq)) for _ in range(n)]


class MiniCNN(nn.Module):
    """Sequential conv chain + GAP head: the cnn domain detects (conv->bn->consumer) coupling for channel pruning."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.features(x)
        return self.head(self.pool(x).flatten(1))


def make_cnn():
    torch.manual_seed(0)
    return MiniCNN().eval()


def cnn_data(n=4, batch=4, size=16):
    torch.manual_seed(2)
    return [torch.randn(batch, 3, size, size) for _ in range(n)]


class Expert(nn.Module):
    def __init__(self, hidden, inter):
        super().__init__()
        self.fc1 = nn.Linear(hidden, inter)
        self.fc2 = nn.Linear(inter, hidden)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class MoELayer(nn.Module):
    def __init__(self, hidden=64, inter=128, n_experts=8, topk=2):
        super().__init__()
        self.num_experts, self.topk = n_experts, topk
        self.gate = nn.Linear(hidden, n_experts, bias=False)
        self.experts = nn.ModuleList([Expert(hidden, inter) for _ in range(n_experts)])

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


class MiniMoE(nn.Module):
    def __init__(self, vocab=256, hidden=64, layers=2):
        super().__init__()
        self.embed = nn.Embedding(vocab, hidden)
        self.layers = nn.ModuleList([MoELayer(hidden=hidden) for _ in range(layers)])
        self.head = nn.Linear(hidden, vocab)

    def forward(self, input_ids):
        h = self.embed(input_ids)
        for layer in self.layers:
            h = h + layer(h)
        return self.head(h)


def make_moe():
    torch.manual_seed(0)
    return MiniMoE().eval()


def moe_data(n=6, batch=4, seq=16, vocab=256):
    torch.manual_seed(3)
    return [torch.randint(0, vocab, (batch, seq)) for _ in range(n)]


def num_experts(m):
    for layer in m.layers:
        return layer.num_experts
    return None


def count_params(m):
    return sum(p.numel() for p in m.parameters())


def _disable_torchaudio_for_transformers():
    """Make transformers treat torchaudio as unavailable in this process.

    Some environments ship a torchaudio build whose import crashes while
    transformers is importing RNNT losses. Qwen3.6 does not need torchaudio,
    so we mask it out instead of touching the system package.
    """

    if getattr(importlib.util.find_spec, "_amct_torchaudio_patched", False):
        return

    original_find_spec = importlib.util.find_spec

    def _find_spec(name, package=None):
        if name == "torchaudio":
            return None
        return original_find_spec(name, package)

    _find_spec._amct_torchaudio_patched = True  # type: ignore[attr-defined]
    importlib.util.find_spec = _find_spec


def load_qwen36_moe(
    model_path,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map=None,
):
    _disable_torchaudio_for_transformers()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=trust_remote_code
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        trust_remote_code=trust_remote_code,
        low_cpu_mem_usage=True,
        device_map=device_map,
    ).eval()
    return model, tokenizer


def build_pileval_batches(tokenizer, n_samples, seq_len):
    from amct_pytorch.common.datasets.preproc import get_pileval

    return get_pileval(tokenizer, n_samples=n_samples, seq_len=seq_len)


def build_wikitext2_batches(tokenizer, seq_len=4096):
    from amct_pytorch.common.datasets.preproc import get_wiki_inputs

    return get_wiki_inputs(tokenizer, seq_len=seq_len)


def iter_model_logits(model, batches, device="cpu"):
    with torch.inference_mode():
        for batch in batches:
            outputs = model(batch.to(device))
            yield outputs.logits[:, :-1, :].contiguous()


def eval_wikitext2_ppl(model, batches, seq_len=4096, device="cpu"):
    """Evaluate WikiText2 PPL on the device where logits actually land.

    ``device`` is where model inputs are placed. For accelerate-dispatched
    models outputs come back on the same device, but for sharded models
    without dispatch hooks logits may stay on the lm_head device; the first
    logits chunk is probed and the evaluation then runs on that device.
    """
    from amct_pytorch.common.evaluate.eval_ppl import wikitext2_ppl

    logits_iter = iter_model_logits(model, batches, device=device)
    first = next(logits_iter, None)

    def _chained():
        if first is not None:
            yield first
        yield from logits_iter

    eval_device = str(first.device) if first is not None else device
    return wikitext2_ppl(
        _chained(),
        batches,
        device=eval_device,
        seq_len=seq_len,
    )


class NegativePplEvaluator:
    def __init__(self, batches, seq_len=4096, device="cpu"):
        self.batches = batches
        self.seq_len = seq_len
        self.device = device

    def evaluate(self, model):
        return -eval_wikitext2_ppl(
            model,
            self.batches,
            seq_len=self.seq_len,
            device=self.device,
        )


def dump_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
