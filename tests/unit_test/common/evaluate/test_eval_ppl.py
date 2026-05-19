#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
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

import math

import pytest
import torch

from amct_pytorch.common.evaluate import eval_ppl


@pytest.fixture(autouse=True)
def _stub_npu_empty_cache(monkeypatch):
    # wikitext2_ppl calls torch.npu.empty_cache() which is unavailable on CPU CI.
    fake_npu = type("F", (), {"empty_cache": staticmethod(lambda: None)})()
    monkeypatch.setattr(torch, "npu", fake_npu, raising=False)


def _make_perfect_preds(samples, vocab_size):
    """Build logits that put all probability mass on the next-token target."""
    preds = []
    for s in samples:
        labels = s[:, 1:]                       # [bs, seq_len-1]
        bs, l = labels.shape
        logits = torch.full((bs, l, vocab_size), -1e4)
        logits.scatter_(-1, labels.unsqueeze(-1), 1e4)
        preds.append(logits)
    return preds


def test_wikitext2_ppl_returns_one_for_perfect_predictions():
    vocab = 16
    seq_len = 8
    samples = [torch.randint(0, vocab, (1, seq_len)) for _ in range(3)]
    preds = _make_perfect_preds(samples, vocab)
    ppl = eval_ppl.wikitext2_ppl(preds, samples, seq_len=seq_len)
    assert ppl == pytest.approx(1.0, abs=1e-3)


def test_wikitext2_ppl_returns_uniform_value_for_uniform_logits():
    vocab = 16
    seq_len = 8
    samples = [torch.randint(0, vocab, (1, seq_len)) for _ in range(2)]
    # Uniform logits → CE = log(vocab) → PPL = vocab.
    preds = [torch.zeros(1, seq_len - 1, vocab) for _ in samples]
    ppl = eval_ppl.wikitext2_ppl(preds, samples, seq_len=seq_len)
    assert ppl == pytest.approx(vocab, rel=1e-3)


def test_wikitext2_ppl_returns_python_float():
    vocab = 8
    seq_len = 4
    samples = [torch.randint(0, vocab, (1, seq_len))]
    preds = [torch.zeros(1, seq_len - 1, vocab)]
    ppl = eval_ppl.wikitext2_ppl(preds, samples, seq_len=seq_len)
    assert isinstance(ppl, float)
    assert math.isfinite(ppl)
