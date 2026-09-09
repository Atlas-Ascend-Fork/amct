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

import gc
import math
import weakref

import pytest
import torch

from amct_pytorch.common.evaluate import eval_ppl


def _make_perfect_preds(samples, vocab_size):
    """Build logits that put all probability mass on the next-token target."""
    preds = []
    for s in samples:
        labels = s[:, 1:]  # [bs, seq_len-1]
        bs, seq_len = labels.shape
        logits = torch.full((bs, seq_len, vocab_size), -1e4)
        logits.scatter_(-1, labels.unsqueeze(-1), 1e4)
        preds.append(logits)
    return preds


def test_wikitext2_ppl_returns_one_for_perfect_predictions():
    vocab = 16
    seq_len = 8
    samples = [torch.randint(0, vocab, (1, seq_len)) for _ in range(3)]
    preds = _make_perfect_preds(samples, vocab)
    ppl = eval_ppl.wikitext2_ppl(preds, samples, device="cpu", seq_len=seq_len)
    assert ppl == pytest.approx(1.0, abs=1e-3)


def test_wikitext2_ppl_returns_uniform_value_for_uniform_logits():
    vocab = 16
    seq_len = 8
    samples = [torch.randint(0, vocab, (1, seq_len)) for _ in range(2)]
    # Uniform logits → CE = log(vocab) → PPL = vocab.
    preds = [torch.zeros(1, seq_len - 1, vocab) for _ in samples]
    ppl = eval_ppl.wikitext2_ppl(preds, samples, device="cpu", seq_len=seq_len)
    assert ppl == pytest.approx(vocab, rel=1e-3)


def test_wikitext2_ppl_returns_python_float():
    vocab = 8
    seq_len = 4
    samples = [torch.randint(0, vocab, (1, seq_len))]
    preds = [torch.zeros(1, seq_len - 1, vocab)]
    ppl = eval_ppl.wikitext2_ppl(preds, samples, device="cpu", seq_len=seq_len)
    assert isinstance(ppl, float)
    assert math.isfinite(ppl)


def test_wikitext2_ppl_matches_reference_on_cpu():
    seq_len = 8
    samples = [torch.randint(0, 16, (1, seq_len)) for _ in range(3)]
    logits = [torch.randn(1, seq_len - 1, 16) for _ in samples]
    losses = [
        torch.nn.functional.cross_entropy(
            item.reshape(-1, item.size(-1)),
            sample[:, 1:].reshape(-1),
        ).float()
        * seq_len
        for item, sample in zip(logits, samples)
    ]
    expected = torch.exp(torch.stack(losses).sum() / (len(samples) * seq_len))

    actual = eval_ppl.wikitext2_ppl(
        iter(logits),
        samples,
        device="cpu",
        seq_len=seq_len,
    )

    assert actual == pytest.approx(expected.item(), rel=1e-6)
    assert isinstance(actual, float)


def test_wikitext2_ppl_preserves_logits_dtype_and_uses_long_labels(monkeypatch):
    observed = {}

    class SpyLoss(torch.nn.Module):
        def forward(self, logits, labels):
            observed.update(logits=logits.dtype, labels=labels.dtype)
            return torch.nn.functional.cross_entropy(logits, labels)

    monkeypatch.setattr(eval_ppl.nn, "CrossEntropyLoss", lambda: SpyLoss())
    samples = [torch.tensor([[0, 1, 2, 3]])]
    logits = [torch.randn(1, 3, 8, dtype=torch.float64)]

    eval_ppl.wikitext2_ppl(logits, samples, "cpu", seq_len=4)

    assert observed == {"logits": torch.float64, "labels": torch.long}


def test_wikitext2_ppl_accumulates_fp32_nll(monkeypatch):
    observed = {}
    real_exp = torch.exp

    def spy_exp(value):
        observed["dtype"] = value.dtype
        return real_exp(value)

    monkeypatch.setattr(eval_ppl.torch, "exp", spy_exp)

    eval_ppl.wikitext2_ppl(
        [torch.zeros(1, 3, 8, dtype=torch.float64)],
        [torch.zeros(1, 4, dtype=torch.long)],
        "cpu",
        seq_len=4,
    )

    assert observed["dtype"] == torch.float32


def test_wikitext2_ppl_rejects_empty_samples():
    with pytest.raises(ValueError, match="must not be empty"):
        eval_ppl.wikitext2_ppl([], [], device="cpu")


def test_wikitext2_ppl_rejects_too_few_logits():
    samples = [torch.zeros(1, 4, dtype=torch.long) for _ in range(2)]
    with pytest.raises(ValueError, match="does not match"):
        eval_ppl.wikitext2_ppl([torch.zeros(1, 3, 8)], samples, device="cpu", seq_len=4)


def test_wikitext2_ppl_rejects_too_many_logits():
    samples = [torch.zeros(1, 4, dtype=torch.long)]
    logits = [torch.zeros(1, 3, 8), torch.zeros(1, 3, 8)]
    with pytest.raises(ValueError, match="exceeds samples count"):
        eval_ppl.wikitext2_ppl(logits, samples, "cpu", seq_len=4)


def test_wikitext2_ppl_rejects_device_mismatch(monkeypatch):
    monkeypatch.setattr(eval_ppl, "_same_device", lambda actual, expected: False)
    with pytest.raises(ValueError, match="expected device"):
        eval_ppl.wikitext2_ppl(
            [torch.zeros(1, 3, 8)],
            [torch.zeros(1, 4, dtype=torch.long)],
            device="npu:3",
            seq_len=4,
        )


def test_wikitext2_ppl_owns_single_progress_bar_and_logs_final_ppl(monkeypatch):
    observed = {"entered": 0, "exited": 0}

    class FakeProgress:
        def __init__(self, total, desc):
            observed.update(total=total, desc=desc, updates=0)

        def __enter__(self):
            observed["entered"] += 1
            return self

        @staticmethod
        def __exit__(exc_type, exc, traceback):
            observed["exited"] += 1

        @staticmethod
        def update(amount=1):
            observed["updates"] += amount

    monkeypatch.setattr(eval_ppl, "tqdm", FakeProgress)
    info = []
    monkeypatch.setattr(
        eval_ppl.logger,
        "info",
        lambda message, *args: info.append((message, args)),
    )
    samples = [torch.zeros(1, 4, dtype=torch.long) for _ in range(2)]
    logits = [torch.zeros(1, 3, 8) for _ in samples]

    ppl = eval_ppl.wikitext2_ppl(logits, samples, device="cpu", seq_len=4)

    assert ppl == pytest.approx(8.0, rel=1e-3)
    assert observed == {
        "entered": 1,
        "exited": 1,
        "total": 2,
        "desc": "PPL Evaluating...",
        "updates": 2,
    }
    assert len(info) == 1
    assert "PPL evaluation completed" in info[0][0]


def test_wikitext2_ppl_releases_logits_before_requesting_next_item():
    released = []

    def logits_iter():
        logits = torch.zeros(1, 3, 8)
        logits_ref = weakref.ref(logits)
        yield logits
        del logits
        gc.collect()
        released.append(logits_ref() is None)
        yield torch.zeros(1, 3, 8)

    samples = [torch.zeros(1, 4, dtype=torch.long) for _ in range(2)]
    ppl = eval_ppl.wikitext2_ppl(logits_iter(), samples, device="cpu", seq_len=4)

    assert ppl == pytest.approx(8.0, rel=1e-3)
    assert released == [True]


def test_wikitext2_ppl_cpu_does_not_touch_npu_cache(monkeypatch):
    fake_npu = type(
        "FakeNpu",
        (),
        {"empty_cache": staticmethod(lambda: pytest.fail("unexpected NPU call"))},
    )()
    monkeypatch.setattr(torch, "npu", fake_npu, raising=False)

    ppl = eval_ppl.wikitext2_ppl(
        [torch.zeros(1, 3, 8)],
        [torch.zeros(1, 4, dtype=torch.long)],
        device="cpu",
        seq_len=4,
    )

    assert ppl == pytest.approx(8.0, rel=1e-3)


def test_cleanup_ppl_resources_closes_iterator_before_emptying_cache(monkeypatch):
    events = []

    class TrackingIterator:
        def __init__(self):
            self.remaining = 1

        def __iter__(self):
            return self

        def __next__(self):
            if self.remaining == 0:
                raise StopIteration
            self.remaining -= 1
            return torch.zeros(1, 3, 8)

        @staticmethod
        def close():
            events.append(("close", None))

    fake_npu = type(
        "FakeNpu",
        (),
        {"empty_cache": staticmethod(lambda: events.append("empty_cache"))},
    )()
    monkeypatch.setattr(torch, "npu", fake_npu, raising=False)

    eval_ppl._cleanup_ppl_resources(  # pylint: disable=protected-access
        TrackingIterator(), "npu:1"
    )

    assert events == [("close", None), "empty_cache"]


def test_eval_ppl_does_not_expose_empty_npu_cache_helper():
    assert not hasattr(eval_ppl, "_empty_npu_cache")


def test_cleanup_ppl_resources_propagates_close_error(monkeypatch):
    events = []

    class BrokenIterator:
        @staticmethod
        def close():
            events.append("close")
            raise RuntimeError("close failed")

    fake_npu = type(
        "FakeNpu",
        (),
        {"empty_cache": staticmethod(lambda: events.append("empty_cache"))},
    )()
    monkeypatch.setattr(torch, "npu", fake_npu, raising=False)

    with pytest.raises(RuntimeError, match="close failed"):
        eval_ppl._cleanup_ppl_resources(  # pylint: disable=protected-access
            BrokenIterator(), "npu:1"
        )

    assert events == ["close"]
