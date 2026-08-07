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


def test_streaming_ppl_matches_legacy_on_cpu():
    samples = [torch.randint(0, 16, (1, 8)) for _ in range(3)]
    preds = [torch.randn(1, 7, 16) for _ in samples]
    expected = eval_ppl.wikitext2_ppl(preds, samples, seq_len=8)

    actual = eval_ppl.wikitext2_ppl_streaming(
        iter(preds), samples, device="cpu", seq_len=8
    )

    assert actual == pytest.approx(expected, rel=1e-6)


def test_streaming_ppl_preserves_logits_dtype_and_uses_long_labels(monkeypatch):
    observed = {}

    class SpyLoss(torch.nn.Module):
        def forward(self, logits, labels):
            observed.update(logits=logits.dtype, labels=labels.dtype)
            return torch.nn.functional.cross_entropy(logits, labels)

    monkeypatch.setattr(eval_ppl.nn, "CrossEntropyLoss", lambda: SpyLoss())
    samples = [torch.tensor([[0, 1, 2, 3]])]
    logits = [torch.randn(1, 3, 8, dtype=torch.float64)]

    eval_ppl.wikitext2_ppl_streaming(logits, samples, "cpu", seq_len=4)

    assert observed == {"logits": torch.float64, "labels": torch.long}


def test_streaming_ppl_accumulates_fp32_nll(monkeypatch):
    observed = {}
    real_exp = torch.exp

    def spy_exp(value):
        observed["dtype"] = value.dtype
        return real_exp(value)

    monkeypatch.setattr(eval_ppl.torch, "exp", spy_exp)

    eval_ppl.wikitext2_ppl_streaming(
        [torch.zeros(1, 3, 8, dtype=torch.float64)],
        [torch.zeros(1, 4, dtype=torch.long)],
        "cpu",
        seq_len=4,
    )

    assert observed["dtype"] == torch.float32


def test_streaming_ppl_rejects_empty_samples():
    with pytest.raises(ValueError, match="must not be empty"):
        eval_ppl.wikitext2_ppl_streaming([], [], device="cpu")


def test_streaming_ppl_rejects_too_few_logits():
    samples = [torch.zeros(1, 4, dtype=torch.long) for _ in range(2)]
    with pytest.raises(ValueError, match="does not match"):
        eval_ppl.wikitext2_ppl_streaming(
            [torch.zeros(1, 3, 8)], samples, device="cpu", seq_len=4
        )


def test_streaming_ppl_rejects_too_many_logits():
    samples = [torch.zeros(1, 4, dtype=torch.long)]
    logits = [torch.zeros(1, 3, 8), torch.zeros(1, 3, 8)]
    with pytest.raises(ValueError, match="exceeds samples count"):
        eval_ppl.wikitext2_ppl_streaming(logits, samples, "cpu", seq_len=4)


def test_streaming_ppl_rejects_device_mismatch(monkeypatch):
    monkeypatch.setattr(eval_ppl, "_same_device", lambda actual, expected: False)
    with pytest.raises(ValueError, match="expected device"):
        eval_ppl.wikitext2_ppl_streaming(
            [torch.zeros(1, 3, 8)],
            [torch.zeros(1, 4, dtype=torch.long)],
            device="npu:3",
            seq_len=4,
        )


def test_streaming_ppl_buffers_memory_logs_until_ppl_item(monkeypatch):
    events = []
    monkeypatch.setattr(eval_ppl, "_is_npu_device", lambda device: True)
    monkeypatch.setattr(
        eval_ppl,
        "_reset_npu_peak",
        lambda device: events.append("peak-reset"),
    )
    monkeypatch.setattr(
        eval_ppl,
        "_read_npu_memory",
        lambda device: {
            "allocated": 1,
            "reserved": 2,
            "peak_allocated": 3,
            "peak_reserved": 4,
        },
    )
    monkeypatch.setattr(
        eval_ppl.logger,
        "info",
        lambda message, *args: events.append(("log", message, args)),
    )

    class PplValue:
        def item(self):
            events.append("ppl-item")
            return 8.0

    monkeypatch.setattr(eval_ppl.torch, "exp", lambda value: PplValue())

    def logits_iter():
        events.append("yield-1")
        yield torch.zeros(1, 3, 8)
        events.append("yield-2")
        yield torch.zeros(1, 3, 8)
        events.append("iterator-finished")

    samples = [torch.zeros(1, 4, dtype=torch.long) for _ in range(2)]
    ppl = eval_ppl.wikitext2_ppl_streaming(
        logits_iter(), samples, device="cpu", seq_len=4
    )

    first_log = next(
        i
        for i, event in enumerate(events)
        if isinstance(event, tuple) and event[0] == "log"
    )
    sample_logs = [
        event
        for event in events
        if isinstance(event, tuple) and "streaming_ppl_sample" in event[1]
    ]
    assert ppl == 8.0
    assert events.index("peak-reset") < events.index("yield-1")
    assert events.index("iterator-finished") < events.index("ppl-item") < first_log
    assert len(sample_logs) == 2


def test_streaming_ppl_disables_monitor_after_query_failure(monkeypatch):
    warnings = []
    queries = []
    monkeypatch.setattr(eval_ppl, "_is_npu_device", lambda device: True)
    monkeypatch.setattr(eval_ppl, "_reset_npu_peak", lambda device: None)

    def fail_stats(device):
        queries.append(device)
        raise RuntimeError("stats unavailable")

    monkeypatch.setattr(eval_ppl, "_read_npu_memory", fail_stats)
    monkeypatch.setattr(
        eval_ppl.logger,
        "warning",
        lambda message, *args: warnings.append((message, args)),
    )
    ppl = eval_ppl.wikitext2_ppl_streaming(
        [torch.zeros(1, 3, 8), torch.zeros(1, 3, 8)],
        [torch.zeros(1, 4, dtype=torch.long) for _ in range(2)],
        device="cpu",
        seq_len=4,
    )

    assert ppl == pytest.approx(8.0, rel=1e-3)
    assert len(queries) == 1
    assert len(warnings) == 1


def test_streaming_ppl_marks_peak_unavailable_after_partial_query_failure(
    monkeypatch,
):
    queries = []
    info_logs = []
    monkeypatch.setattr(eval_ppl, "_is_npu_device", lambda device: True)
    monkeypatch.setattr(eval_ppl, "_reset_npu_peak", lambda device: None)

    def read_then_fail(device):
        queries.append(device)
        if len(queries) == 1:
            return {
                "allocated": 1,
                "reserved": 2,
                "peak_allocated": 3,
                "peak_reserved": 4,
            }
        raise RuntimeError("stats became unavailable")

    monkeypatch.setattr(eval_ppl, "_read_npu_memory", read_then_fail)
    monkeypatch.setattr(eval_ppl.logger, "warning", lambda *args: None)
    monkeypatch.setattr(
        eval_ppl.logger,
        "info",
        lambda message, *args: info_logs.append((message, args)),
    )

    ppl = eval_ppl.wikitext2_ppl_streaming(
        [torch.zeros(1, 3, 8), torch.zeros(1, 3, 8)],
        [torch.zeros(1, 4, dtype=torch.long) for _ in range(2)],
        device="cpu",
        seq_len=4,
    )

    completion = next(
        args for message, args in info_logs if "streaming_ppl_complete" in message
    )
    assert ppl == pytest.approx(8.0, rel=1e-3)
    assert len(queries) == 2
    assert completion[-2:] == ("n/a", "n/a")


def test_streaming_ppl_disables_monitor_when_peak_reset_fails(monkeypatch):
    warnings = []
    monkeypatch.setattr(eval_ppl, "_is_npu_device", lambda device: True)
    monkeypatch.setattr(
        eval_ppl,
        "_reset_npu_peak",
        lambda device: (_ for _ in ()).throw(RuntimeError("reset unavailable")),
    )
    monkeypatch.setattr(
        eval_ppl,
        "_read_npu_memory",
        lambda device: pytest.fail("memory query must stay disabled"),
    )
    monkeypatch.setattr(
        eval_ppl.logger,
        "warning",
        lambda message, *args: warnings.append((message, args)),
    )

    ppl = eval_ppl.wikitext2_ppl_streaming(
        [torch.zeros(1, 3, 8)],
        [torch.zeros(1, 4, dtype=torch.long)],
        device="cpu",
        seq_len=4,
    )

    assert ppl == pytest.approx(8.0, rel=1e-3)
    assert len(warnings) == 1


def test_streaming_ppl_memory_log_failure_does_not_change_result(monkeypatch):
    warnings = []
    monkeypatch.setattr(eval_ppl, "_is_npu_device", lambda device: True)
    monkeypatch.setattr(eval_ppl, "_reset_npu_peak", lambda device: None)
    monkeypatch.setattr(
        eval_ppl,
        "_read_npu_memory",
        lambda device: {
            "allocated": 1,
            "reserved": 2,
            "peak_allocated": 3,
            "peak_reserved": 4,
        },
    )
    monkeypatch.setattr(
        eval_ppl.logger,
        "info",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("log failed")),
    )
    monkeypatch.setattr(
        eval_ppl.logger,
        "warning",
        lambda message, *args: warnings.append((message, args)),
    )

    ppl = eval_ppl.wikitext2_ppl_streaming(
        [torch.zeros(1, 3, 8)],
        [torch.zeros(1, 4, dtype=torch.long)],
        device="cpu",
        seq_len=4,
    )

    assert ppl == pytest.approx(8.0, rel=1e-3)
    assert warnings


def test_streaming_ppl_closes_generator_on_loss_error(monkeypatch):
    closed = []

    def logits_iter():
        try:
            yield torch.zeros(1, 3, 8)
        finally:
            closed.append(True)

    class BrokenLoss(torch.nn.Module):
        def forward(self, logits, labels):
            raise RuntimeError("loss failed")

    monkeypatch.setattr(eval_ppl.nn, "CrossEntropyLoss", lambda: BrokenLoss())
    with pytest.raises(RuntimeError, match="loss failed"):
        eval_ppl.wikitext2_ppl_streaming(
            logits_iter(),
            [torch.zeros(1, 4, dtype=torch.long)],
            device="cpu",
            seq_len=4,
        )
    assert closed == [True]


def test_streaming_ppl_cleanup_error_does_not_replace_loss_error(monkeypatch):
    class BrokenIterator:
        def __iter__(self):
            return self

        def __next__(self):
            return torch.zeros(1, 3, 8)

        def close(self):
            raise RuntimeError("close failed")

    class BrokenLoss(torch.nn.Module):
        def forward(self, logits, labels):
            raise ValueError("loss failed")

    monkeypatch.setattr(eval_ppl.nn, "CrossEntropyLoss", lambda: BrokenLoss())
    monkeypatch.setattr(
        eval_ppl,
        "_empty_npu_cache",
        lambda device: (_ for _ in ()).throw(RuntimeError("cache failed")),
    )
    with pytest.raises(ValueError, match="loss failed"):
        eval_ppl.wikitext2_ppl_streaming(
            BrokenIterator(),
            [torch.zeros(1, 4, dtype=torch.long)],
            "cpu",
            seq_len=4,
        )


def test_streaming_ppl_close_lookup_error_does_not_replace_loss_error(monkeypatch):
    class BrokenCloseLookupIterator:
        def __iter__(self):
            return self

        def __next__(self):
            return torch.zeros(1, 3, 8)

        def __getattribute__(self, name):
            if name == "close":
                raise RuntimeError("close lookup failed")
            return object.__getattribute__(self, name)

    class BrokenLoss(torch.nn.Module):
        def forward(self, logits, labels):
            raise ValueError("business loss failed")

    monkeypatch.setattr(eval_ppl.nn, "CrossEntropyLoss", lambda: BrokenLoss())
    with pytest.raises(ValueError, match="business loss failed"):
        eval_ppl.wikitext2_ppl_streaming(
            BrokenCloseLookupIterator(),
            [torch.zeros(1, 4, dtype=torch.long)],
            "cpu",
            seq_len=4,
        )
