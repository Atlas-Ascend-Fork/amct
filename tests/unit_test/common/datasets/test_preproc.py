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

import pytest
import torch

from amct_pytorch.common.datasets import preproc


class _StubTokenizer:
    """Whitespace tokenizer: each word -> id = ord(first char)."""

    def __call__(self, text, return_tensors=None):
        ids = self.encode(text)
        return type("Enc", (), {"input_ids": torch.tensor([ids])})()

    @staticmethod
    def encode(text):
        return [ord(t[0]) for t in text.split() if t]


class _FakeDataset(list):
    """Iterable with a deterministic shuffle so we control sample order."""

    def shuffle(self, seed):
        return self


# ---- pileval_awq ---------------------------------------------------------


def _texts_to_dataset(texts):
    return _FakeDataset({"text": t} for t in texts)


def test_pileval_awq_collects_n_samples_when_total_tokens_sufficient():
    # Each "a a a a" -> 4 tokens. seq_len=4 -> each sample = one row of 4 tokens.
    ds = _texts_to_dataset(["a a a a", "b b b b", "c c c c"])
    samples = preproc.pileval_awq(ds, _StubTokenizer(), n_samples=3, seq_len=4)
    assert len(samples) == 3
    assert all(s.shape == (1, 4) for s in samples)


def test_pileval_awq_skips_oversize_lines():
    # 5-token line should be skipped when seq_len=4; a 4-token line then satisfies.
    ds = _texts_to_dataset(["a a a a a", "b b b b", "c c c c"])
    samples = preproc.pileval_awq(ds, _StubTokenizer(), n_samples=2, seq_len=4)
    assert len(samples) == 2


def test_pileval_awq_raises_when_no_valid_samples():
    # All lines exceed seq_len.
    ds = _texts_to_dataset(["a a a a a a"] * 4)
    with pytest.raises(ValueError, match="No valid pileval samples"):
        preproc.pileval_awq(ds, _StubTokenizer(), n_samples=1, seq_len=2)


def test_pileval_awq_raises_when_not_enough_tokens_for_n_samples():
    # 2 lines × 2 tokens = 4 tokens, but n_samples=2 × seq_len=4 needs 8.
    ds = _texts_to_dataset(["a a", "b b"])
    with pytest.raises(ValueError, match="Not enough pileval tokens"):
        preproc.pileval_awq(ds, _StubTokenizer(), n_samples=2, seq_len=4)


def test_pileval_awq_skips_empty_encoded_samples():
    class _SelectiveTokenizer:
        @staticmethod
        def encode(text):
            if text == "skip me":
                return []
            return [ord(c) for c in text.split()]
    ds = _texts_to_dataset(["skip me", "a a a a a a a a", "b b b b b b b b"])
    samples = preproc.pileval_awq(ds, _SelectiveTokenizer(), n_samples=1, seq_len=8)
    assert len(samples) == 1


# ---- get_pileval / get_wikitext2 (load_dataset is mocked) -----------------


def test_get_pileval_passes_through_to_pileval_awq(monkeypatch):
    captured = {}

    def fake_load_dataset(name, *args, **kwargs):
        captured["name"] = name
        captured["split"] = kwargs.get("split") or (args[1] if len(args) >= 2 else None)
        return _texts_to_dataset(["x x x x"] * 4)

    monkeypatch.setattr(preproc, "load_dataset", fake_load_dataset)
    samples = preproc.get_pileval(_StubTokenizer(), n_samples=2, seq_len=4)
    assert captured["name"] == "mit-han-lab/pile-val-backup"
    assert captured["split"] == "validation"
    assert len(samples) == 2


def test_get_wikitext2_concatenates_and_tokenizes(monkeypatch):
    monkeypatch.setattr(
        preproc,
        "load_dataset",
        lambda *a, **k: {"text": ["hello world", "foo"]},
    )

    class _Tok:
        def __call__(self, text, return_tensors=None):
            assert "\n\n" in text  # confirms concatenation
            return type("Enc", (), {"input_ids": torch.tensor([[1, 2, 3]])})()

    enc = preproc.get_wikitext2(_Tok())
    assert torch.equal(enc.input_ids, torch.tensor([[1, 2, 3]]))


# ---- get_wiki_inputs ------------------------------------------------------


def test_get_wiki_inputs_chunks_into_seq_len_pieces(monkeypatch):
    # 10 tokens, seq_len=4 -> 2 full chunks (10 // 4 = 2).
    fake_enc = type("Enc", (), {"input_ids": torch.arange(10).unsqueeze(0)})()
    monkeypatch.setattr(preproc, "get_wikitext2", lambda tokenizer: fake_enc)

    chunks = preproc.get_wiki_inputs(tokenizer=None, seq_len=4)
    assert len(chunks) == 2
    assert chunks[0].shape == (1, 4)
    assert torch.equal(chunks[0], torch.tensor([[0, 1, 2, 3]]))
    assert torch.equal(chunks[1], torch.tensor([[4, 5, 6, 7]]))


def test_get_wiki_inputs_returns_empty_when_seq_len_exceeds_tokens(monkeypatch):
    fake_enc = type("Enc", (), {"input_ids": torch.arange(3).unsqueeze(0)})()
    monkeypatch.setattr(preproc, "get_wikitext2", lambda tokenizer: fake_enc)
    chunks = preproc.get_wiki_inputs(tokenizer=None, seq_len=8)
    assert not chunks
