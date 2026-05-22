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
"""Logic tests for LlmExtractPtqDataWorkflow.

`run` requires a real tokenizer + pipeline; we cover the input validation and
hook-name selection here.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from amct_pytorch.workflows.llm_extract_ptq_data import LlmExtractPtqDataWorkflow

INPUT_LAYERNORM = 'input_layernorm'

POST_ATTENTION_LAYERNORM = 'post_attention_layernorm'


def _make_args(quant_target):
    return SimpleNamespace(
        quant_target=list(quant_target),
        seq_len=512,
        nsamples=8,
        device="cpu",
        model_name="qwen3",
        granularity="block",
    )


def test_init_rejects_no_quant_target():
    with pytest.raises(ValueError, match="single quant_target"):
        LlmExtractPtqDataWorkflow(_make_args([]))


def test_init_rejects_multiple_quant_targets():
    with pytest.raises(ValueError, match="single quant_target"):
        LlmExtractPtqDataWorkflow(_make_args(["mlp", "attn-linear"]))


def test_init_unwraps_single_target_to_string():
    wf = LlmExtractPtqDataWorkflow(_make_args(["mlp"]))
    assert wf.quant_target == "mlp"
    assert wf.seq_len == 512
    assert wf.nsamples == 8


@pytest.mark.parametrize(
    "target,expected_hook",
    [
        ("attn", INPUT_LAYERNORM),
        ("attn-linear", INPUT_LAYERNORM),
        ("attn-cache", INPUT_LAYERNORM),
        ("mlp", POST_ATTENTION_LAYERNORM),
        ("moe", POST_ATTENTION_LAYERNORM),
    ],
)
def test_run_blockwise_picks_hook_name_by_quant_target(monkeypatch, target, expected_hook):
    """Hijack pipeline + samples; verify the hook_name selected matches the target class."""
    wf = LlmExtractPtqDataWorkflow(_make_args([target]))

    seen = {}

    class _FakePipeline:
        num_layers = 0

        def __init__(self):
            self.tokenizer = "tokenizer"

        @staticmethod
        def do_embedding_forward(samples, hook_name):
            seen["embed_hook"] = hook_name
            return []

        @staticmethod
        def do_block_forward(layer_idx, inter_io, hook_name):
            return inter_io

    wf.pipeline = _FakePipeline()
    monkeypatch.setattr(
        "amct_pytorch.workflows.llm_extract_ptq_data.get_pileval",
        lambda tokenizer, n, seq_len: ["s"] * 2,
    )
    wf._run_blockwise()
    assert seen["embed_hook"] == expected_hook


def _make_extract_workflow(**overrides):
    defaults = dict(
        model="/tmp/fake", model_name="qwen3", quant_target=["mlp"],
        device="cpu", seq_len=2048, nsamples=32, output_dir="/tmp/fake",
        granularity="block",
    )
    defaults.update(overrides)
    args = SimpleNamespace(**defaults)
    wf = LlmExtractPtqDataWorkflow.__new__(LlmExtractPtqDataWorkflow)
    for k, v in vars(args).items():
        setattr(wf, k, v)
    wf.args = args
    return wf


def test_extract_init_rejects_multiple_quant_targets():
    with pytest.raises(ValueError, match="only supports a single quant_target"):
        LlmExtractPtqDataWorkflow(_make_args(["mlp", "attn-linear"]))


def test_extract_init_accepts_single_quant_target():
    wf = LlmExtractPtqDataWorkflow(_make_args(["mlp"]))
    assert wf.quant_target == "mlp"


def test_extract_setup_returns_sink_id(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "amct_pytorch.workflows.llm_extract_ptq_data.register_llm_models", lambda: None)
    monkeypatch.setattr(
        "amct_pytorch.workflows.llm_extract_ptq_data.MODEL_REGISTRY",
        SimpleNamespace(get=lambda k: type("FM", (), {"__init__": lambda s, a: None})),
    )
    wf = _make_extract_workflow(output_dir=str(tmp_path))
    sink_id = wf.setup()
    assert sink_id is not None
    assert wf.pipeline is not None


def test_run_completes(monkeypatch):
    monkeypatch.setattr(
        "amct_pytorch.workflows.llm_extract_ptq_data.register_llm_models", lambda: None)

    class FakePipeline:
        num_layers = 1
        tokenizer = MagicMock()

        def __init__(self, args):
            pass

        @staticmethod
        def do_embedding_forward(samples, hook_name):
            return []

        @staticmethod
        def do_block_forward(layer_idx, inter_io, hook_name):
            return []

    monkeypatch.setattr(
        "amct_pytorch.workflows.llm_extract_ptq_data.MODEL_REGISTRY",
        SimpleNamespace(get=lambda k: FakePipeline),
    )
    monkeypatch.setattr(
        "amct_pytorch.workflows.llm_extract_ptq_data.get_pileval",
        lambda tokenizer, n, seq_len: [],
    )
    monkeypatch.setattr(
        "amct_pytorch.workflows.llm_extract_ptq_data.logger",
        SimpleNamespace(remove=lambda h: None, info=lambda *a, **kw: None),
    )
    wf = _make_extract_workflow(quant_target=["mlp"])
    wf.args.output_dir = "/tmp"
    wf.run()

