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
"""Logic tests for LlmEvalWorkflow.

The end-to-end blockwise / modelwise loops require a real model checkpoint and
NPU device, so we cover only the pure decision logic here:
``_resolve_eval_states`` and ``_has_relevant_quant``.
"""

import gc
import importlib
from types import SimpleNamespace
from unittest.mock import MagicMock
import weakref

import pytest
import torch

from amct_pytorch.quantization.bit_policy import BitPolicy
from amct_pytorch.workflows.llm_eval import LlmEvalWorkflow


def _make_bit_policy(**overrides):
    cfg = {"w_bits": 16, "a_bits": 16}
    cfg.update(overrides)
    return BitPolicy(cfg)


def _make_workflow(eval_mode="bf16", quant_target=(), bit_policy=None):
    """Build a workflow without invoking the heavy __init__ path."""
    workflow = LlmEvalWorkflow.__new__(LlmEvalWorkflow)
    if bit_policy is None:
        bit_policy = _make_bit_policy()
    args = SimpleNamespace(
        eval_mode=eval_mode,
        quant_target=list(quant_target),
        bit_policy=bit_policy,
        quant_dtype="int",
    )
    workflow.args = args
    workflow.eval_mode = args.eval_mode
    workflow.model_name = getattr(args, "model_name", "test_model")
    workflow.quant_target = args.quant_target
    workflow.quant_dtype = "int"
    workflow.bit_policy = bit_policy
    workflow.device = getattr(args, "device", "cpu")
    workflow.granularity = getattr(args, "granularity", "model")
    workflow.seq_len = getattr(args, "seq_len", 2048)
    workflow.pipeline = None
    workflow.data_provider = None
    return workflow


# ---- _has_relevant_quant --------------------------------------------------


def test_relevant_bits_empty_when_no_quant_target():
    assert _make_workflow()._has_relevant_quant() is False


@pytest.mark.parametrize("target", ["mlp", "moe", "attn-linear"])
def test_relevant_bits_for_linear_targets_uses_bit_policy(target):
    wf = _make_workflow(
        quant_target=[target], bit_policy=_make_bit_policy(w_bits=8, a_bits=4)
    )
    assert wf._has_relevant_quant() is True


def test_relevant_bits_for_attn_cache_uses_bit_policy():
    wf = _make_workflow(
        quant_target=["attn-cache"],
        bit_policy=_make_bit_policy(**{"attn-cache": {"q": 8, "k": 8, "p": 4, "v": 4}}),
    )
    assert wf._has_relevant_quant() is True


def test_relevant_bits_combines_linear_and_cache_targets():
    wf = _make_workflow(
        quant_target=["mlp", "attn-cache"],
        bit_policy=_make_bit_policy(
            w_bits=4,
            a_bits=8,
            **{"attn-cache": {"q": 16, "k": 16, "p": 16, "v": 16}},
        ),
    )
    assert wf._has_relevant_quant() is True


# ---- _resolve_eval_states -------------------------------------------------


def test_resolve_eval_states_bf16_disables_quant():
    use_quant_block, enable_quant, msg = _make_workflow()._resolve_eval_states()
    assert use_quant_block is False
    assert enable_quant is False
    assert "bf16" in msg


def test_resolve_eval_states_quant_with_no_targets_disables_quant_but_rebuilds():
    wf = _make_workflow(eval_mode="quant", quant_target=[])
    use_quant_block, enable_quant, msg = wf._resolve_eval_states()
    assert use_quant_block is True
    assert enable_quant is False
    assert "disabled" in msg


def test_resolve_eval_states_quant_with_all_bits_16_skips_real_quant():
    wf = _make_workflow(
        eval_mode="quant",
        quant_target=["mlp"],
        bit_policy=_make_bit_policy(w_bits=16, a_bits=16),
    )
    use_quant_block, enable_quant, _ = wf._resolve_eval_states()
    assert use_quant_block is True
    assert enable_quant is False


def test_resolve_eval_states_quant_enables_when_any_relevant_bit_lt_16():
    wf = _make_workflow(
        eval_mode="quant",
        quant_target=["mlp"],
        bit_policy=_make_bit_policy(w_bits=8, a_bits=16),
    )
    use_quant_block, enable_quant, msg = wf._resolve_eval_states()
    assert use_quant_block is True
    assert enable_quant is True
    assert "enabled" in msg


def test_resolve_eval_states_ignores_irrelevant_low_bits_for_target():
    # quant_target=mlp uses only linear bits; attn-cache bits should NOT trigger enable.
    wf = _make_workflow(
        eval_mode="quant",
        quant_target=["mlp"],
        bit_policy=_make_bit_policy(
            w_bits=16,
            a_bits=16,
            **{"attn-cache": {"q": 4, "k": 4, "p": 4, "v": 4}},
        ),
    )
    _, enable_quant, _ = wf._resolve_eval_states()
    assert enable_quant is False


def test_resolve_eval_states_attn_cache_low_bit_triggers_enable():
    wf = _make_workflow(
        eval_mode="quant",
        quant_target=["attn-cache"],
        bit_policy=_make_bit_policy(
            **{"attn-cache": {"q": 8, "k": 16, "p": 16, "v": 16}}
        ),
    )
    _, enable_quant, _ = wf._resolve_eval_states()
    assert enable_quant is True


def test_resolve_eval_states_unknown_mode_falls_back_to_quant():
    wf = _make_workflow(
        eval_mode="unknown",
        quant_target=["mlp"],
        bit_policy=_make_bit_policy(w_bits=8, a_bits=8),
    )
    use_quant, enable, msg = wf._resolve_eval_states()
    assert use_quant is True
    assert enable is True


def test_llm_eval_run_blockwise(monkeypatch):
    wf = _make_workflow(
        eval_mode="quant",
        quant_target=["mlp"],
        bit_policy=_make_bit_policy(w_bits=8, a_bits=8),
    )
    wf.granularity = "block"

    def setup():
        return "sink"

    wf.setup = setup

    called = {}

    def _run_blockwise():
        called.update({"blockwise": 10.5})
        return 10.5

    wf._run_blockwise = _run_blockwise
    monkeypatch.setattr(
        "amct_pytorch.workflows.llm_eval.logger",
        importlib.import_module("types").SimpleNamespace(
            info=lambda m: None, remove=lambda h: None
        ),
    )
    wf.run()
    assert called.get("blockwise") == 10.5


def test_llm_eval_run_unknown_granularity(monkeypatch):
    wf = _make_workflow()
    wf.granularity = "unknown"

    def setup():
        return "sink"

    wf.setup = setup
    monkeypatch.setattr(
        "amct_pytorch.workflows.llm_eval.logger",
        importlib.import_module("types").SimpleNamespace(remove=lambda h: None),
    )
    with pytest.raises(ValueError, match="Unsupported .*granularity"):
        wf.run()


# ---- _make_eval_workflow helper -------------------------------------------


def _make_eval_workflow(**overrides):
    bit_policy = overrides.pop("bit_policy", _make_bit_policy(w_bits=8, a_bits=8))
    defaults = dict(
        model="/tmp/fake",
        model_name="qwen3",
        quant_target=["mlp"],
        device="cpu",
        granularity="block",
        eval_mode="bf16",
        seq_len=2048,
        output_dir="/tmp/fake",
        quant_dtype="int",
    )
    defaults.update(overrides)
    args = SimpleNamespace(bit_policy=bit_policy, **defaults)
    wf = LlmEvalWorkflow.__new__(LlmEvalWorkflow)
    for k, v in vars(args).items():
        setattr(wf, k, v)
    wf.args = args
    return wf


# ---- _has_relevant_quant (via _make_eval_workflow) ---------------------


def test_eval_get_relevant_quant_bits_mlp():
    wf = _make_eval_workflow(
        quant_target=["mlp"], bit_policy=_make_bit_policy(w_bits=4, a_bits=8)
    )
    assert wf._has_relevant_quant() is True


def test_eval_get_relevant_quant_bits_attn_cache():
    wf = _make_eval_workflow(
        quant_target=["attn-cache"],
        bit_policy=_make_bit_policy(**{"attn-cache": {"q": 4, "k": 4, "p": 8, "v": 8}}),
    )
    assert wf._has_relevant_quant() is True


def test_eval_get_relevant_quant_bits_both_targets():
    wf = _make_eval_workflow(
        quant_target=["mlp", "attn-cache"],
        bit_policy=_make_bit_policy(w_bits=8, a_bits=8),
    )
    assert wf._has_relevant_quant() is True


# ---- _resolve_eval_states (via _make_eval_workflow) -----------------------


def test_eval_resolve_eval_states_bf16():
    wf = _make_eval_workflow(eval_mode="bf16")
    use_quant, enable_quant, msg = wf._resolve_eval_states()
    assert use_quant is False
    assert enable_quant is False
    assert "BF16" in msg


def test_eval_resolve_eval_states_quant_all_16bit():
    wf = _make_eval_workflow(
        eval_mode="quant", bit_policy=_make_bit_policy(w_bits=16, a_bits=16)
    )
    use_quant, enable_quant, msg = wf._resolve_eval_states()
    assert use_quant is True
    assert enable_quant is False
    assert "disabled" in msg


def test_eval_resolve_eval_states_quant_below_16():
    wf = _make_eval_workflow(
        eval_mode="quant", bit_policy=_make_bit_policy(w_bits=8, a_bits=4)
    )
    use_quant, enable_quant, msg = wf._resolve_eval_states()
    assert use_quant is True
    assert enable_quant is True
    assert "enabled" in msg


# ---- setup ----------------------------------------------------------------


def test_eval_setup_returns_sink_id(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "amct_pytorch.workflows.llm_eval.register_llm_models", lambda: None
    )
    monkeypatch.setattr("amct_pytorch.workflows.llm_eval.register_dtype", lambda: None)
    monkeypatch.setattr(
        "amct_pytorch.workflows.llm_eval.register_algorithms", lambda: None
    )
    monkeypatch.setattr(
        "amct_pytorch.workflows.llm_eval.MODEL_REGISTRY",
        SimpleNamespace(
            get=lambda k: type("FakeModel", (), {"__init__": lambda s, a: None})
        ),
    )
    wf = _make_eval_workflow(output_dir=str(tmp_path))
    sink_id = wf.setup()
    assert sink_id is not None
    assert wf.pipeline is not None


# ---- _run_blockwise (mocked pipeline) ------------------------------------


def test_eval_run_blockwise_streams_head_logits_on_workflow_device(
    monkeypatch, tmp_path
):
    samples = [torch.randint(0, 100, (2, 4))]
    monkeypatch.setattr(
        "amct_pytorch.workflows.llm_eval.get_wiki_inputs",
        lambda tokenizer, seq_len: samples,
    )
    observed = {}

    def fake_ppl(logits_iter, actual_samples, device, seq_len):
        observed.update(
            logits=list(logits_iter),
            samples=actual_samples,
            device=device,
            seq_len=seq_len,
        )
        return 12.34

    monkeypatch.setattr(
        "amct_pytorch.workflows.llm_eval.wikitext2_ppl",
        fake_ppl,
        raising=False,
    )
    fake_logger = MagicMock()
    monkeypatch.setattr(
        "amct_pytorch.workflows.llm_eval.logger",
        fake_logger,
    )
    monkeypatch.setattr(
        "amct_pytorch.workflows.llm_eval.tqdm",
        lambda iterable, desc="": iterable,
    )

    wf = _make_eval_workflow(eval_mode="bf16", output_dir=str(tmp_path))
    wf.pipeline = MagicMock()
    wf.pipeline.tokenizer = MagicMock()
    wf.pipeline.num_layers = 2
    wf.pipeline.do_embedding_forward = MagicMock(return_value=[torch.randn(2, 4, 8)])
    wf.pipeline.do_block_forward = MagicMock(return_value=[torch.randn(2, 4, 8)])
    streamed = [torch.randn(2, 3, 100)]
    wf.pipeline.do_head_forward.return_value = iter(streamed)

    result = wf._run_blockwise()  # pylint: disable=protected-access

    assert result == pytest.approx(12.34)
    assert len(observed["logits"]) == 1
    assert torch.equal(observed["logits"][0], streamed[0])
    assert observed["samples"] is samples
    assert observed["device"] == "cpu"
    assert observed["seq_len"] == wf.seq_len
    wf.pipeline.do_head_forward.assert_called_once()
    messages = [call.args[0] for call in fake_logger.info.call_args_list]
    assert messages == [
        "Loaded {} eval samples for blockwise eval.",
        "Eval mode=bf16: use original BF16 blocks.",
    ]


# ---- _iter_model_logits --------------------------------------------------


def test_iter_model_logits_yields_shifted_logits_in_order_without_grad(monkeypatch):
    monkeypatch.setattr(
        "amct_pytorch.workflows.llm_eval.tqdm",
        lambda *args, **kwargs: pytest.fail(
            "_iter_model_logits must not own the PPL progress bar"
        ),
    )
    wf = _make_eval_workflow(eval_mode="bf16", granularity="model")
    wf.device = "cpu"
    samples = [
        torch.tensor([[0, 1, 2, 3]]),
        torch.tensor([[4, 5, 6, 7]]),
    ]
    raw_logits = [
        torch.arange(32, dtype=torch.float32).reshape(1, 4, 8),
        torch.arange(32, 64, dtype=torch.float32).reshape(1, 4, 8),
    ]
    observed = []

    class FakeModel:
        @staticmethod
        def __call__(sample):
            index = len(observed)
            observed.append((sample.clone(), torch.is_grad_enabled()))
            return SimpleNamespace(logits=raw_logits[index])

    actual = list(
        wf._iter_model_logits(FakeModel(), samples)  # pylint: disable=protected-access
    )

    expected = [logits[:, :-1, :].contiguous() for logits in raw_logits]
    assert len(actual) == len(expected)
    assert all(torch.equal(got, want) for got, want in zip(actual, expected))
    assert all(torch.equal(item[0], sample) for item, sample in zip(observed, samples))
    assert [item[1] for item in observed] == [False, False]


def test_iter_model_logits_releases_previous_tensors_before_next_forward(
    monkeypatch,
):
    monkeypatch.setattr(
        "amct_pytorch.workflows.llm_eval.tqdm",
        lambda *args, **kwargs: pytest.fail(
            "_iter_model_logits must not own the PPL progress bar"
        ),
    )
    wf = _make_eval_workflow(eval_mode="bf16", granularity="model")
    wf.device = "cpu"
    samples = [
        torch.zeros(2, 4, dtype=torch.long),
        torch.ones(2, 4, dtype=torch.long),
    ]
    refs = {"full": [], "previous_shift": None}

    class FakeModel:
        def __init__(self):
            self.calls = 0

        def __call__(self, sample):
            if self.calls == 1:
                gc.collect()
                assert refs["previous_shift"]() is None
            logits = torch.arange(64, dtype=torch.float32).reshape(2, 4, 8)
            logits = logits + self.calls
            refs["full"].append(weakref.ref(logits))
            self.calls += 1
            return SimpleNamespace(logits=logits)

    model = FakeModel()
    iterator = wf._iter_model_logits(  # pylint: disable=protected-access
        model, samples
    )
    first = next(iterator)
    gc.collect()
    assert refs["full"][0]() is None
    refs["previous_shift"] = weakref.ref(first)
    del first

    second = next(iterator)
    assert model.calls == 2
    del second
    iterator.close()


# ---- _run_modelwise (mocked pipeline) ------------------------------------


def test_eval_run_modelwise_streams_logits_on_workflow_device(monkeypatch):
    samples = [torch.tensor([[0, 1, 2, 3]])]
    raw_logits = torch.zeros(1, 4, 8)
    monkeypatch.setattr(
        "amct_pytorch.workflows.llm_eval.get_wiki_inputs",
        lambda tokenizer, seq_len: samples,
    )
    monkeypatch.setattr(
        "amct_pytorch.workflows.llm_eval.tqdm",
        lambda *args, **kwargs: pytest.fail(
            "_iter_model_logits must not own the PPL progress bar"
        ),
    )
    observed = {}

    def fake_ppl(logits_iter, actual_samples, device, seq_len):
        observed.update(
            logits=list(logits_iter),
            samples=actual_samples,
            device=device,
            seq_len=seq_len,
        )
        return 12.34

    monkeypatch.setattr(
        "amct_pytorch.workflows.llm_eval.wikitext2_ppl",
        fake_ppl,
        raising=False,
    )
    fake_logger = MagicMock()
    monkeypatch.setattr(
        "amct_pytorch.workflows.llm_eval.logger",
        fake_logger,
    )

    wf = _make_eval_workflow(eval_mode="bf16", granularity="model")
    wf.device = "cpu"
    wf.seq_len = 4
    wf.pipeline = MagicMock()
    wf.pipeline.tokenizer = MagicMock()
    forward_fn = MagicMock(return_value=SimpleNamespace(logits=raw_logits))
    wf.pipeline.float_model.return_value.eval.return_value.to.return_value = forward_fn

    result = wf._run_modelwise()  # pylint: disable=protected-access

    assert result == pytest.approx(12.34)
    assert observed["samples"] is samples
    assert observed["device"] == "cpu"
    assert observed["seq_len"] == 4
    assert len(observed["logits"]) == 1
    assert torch.equal(observed["logits"][0], raw_logits[:, :-1, :])
    forward_fn.assert_called_once()
    assert torch.equal(forward_fn.call_args.args[0], samples[0])
    messages = [call.args[0] for call in fake_logger.info.call_args_list]
    assert messages == ["Loaded {} eval samples for modelwise eval."]


# ---- __init__ -------------------------------------------------------------


def test_eval_init_sets_all_attributes_from_args():
    bp = _make_bit_policy(w_bits=8, a_bits=8)
    args = SimpleNamespace(
        model="/tmp/fake",
        model_name="qwen3",
        quant_target=["mlp", "attn-cache"],
        device="cuda:0",
        granularity="block",
        eval_mode="quant",
        seq_len=1024,
        output_dir="/tmp/out",
        quant_dtype="int4",
        bit_policy=bp,
    )
    wf = LlmEvalWorkflow(args)
    assert wf.args is args
    assert wf.seq_len == 1024
    assert wf.device == "cuda:0"
    assert wf.granularity == "block"
    assert wf.eval_mode == "quant"
    assert wf.pipeline is None
    assert wf.data_provider is None
    assert wf.model_name == "qwen3"
    assert wf.quant_target == ["mlp", "attn-cache"]
    assert wf.quant_dtype == "int4"
    assert wf.bit_policy is bp


# ---- run with model granularity -------------------------------------------


def test_eval_run_modelwise(monkeypatch):
    monkeypatch.setattr(
        "amct_pytorch.workflows.llm_eval.get_wiki_inputs",
        lambda tokenizer, seq_len: [torch.randint(0, 100, (2, 4))],
    )
    monkeypatch.setattr(
        "amct_pytorch.workflows.llm_eval.logger",
        MagicMock(),
    )
    monkeypatch.setattr(
        "amct_pytorch.workflows.llm_eval.tqdm",
        lambda *args, **kwargs: pytest.fail(
            "_iter_model_logits must not own the PPL progress bar"
        ),
    )
    observed = {"calls": 0}

    def fake_ppl(logits_iter, samples, device, seq_len):
        observed["calls"] += 1
        assert len(list(logits_iter)) == len(samples)
        return 12.34

    monkeypatch.setattr(
        "amct_pytorch.workflows.llm_eval.wikitext2_ppl",
        fake_ppl,
        raising=False,
    )
    wf = _make_eval_workflow(eval_mode="bf16", granularity="model")
    wf.pipeline = MagicMock()
    wf.pipeline.tokenizer = MagicMock()
    call_result = SimpleNamespace(logits=torch.randn(2, 3, 100))
    forward_fn = MagicMock(return_value=call_result)
    monkeypatch.setattr(
        wf.pipeline,
        "float_model",
        MagicMock(
            return_value=MagicMock(
                eval=MagicMock(
                    return_value=MagicMock(to=MagicMock(return_value=forward_fn))
                )
            )
        ),
    )

    def setup():
        return "sink"

    wf.setup = setup
    wf.run()
    assert observed["calls"] == 1


# ---- setup with sharded_block ---------------------------------------------


def test_eval_setup_enables_sharded_block(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "amct_pytorch.workflows.llm_eval.register_llm_models", lambda: None
    )
    monkeypatch.setattr("amct_pytorch.workflows.llm_eval.register_dtype", lambda: None)
    monkeypatch.setattr(
        "amct_pytorch.workflows.llm_eval.register_algorithms", lambda: None
    )

    class FakePipeline:
        sharded_block = False

        def __init__(self, args):
            pass

    monkeypatch.setattr(
        "amct_pytorch.workflows.llm_eval.MODEL_REGISTRY",
        SimpleNamespace(get=lambda k: FakePipeline),
    )

    wf = _make_eval_workflow(output_dir=str(tmp_path))
    wf.setup()
    assert wf.pipeline.sharded_block is True


# ---- _save_inter_result ---------------------------------------------------


def test_eval_save_inter_result(tmp_path):
    wf = _make_eval_workflow(output_dir=str(tmp_path))
    t = torch.tensor([1.0, 2.0])
    wf._save_inter_result(t, "test_result")
    saved = tmp_path / "test_result.pkl"
    assert saved.exists()
    loaded = torch.load(str(saved))
    assert torch.equal(loaded, t)


def test_has_relevant_quant_with_cache_target():
    bp = _make_bit_policy(**{"attn-cache": {"q": 8, "k": 8, "p": 8, "v": 8}})
    workflow = _make_workflow(
        eval_mode="quant", quant_target=("attn-cache",), bit_policy=bp
    )
    assert workflow._has_relevant_quant() is True


def test_get_relevant_quant_bits_with_cache():
    bp = _make_bit_policy(
        w_bits=16, a_bits=16, **{"attn-cache": {"q": 4, "k": 4, "p": 4, "v": 4}}
    )
    workflow = _make_workflow(
        eval_mode="quant", quant_target=("attn-cache",), bit_policy=bp
    )
    bits = workflow._get_relevant_quant_bits()
    assert 4 in bits


def test_get_relevant_quant_bits_with_mlp_and_cache():
    bp = _make_bit_policy(
        w_bits=8, a_bits=8, **{"attn-cache": {"q": 4, "k": 4, "p": 4, "v": 4}}
    )
    workflow = _make_workflow(
        eval_mode="quant", quant_target=("mlp", "attn-cache"), bit_policy=bp
    )
    bits = workflow._get_relevant_quant_bits()
    assert 8 in bits
    assert 4 in bits
