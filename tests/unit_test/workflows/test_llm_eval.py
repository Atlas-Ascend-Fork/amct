"""Logic tests for LlmEvalWorkflow.

The end-to-end blockwise / modelwise loops require a real model checkpoint and
NPU device, so we cover only the pure decision logic here:
``_resolve_eval_states`` and ``_get_relevant_quant_bits``.
"""

from types import SimpleNamespace

import pytest

from amct_pytorch.workflows.llm_eval import LlmEvalWorkflow


def _make_workflow(eval_mode="bf16", quant_target=(), **bits):
    """Build a workflow without invoking the heavy __init__ path."""
    workflow = LlmEvalWorkflow.__new__(LlmEvalWorkflow)
    args = SimpleNamespace(
        eval_mode=eval_mode,
        quant_target=list(quant_target),
        a_bits=bits.get("a_bits", 16),
        w_bits=bits.get("w_bits", 16),
        q_bits=bits.get("q_bits", 16),
        k_bits=bits.get("k_bits", 16),
        p_bits=bits.get("p_bits", 16),
        v_bits=bits.get("v_bits", 16),
    )
    workflow.args = args
    workflow.eval_mode = args.eval_mode
    return workflow


# ---- _get_relevant_quant_bits --------------------------------------------


def test_relevant_bits_empty_when_no_quant_target():
    assert _make_workflow()._get_relevant_quant_bits() == []


@pytest.mark.parametrize("target", ["mlp", "moe", "attn-linear"])
def test_relevant_bits_for_linear_targets_uses_a_and_w_bits(target):
    wf = _make_workflow(quant_target=[target], a_bits=8, w_bits=4)
    assert sorted(wf._get_relevant_quant_bits()) == [4, 8]


def test_relevant_bits_for_attn_cache_uses_qkpv_bits():
    wf = _make_workflow(
        quant_target=["attn-cache"], q_bits=8, k_bits=8, p_bits=4, v_bits=4
    )
    assert sorted(wf._get_relevant_quant_bits()) == [4, 4, 8, 8]


def test_relevant_bits_combines_linear_and_cache_targets():
    wf = _make_workflow(
        quant_target=["mlp", "attn-cache"],
        a_bits=8,
        w_bits=4,
        q_bits=16,
        k_bits=16,
        p_bits=16,
        v_bits=16,
    )
    assert sorted(wf._get_relevant_quant_bits()) == [4, 8, 16, 16, 16, 16]


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
    assert "quantization disabled" in msg


def test_resolve_eval_states_quant_with_all_bits_16_skips_real_quant():
    wf = _make_workflow(
        eval_mode="quant", quant_target=["mlp"], a_bits=16, w_bits=16
    )
    use_quant_block, enable_quant, _ = wf._resolve_eval_states()
    assert use_quant_block is True
    assert enable_quant is False


def test_resolve_eval_states_quant_enables_when_any_relevant_bit_lt_16():
    wf = _make_workflow(
        eval_mode="quant", quant_target=["mlp"], a_bits=8, w_bits=16
    )
    use_quant_block, enable_quant, msg = wf._resolve_eval_states()
    assert use_quant_block is True
    assert enable_quant is True
    assert "quantization enabled" in msg


def test_resolve_eval_states_ignores_irrelevant_low_bits_for_target():
    # quant_target=mlp uses only a/w bits; q/k/p/v=4 should NOT trigger enable.
    wf = _make_workflow(
        eval_mode="quant",
        quant_target=["mlp"],
        a_bits=16,
        w_bits=16,
        q_bits=4,
        k_bits=4,
        p_bits=4,
        v_bits=4,
    )
    _, enable_quant, _ = wf._resolve_eval_states()
    assert enable_quant is False


def test_resolve_eval_states_attn_cache_low_bit_triggers_enable():
    wf = _make_workflow(
        eval_mode="quant",
        quant_target=["attn-cache"],
        q_bits=8, k_bits=16, p_bits=16, v_bits=16,
    )
    _, enable_quant, _ = wf._resolve_eval_states()
    assert enable_quant is True
