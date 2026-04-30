import pytest
import torch

from amct_pytorch.common.models.llm.qwen.moe_common import (
    is_packed_experts,
    pack_gated_expert_weights,
)


# ---- is_packed_experts ----------------------------------------------------


class _Packed:
    gate_up_proj = "x"
    down_proj = "y"


class _Unpacked:
    pass


def test_is_packed_experts_true_when_both_attrs_present():
    assert is_packed_experts(_Packed()) is True


def test_is_packed_experts_false_when_attr_missing():
    assert is_packed_experts(_Unpacked()) is False


def test_is_packed_experts_false_when_only_one_attr_present():
    class _Half:
        gate_up_proj = "x"

    assert is_packed_experts(_Half()) is False


# ---- pack_gated_expert_weights -------------------------------------------


def _expert_weights(num_experts=2, hidden=4, intermediate=3):
    sd = {}
    for idx in range(num_experts):
        sd[f"mlp.experts.{idx}.gate_proj.weight"] = torch.full((intermediate, hidden), float(idx) + 0.1)
        sd[f"mlp.experts.{idx}.up_proj.weight"] = torch.full((intermediate, hidden), float(idx) + 0.2)
        sd[f"mlp.experts.{idx}.down_proj.weight"] = torch.full((hidden, intermediate), float(idx) + 0.3)
    return sd


def test_pack_gated_expert_weights_combines_gate_up_into_single_tensor():
    sd = _expert_weights(num_experts=2, hidden=4, intermediate=3)
    packed = pack_gated_expert_weights(sd)
    # Original per-expert keys removed.
    assert "mlp.experts.0.gate_proj.weight" not in packed
    assert "mlp.experts.0.up_proj.weight" not in packed
    # Stacked tensors expose the expected shape.
    gu = packed["mlp.experts.gate_up_proj"]
    dp = packed["mlp.experts.down_proj"]
    assert gu.shape == (2, 6, 4)        # (num_experts, 2*intermediate, hidden)
    assert dp.shape == (2, 4, 3)        # (num_experts, hidden, intermediate)


def test_pack_gated_expert_weights_preserves_unrelated_keys():
    sd = _expert_weights()
    sd["other.weight"] = torch.zeros(4)
    packed = pack_gated_expert_weights(sd)
    assert torch.equal(packed["other.weight"], torch.zeros(4))


def test_pack_gated_expert_weights_returns_input_when_no_expert_keys_match():
    sd = {"some.module.weight": torch.ones(4)}
    out = pack_gated_expert_weights(sd)
    assert out is sd


def test_pack_gated_expert_weights_respects_custom_prefix():
    hidden, inter = 4, 3
    sd = {}
    for idx in range(2):
        sd[f"backbone.experts.{idx}.gate_proj.weight"] = torch.zeros(inter, hidden)
        sd[f"backbone.experts.{idx}.up_proj.weight"] = torch.zeros(inter, hidden)
        sd[f"backbone.experts.{idx}.down_proj.weight"] = torch.zeros(hidden, inter)
    packed = pack_gated_expert_weights(sd, expert_prefix="backbone.experts")
    assert "backbone.experts.gate_up_proj" in packed


def test_pack_gated_expert_weights_orders_experts_by_index():
    # Insert in scrambled order; result must still be sorted by expert index.
    sd = _expert_weights(num_experts=3)
    scrambled = {k: sd[k] for k in reversed(list(sd))}
    packed = pack_gated_expert_weights(scrambled)
    gate_up = packed["mlp.experts.gate_up_proj"]
    # Expert 0's gate_proj rows are filled with 0.1.
    assert gate_up[0, 0, 0].item() == pytest.approx(0.1)
    assert gate_up[1, 0, 0].item() == pytest.approx(1.1)
    assert gate_up[2, 0, 0].item() == pytest.approx(2.1)


def test_pack_gated_expert_weights_raises_on_inconsistent_experts():
    sd = _expert_weights(num_experts=2)
    # Drop one expert's down_proj to create asymmetry.
    sd.pop("mlp.experts.1.down_proj.weight")
    with pytest.raises(KeyError, match="Inconsistent expert weights"):
        pack_gated_expert_weights(sd)
