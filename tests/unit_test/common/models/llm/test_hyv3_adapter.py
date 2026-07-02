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
"""Unit tests for HunyuanVideo V3 (hy_v3) model adapter."""
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

# transformers.models.hy_v3 is a custom addition not present in stock transformers.
# Mock it before any hyv3 module is imported so collection succeeds in CI.
_hy_v3_mod = MagicMock()
_hy_v3_mod.HYV3Config = type("HYV3Config", (), {})
_hy_v3_mod.HYV3DecoderLayer = type("HYV3DecoderLayer", (), {})
_hy_v3_mod.HYV3Attention = type("HYV3Attention", (torch.nn.Module,), {
    "__init__": lambda self, config, layer_idx: torch.nn.Module.__init__(self),
})
_hy_v3_mod.HYV3MoE = type("HYV3MoE", (torch.nn.Module,), {
    "__init__": lambda self, *a, **kw: torch.nn.Module.__init__(self),
})
_hy_v3_mod.apply_rotary_pos_emb = lambda *a, **kw: a[:2]

sys.modules.setdefault("transformers.models.hy_v3", _hy_v3_mod)
sys.modules.setdefault("transformers.models.hy_v3.modeling_hy_v3", _hy_v3_mod)

from amct_pytorch.common.models.llm.hyv3.hyv3 import HyV3, remap_hyv3_keys  # noqa: E402
from amct_pytorch.common.models.llm.hyv3.quant_module import QuantHYV3MoE  # noqa: E402


def _stub(quant_target=("moe",), **attrs):
    """Bypass HyV3.__init__ and set only the attributes needed for testing."""
    obj = HyV3.__new__(HyV3)
    obj.args = SimpleNamespace(quant_target=list(quant_target))
    obj.quant_target = list(quant_target)
    for k, v in attrs.items():
        setattr(obj, k, v)
    return obj


# ---------------------------------------------------------------------------
# remap_hyv3_keys — pure function
# ---------------------------------------------------------------------------

class TestRemapHyV3Keys:
    def test_router_gate_remapped(self):
        sd = {"model.layers.0.mlp.router.gate.weight": torch.zeros(4, 4)}
        out = remap_hyv3_keys(sd)
        assert "model.layers.0.mlp.gate.weight" in out
        assert "model.layers.0.mlp.router.gate.weight" not in out

    def test_expert_bias_remapped(self):
        sd = {"model.layers.0.mlp.expert_bias": torch.zeros(4)}
        out = remap_hyv3_keys(sd)
        assert "model.layers.0.mlp.e_score_correction_bias" in out
        assert "model.layers.0.mlp.expert_bias" not in out

    def test_shared_mlp_remapped(self):
        sd = {"model.layers.0.mlp.shared_mlp.gate_proj.weight": torch.zeros(4, 4)}
        out = remap_hyv3_keys(sd)
        assert "model.layers.0.mlp.shared_experts.gate_proj.weight" in out
        assert "model.layers.0.mlp.shared_mlp.gate_proj.weight" not in out

    def test_unrelated_keys_pass_through(self):
        sd = {
            "model.embed_tokens.weight": torch.zeros(4, 4),
            "model.norm.weight": torch.zeros(4),
        }
        out = remap_hyv3_keys(sd)
        assert set(out.keys()) == set(sd.keys())

    def test_multiple_keys_all_remapped(self):
        sd = {
            "mlp.router.gate.weight": torch.zeros(4, 4),
            "mlp.expert_bias": torch.zeros(4),
            "mlp.shared_mlp.up_proj.weight": torch.zeros(4, 4),
            "other.weight": torch.zeros(4, 4),
        }
        out = remap_hyv3_keys(sd)
        assert "mlp.gate.weight" in out
        assert "mlp.e_score_correction_bias" in out
        assert "mlp.shared_experts.up_proj.weight" in out
        assert "other.weight" in out
        assert len(out) == 4


# ---------------------------------------------------------------------------
# HyV3 adapter methods
# ---------------------------------------------------------------------------

class TestHyV3Adapter:
    def test_parse_quant_mode_rejects_mlp(self):
        obj = _stub(quant_target=["mlp"])
        with pytest.raises(ValueError, match="does not support quant_target='mlp'"):
            obj.parse_quant_mode()

    def test_parse_quant_mode_accepts_moe(self):
        obj = _stub(quant_target=["moe"])
        obj.parse_quant_mode()  # must not raise

    def test_parse_quant_mode_accepts_attn_linear(self):
        obj = _stub(quant_target=["attn-linear"])
        obj.parse_quant_mode()  # must not raise

    def test_get_layer_weight_prefix(self):
        obj = _stub()
        assert obj.get_layer_weight_prefix(3) == "model.layers.3."

    def test_load_layer_weight_applies_remap_and_pack(self, monkeypatch):
        obj = _stub()
        raw = {
            "mlp.router.gate.weight": torch.zeros(4, 4),
            "mlp.experts.0.gate_proj.weight": torch.zeros(4, 4),
            "mlp.experts.0.up_proj.weight": torch.zeros(4, 4),
            "mlp.experts.0.down_proj.weight": torch.zeros(4, 4),
        }
        monkeypatch.setattr(
            "amct_pytorch.common.models.llm.common.base.BaseModel.load_layer_weight",
            lambda self, prefix: dict(raw),
        )
        captured_prefix = {}

        def _mock_pack(sd, expert_prefix):
            captured_prefix["ep"] = expert_prefix
            return sd
        monkeypatch.setattr(
            "amct_pytorch.common.models.llm.hyv3.hyv3.pack_gated_expert_weights",
            _mock_pack,
        )
        result = obj.load_layer_weight("model.layers.0.")
        assert "mlp.gate.weight" in result
        assert captured_prefix["ep"] == "mlp.experts"

    def test_load_layer_weight_skips_pack_when_no_gate_proj(self, monkeypatch):
        obj = _stub()
        raw = {"mlp.router.gate.weight": torch.zeros(4, 4)}
        monkeypatch.setattr(
            "amct_pytorch.common.models.llm.common.base.BaseModel.load_layer_weight",
            lambda self, prefix: dict(raw),
        )
        pack_called = []
        monkeypatch.setattr(
            "amct_pytorch.common.models.llm.hyv3.hyv3.pack_gated_expert_weights",
            lambda sd, **kw: pack_called.append(1) or sd,
        )
        obj.load_layer_weight("model.layers.0.")
        assert not pack_called

    def test_iter_ptq_units_moe_yields_experts(self):
        obj = _stub(quant_target=["moe"])
        block = nn.Module()
        block.mlp = nn.Module()
        block.mlp.experts = nn.ModuleList([nn.Linear(4, 4), nn.Linear(4, 4)])
        units = list(obj.iter_ptq_units(0, block))
        assert len(units) == 2
        assert all(u.kind == "moe" for u in units)

    def test_iter_ptq_units_attn_linear_yields_attn(self):
        obj = _stub(quant_target=["attn-linear"])
        block = nn.Module()
        block.self_attn = nn.Linear(4, 4)
        units = list(obj.iter_ptq_units(0, block))
        assert len(units) == 1
        assert units[0].kind == "attn"

    def test_build_quant_block_moe_replaces_mlp(self, monkeypatch):
        obj = _stub(quant_target=["moe"])
        dl = nn.Module()
        moe = nn.Module()
        moe.experts = nn.Linear(4, 4)
        dl.mlp = moe
        monkeypatch.setattr(obj, "block", lambda idx: dl)
        sentinel = nn.Linear(2, 2)
        monkeypatch.setattr(
            "amct_pytorch.common.models.llm.hyv3.hyv3.QuantHYV3MoE",
            lambda args, mlp: sentinel,
        )
        result = obj.build_quant_block(0)
        assert result.mlp is sentinel

    def test_build_quant_block_moe_fallback_to_mlp_when_no_experts(self, monkeypatch):
        obj = _stub(quant_target=["moe"])
        dl = nn.Module()
        dl.mlp = nn.Linear(4, 4)
        monkeypatch.setattr(obj, "block", lambda idx: dl)
        sentinel = nn.Linear(2, 2)
        monkeypatch.setattr(
            "amct_pytorch.common.models.llm.hyv3.hyv3.QuantHYV3MLP",
            lambda args, mlp, group: sentinel,
        )
        result = obj.build_quant_block(0)
        assert result.mlp is sentinel

    def test_build_quant_block_attn_linear_calls_apply_quant(self, monkeypatch):
        obj = _stub(quant_target=["attn-linear"])
        dl = nn.Module()
        monkeypatch.setattr(obj, "block", lambda idx: dl)
        captured = {}
        monkeypatch.setattr(
            "amct_pytorch.common.models.llm.hyv3.hyv3.apply_quant_to_attn",
            lambda args, block, cls: captured.update({"cls": cls}),
        )
        from amct_pytorch.common.models.llm.hyv3.quant_module import QuantHYV3Attn
        obj.build_quant_block(0)
        assert captured["cls"] is QuantHYV3Attn

    def test_iter_deploy_bindings_yields_quant_linears(self):
        from amct_pytorch.quantization.modules.quant_linear import QuantLinear
        obj = _stub()

        ql = QuantLinear.__new__(QuantLinear)
        ql._parameters, ql._modules, ql._buffers = {}, {}, {}
        block = nn.Module()
        block.proj = ql

        bindings = list(obj.iter_deploy_bindings(0, block))
        assert len(bindings) == 1
        assert bindings[0][0] == "model.layers.0.proj.weight"
        assert bindings[0][1] is ql


# ---------------------------------------------------------------------------
# QuantHYV3MoE.forward
# ---------------------------------------------------------------------------

class TestQuantHYV3MoEForward:
    def _make_moe(self, num_experts=4, top_k=2, enable_fp32=False):
        """Build a QuantHYV3MoE stub without calling __init__."""
        obj = QuantHYV3MoE.__new__(QuantHYV3MoE)
        obj.top_k = top_k
        obj.enable_moe_fp32_combine = enable_fp32

        # gate: returns (logits, weights, indices)
        def _gate(hidden, bias):
            bsz = hidden.shape[0]
            weights = torch.ones(bsz, top_k) / top_k
            indices = torch.zeros(bsz, top_k, dtype=torch.long)
            return None, weights, indices

        obj.gate = _gate
        obj.register_buffer = lambda name, val: setattr(obj, name, val)
        obj.e_score_correction_bias = torch.zeros(num_experts)

        # experts: returns zeros of same shape as input
        mock_experts = MagicMock()
        mock_experts.return_value = torch.zeros(1, 8)
        obj.experts = mock_experts

        # shared_experts: identity
        obj.shared_experts = lambda x: torch.zeros_like(x)

        return obj

    def test_forward_output_shape(self):
        obj = self._make_moe()
        h = torch.randn(2, 4, 8)
        out = obj.forward(h)
        assert out.shape == (2, 4, 8)

    def test_forward_fp32_combine_path(self):
        obj = self._make_moe(enable_fp32=True)
        h = torch.randn(1, 3, 8)
        out = obj.forward(h)
        assert out.shape == (1, 3, 8)
        assert out.dtype == h.dtype
