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
import torch.nn.functional as F

# transformers.models.hy_v3 is a custom addition not present in stock transformers.
# Mock it before any hyv3 module is imported so collection succeeds in CI.
# HYV3Attention.__init__ is stubbed to read head_dim/scaling from config so that
# QuantHYV3Attn (which calls super().__init__ then uses self.head_dim) works.
_hy_v3_mod = MagicMock()
_hy_v3_mod.HYV3Config = type("HYV3Config", (), {})
_hy_v3_mod.HYV3DecoderLayer = type("HYV3DecoderLayer", (), {})


def _hyv3_attn_init(self, config, layer_idx):
    torch.nn.Module.__init__(self)
    self.head_dim = config.head_dim
    self.attention_dropout = getattr(config, "attention_dropout", 0.0)
    self.scaling = getattr(config, "scaling", 1.0)
    nq = getattr(config, "num_attention_heads", 1)
    nkv = getattr(config, "num_key_value_heads", nq)
    self.num_key_value_groups = nq // nkv


_hy_v3_mod.HYV3Attention = type("HYV3Attention", (torch.nn.Module,), {"__init__": _hyv3_attn_init})
_hy_v3_mod.HYV3MoE = type("HYV3MoE", (torch.nn.Module,), {
    "__init__": lambda self, *a, **kw: torch.nn.Module.__init__(self),
})
_hy_v3_mod.apply_rotary_pos_emb = lambda *a, **kw: a[:2]

sys.modules.setdefault("transformers.models.hy_v3", _hy_v3_mod)
sys.modules.setdefault("transformers.models.hy_v3.modeling_hy_v3", _hy_v3_mod)

from amct_pytorch.common.models.llm.hyv3.hyv3 import HyV3, remap_hyv3_keys  # noqa: E402
from amct_pytorch.common.models.llm.hyv3.quant_module import QuantHYV3Attn, QuantHYV3MLP, QuantHYV3MoE  # noqa: E402
from amct_pytorch.common.models.llm.common.quant_apply import PlainLinear  # noqa: E402
from amct_pytorch.quantization.bit_policy import BitPolicy  # noqa: E402
from amct_pytorch.quantization.dtypes import register_dtype  # noqa: E402

register_dtype()

W_BITS = "w_bits"
A_BITS = "a_bits"
ATTN_LINEAR = "attn-linear"
ATTN_CACHE = "attn-cache"
MLP = "mlp"


def _quant_args(quant_target=(ATTN_LINEAR,)):
    """Build a minimal quant_args with a BitPolicy covering all hyv3 groups."""
    return SimpleNamespace(
        algos=[],
        quant_dtype="int",
        w_bits=8,
        a_bits=8,
        q_bits=8,
        k_bits=8,
        p_bits=8,
        v_bits=8,
        quant_target=list(quant_target),
        bit_policy=BitPolicy({
            ATTN_LINEAR: {n: {W_BITS: 8, A_BITS: 8} for n in ("q_proj", "k_proj", "v_proj", "o_proj")},
            ATTN_CACHE: {"q": 8, "k": 8, "p": 8, "v": 8},
            MLP: {n: {W_BITS: 8, A_BITS: 8} for n in ("gate_proj", "up_proj", "down_proj")},
            "moe": {
                "routed": {n: {W_BITS: 8, A_BITS: 8} for n in ("gate_proj", "up_proj", "down_proj")},
                "shared": {n: {W_BITS: 8, A_BITS: 8} for n in ("gate_proj", "up_proj", "down_proj")},
            },
        }),
    )


def _attn_config():
    return SimpleNamespace(
        hidden_size=64,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=16,
        attention_dropout=0.0,
        scaling=1.0,
    )


def _make_attn_module(cfg):
    """Fake HYV3Attention with real nn.Linear projections for init/forward."""
    m = nn.Module()
    m.config = cfg
    m.layer_idx = 0
    m.q_proj = nn.Linear(cfg.hidden_size, cfg.num_attention_heads * cfg.head_dim)
    m.k_proj = nn.Linear(cfg.hidden_size, cfg.num_key_value_heads * cfg.head_dim)
    m.v_proj = nn.Linear(cfg.hidden_size, cfg.num_key_value_heads * cfg.head_dim)
    m.o_proj = nn.Linear(cfg.num_attention_heads * cfg.head_dim, cfg.hidden_size)
    m.q_norm = nn.Identity()
    m.k_norm = nn.Identity()
    return m


def _position_embeddings(bs, seq, head_dim):
    return torch.randn(bs, seq, head_dim), torch.randn(bs, seq, head_dim)


def _make_moe_module(num_experts=2, hidden=64, intermediate=32, top_k=2, fp32=False):
    """Fake HYV3MoE with packed experts (gate_up/down) + shared GatedMLP."""
    m = nn.Module()
    m.top_k = top_k
    m.enable_moe_fp32_combine = fp32
    m.e_score_correction_bias = torch.zeros(num_experts)

    def _gate(hd, bias):
        bsz = hd.shape[0]
        return None, torch.ones(bsz, top_k) / top_k, torch.zeros(bsz, top_k, dtype=torch.long)

    m.gate = _gate
    experts = nn.Module()
    experts.num_experts = num_experts
    experts.hidden_dim = hidden
    experts.intermediate_dim = intermediate
    experts.act_fn = F.silu
    experts.gate_up_proj = nn.Parameter(torch.randn(num_experts, 2 * intermediate, hidden))
    experts.down_proj = nn.Parameter(torch.randn(num_experts, hidden, intermediate))
    m.experts = experts
    shared = nn.Module()
    shared.hidden_size = hidden
    shared.intermediate_size = intermediate
    shared.act_fn = F.silu
    shared.gate_proj = nn.Linear(hidden, intermediate)
    shared.up_proj = nn.Linear(hidden, intermediate)
    shared.down_proj = nn.Linear(intermediate, hidden)
    m.shared_experts = shared
    return m


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
    @staticmethod
    def test_router_gate_remapped():
        sd = {"model.layers.0.mlp.router.gate.weight": torch.zeros(4, 4)}
        out = remap_hyv3_keys(sd)
        assert "model.layers.0.mlp.gate.weight" in out
        assert "model.layers.0.mlp.router.gate.weight" not in out

    @staticmethod
    def test_expert_bias_remapped():
        sd = {"model.layers.0.mlp.expert_bias": torch.zeros(4)}
        out = remap_hyv3_keys(sd)
        assert "model.layers.0.mlp.e_score_correction_bias" in out
        assert "model.layers.0.mlp.expert_bias" not in out

    @staticmethod
    def test_shared_mlp_remapped():
        sd = {"model.layers.0.mlp.shared_mlp.gate_proj.weight": torch.zeros(4, 4)}
        out = remap_hyv3_keys(sd)
        assert "model.layers.0.mlp.shared_experts.gate_proj.weight" in out
        assert "model.layers.0.mlp.shared_mlp.gate_proj.weight" not in out

    @staticmethod
    def test_unrelated_keys_pass_through():
        sd = {
            "model.embed_tokens.weight": torch.zeros(4, 4),
            "model.norm.weight": torch.zeros(4),
        }
        out = remap_hyv3_keys(sd)
        assert set(out.keys()) == set(sd.keys())

    @staticmethod
    def test_multiple_keys_all_remapped():
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
    @staticmethod
    def test_parse_quant_mode_rejects_mlp():
        obj = _stub(quant_target=["mlp"])
        with pytest.raises(ValueError, match="does not support quant_target='mlp'"):
            obj.parse_quant_mode()

    @staticmethod
    def test_parse_quant_mode_accepts_moe():
        obj = _stub(quant_target=["moe"])
        obj.parse_quant_mode()  # must not raise

    @staticmethod
    def test_parse_quant_mode_accepts_attn_linear():
        obj = _stub(quant_target=["attn-linear"])
        obj.parse_quant_mode()  # must not raise

    @staticmethod
    def test_get_layer_weight_prefix():
        obj = _stub()
        assert obj.get_layer_weight_prefix(3) == "model.layers.3."

    @staticmethod
    def test_get_scale_name_returns_scale_inv_suffix():
        obj = _stub()
        scale_prefix, scale_inv_name = obj.get_scale_name("mlp.experts.0.gate_proj.weight")
        assert scale_prefix == "_scale_inv"
        assert scale_inv_name == "mlp.experts.0.gate_proj.weight_scale_inv"

    @staticmethod
    def test_load_layer_weight_applies_remap_and_pack(monkeypatch):
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

    @staticmethod
    def test_load_layer_weight_skips_pack_when_no_gate_proj(monkeypatch):
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

    @staticmethod
    def test_iter_ptq_units_moe_yields_experts():
        obj = _stub(quant_target=["moe"])
        block = nn.Module()
        block.mlp = nn.Module()
        block.mlp.experts = nn.ModuleList([nn.Linear(4, 4), nn.Linear(4, 4)])
        units = list(obj.iter_ptq_units(0, block))
        assert len(units) == 2
        assert all(u.kind == "moe" for u in units)

    @staticmethod
    def test_iter_ptq_units_attn_linear_yields_attn():
        obj = _stub(quant_target=["attn-linear"])
        block = nn.Module()
        block.self_attn = nn.Linear(4, 4)
        units = list(obj.iter_ptq_units(0, block))
        assert len(units) == 1
        assert units[0].kind == "attn"

    @staticmethod
    def test_build_quant_block_moe_replaces_mlp(monkeypatch):
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

    @staticmethod
    def test_build_quant_block_moe_fallback_to_mlp_when_no_experts(monkeypatch):
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

    @staticmethod
    def test_build_quant_block_attn_linear_calls_apply_quant(monkeypatch):
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
        assert captured.get("cls") is QuantHYV3Attn

    @staticmethod
    def test_iter_deploy_bindings_yields_quant_linears():
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

    @staticmethod
    def test_iter_deploy_bindings_remaps_packed_experts():
        from amct_pytorch.quantization.modules.quant_linear import QuantLinear
        obj = _stub(quant_target=["moe"])

        def _ql():
            q = QuantLinear.__new__(QuantLinear)
            q._parameters, q._modules, q._buffers = {}, {}, {}
            return q

        # QuantGatedMLP stores gate/up/down_proj as QuantLinear directly, so each
        # QuantLinear surfaces at the 5-segment name
        # mlp.experts.expert_modules.<idx>.<proj> (no extra .quant_linear child).
        # Mirror that real structure so the test exercises the true deploy path.
        expert_0 = nn.Module()
        expert_0.gate_proj = _ql()
        expert_0.up_proj = _ql()
        expert_0.down_proj = _ql()

        expert_modules = nn.ModuleList([expert_0])

        experts_wrapper = nn.Module()
        experts_wrapper.expert_modules = expert_modules

        mlp = nn.Module()
        mlp.experts = experts_wrapper

        block = nn.Module()
        block.mlp = mlp

        bindings = sorted(obj.iter_deploy_bindings(0, block), key=lambda b: b[0])
        expected_prefixes = [
            "model.layers.0.mlp.experts.0.down_proj.weight",
            "model.layers.0.mlp.experts.0.gate_proj.weight",
            "model.layers.0.mlp.experts.0.up_proj.weight",
        ]
        actual_prefixes = [b[0] for b in bindings]
        assert actual_prefixes == expected_prefixes


# ---------------------------------------------------------------------------
# QuantHYV3MoE.forward
# ---------------------------------------------------------------------------

class TestQuantHYV3MoEForward:
    @staticmethod
    def test_forward_output_shape():
        obj = TestQuantHYV3MoEForward._make_moe()
        h = torch.randn(2, 4, 8)
        out = obj.forward(h)
        assert out.shape == (2, 4, 8)

    @staticmethod
    def test_forward_fp32_combine_path():
        obj = TestQuantHYV3MoEForward._make_moe(enable_fp32=True)
        h = torch.randn(1, 3, 8)
        out = obj.forward(h)
        assert out.shape == (1, 3, 8)
        assert out.dtype == h.dtype

    @staticmethod
    def _make_moe(num_experts=4, top_k=2, enable_fp32=False):
        """Build a QuantHYV3MoE stub without calling __init__."""
        obj = QuantHYV3MoE.__new__(QuantHYV3MoE)
        obj.top_k = top_k
        obj.enable_moe_fp32_combine = enable_fp32

        def _gate(hidden, bias):
            bsz = hidden.shape[0]
            weights = torch.ones(bsz, top_k) / top_k
            indices = torch.zeros(bsz, top_k, dtype=torch.long)
            return None, weights, indices

        obj.gate = _gate
        obj.register_buffer = lambda name, val: setattr(obj, name, val)
        obj.e_score_correction_bias = torch.zeros(num_experts)

        mock_experts = MagicMock()
        mock_experts.return_value = torch.zeros(1, 8)
        obj.experts = mock_experts

        obj.shared_experts = lambda x: torch.zeros_like(x)

        return obj


# ---------------------------------------------------------------------------
# QuantHYV3Attn — __init__ / forward (real construction)
# ---------------------------------------------------------------------------


class TestQuantHYV3Attn:
    @staticmethod
    def test_init_attn_linear_uses_quant_linear_projections():
        cfg = _attn_config()
        attn = _make_attn_module(cfg)
        qattn = QuantHYV3Attn(_quant_args([ATTN_LINEAR]), attn)
        from amct_pytorch.quantization.modules.quant_linear import QuantLinear
        for proj in (qattn.q_proj, qattn.k_proj, qattn.v_proj, qattn.o_proj):
            assert isinstance(proj, QuantLinear)
        assert not isinstance(qattn.inp_afq, nn.Identity)
        assert not isinstance(qattn.o_proj_afq, nn.Identity)
        assert qattn.head_dim == cfg.head_dim
        assert qattn.value_dim == cfg.head_dim * cfg.num_attention_heads

    @staticmethod
    def test_init_cache_only_uses_plain_linear_and_identity_afq():
        cfg = _attn_config()
        attn = _make_attn_module(cfg)
        qattn = QuantHYV3Attn(_quant_args([ATTN_CACHE]), attn)
        for proj in (qattn.q_proj, qattn.k_proj, qattn.v_proj, qattn.o_proj):
            assert isinstance(proj, PlainLinear)
        assert qattn.input_transform is None
        assert qattn.out_transform is None
        assert isinstance(qattn.inp_afq, nn.Identity)
        assert isinstance(qattn.o_proj_afq, nn.Identity)

    @staticmethod
    def test_init_builds_qk_and_pv_matmul():
        cfg = _attn_config()
        attn = _make_attn_module(cfg)
        qattn = QuantHYV3Attn(_quant_args([ATTN_CACHE]), attn)
        assert qattn.qk_matmul is not None
        assert qattn.pv_matmul is not None

    @staticmethod
    def test_forward_attn_linear_returns_correct_shape():
        cfg = _attn_config()
        attn = _make_attn_module(cfg)
        qattn = QuantHYV3Attn(_quant_args([ATTN_LINEAR]), attn)
        bs, seq = 1, 6
        h = torch.randn(bs, seq, cfg.hidden_size)
        cos, sin = _position_embeddings(bs, seq, cfg.head_dim)
        out, weights = qattn(h, position_embeddings=(cos, sin))
        assert out.shape == h.shape
        assert weights is None

    @staticmethod
    def test_forward_cache_only_returns_finite_output():
        cfg = _attn_config()
        attn = _make_attn_module(cfg)
        qattn = QuantHYV3Attn(_quant_args([ATTN_CACHE]), attn)
        bs, seq = 1, 4
        h = torch.randn(bs, seq, cfg.hidden_size)
        cos, sin = _position_embeddings(bs, seq, cfg.head_dim)
        out, _ = qattn(h, position_embeddings=(cos, sin))
        assert out.shape == h.shape
        assert torch.isfinite(out).all()

    @staticmethod
    def test_forward_with_structure_transform_branches():
        cfg = _attn_config()
        attn = _make_attn_module(cfg)
        qattn = QuantHYV3Attn(_quant_args([ATTN_LINEAR]), attn)

        class _IdentityTransform(nn.Module):
            def forward(self, x, **kwargs):
                return x

        qattn.input_transform = _IdentityTransform()
        qattn.out_transform = _IdentityTransform()
        bs, seq = 1, 4
        h = torch.randn(bs, seq, cfg.hidden_size)
        cos, sin = _position_embeddings(bs, seq, cfg.head_dim)
        out, _ = qattn(h, position_embeddings=(cos, sin))
        assert out.shape == h.shape


# ---------------------------------------------------------------------------
# QuantHYV3MoE — __init__ / forward (real construction)
# ---------------------------------------------------------------------------


class TestQuantHYV3MoEInit:
    @staticmethod
    def test_init_wraps_experts_and_shared():
        moe = _make_moe_module()
        qmoe = QuantHYV3MoE(_quant_args(["moe"]), moe)
        assert qmoe.top_k == moe.top_k
        assert qmoe.gate is moe.gate
        assert qmoe.enable_moe_fp32_combine is False
        assert torch.equal(qmoe.e_score_correction_bias, moe.e_score_correction_bias)
        from amct_pytorch.common.models.llm.qwen.moe_common import QuantGatedExperts
        assert isinstance(qmoe.experts, QuantGatedExperts)
        assert isinstance(qmoe.shared_experts, QuantHYV3MLP)

    @staticmethod
    def test_forward_default_combine_path():
        moe = _make_moe_module()
        qmoe = QuantHYV3MoE(_quant_args(["moe"]), moe)
        h = torch.randn(2, 4, 64)
        out = qmoe(h)
        assert out.shape == h.shape

    @staticmethod
    def test_forward_fp32_combine_path():
        moe = _make_moe_module(fp32=True)
        qmoe = QuantHYV3MoE(_quant_args(["moe"]), moe)
        h = torch.randn(1, 3, 64)
        out = qmoe(h)
        assert out.shape == h.shape
        assert out.dtype == h.dtype
