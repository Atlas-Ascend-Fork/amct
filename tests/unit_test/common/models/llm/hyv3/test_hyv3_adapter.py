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
"""Unit tests for the HyV3 adapter (key remap, deploy bindings, PTQ units)."""
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn
from transformers.models.hy_v3.modeling_hy_v3 import HYV3Config, HYV3DecoderLayer

from amct_pytorch.common.models.llm.hyv3.hyv3 import HyV3, remap_hyv3_keys
from amct_pytorch.quantization.modules.quant_linear import QuantLinear


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
        assert scale_prefix == "_scale"
        assert scale_inv_name == "mlp.experts.0.gate_proj.weight_scale"

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
        obj = _stub(quant_target=["moe"])

        def _ql():
            q = QuantLinear.__new__(QuantLinear)
            q._parameters, q._modules, q._buffers = {}, {}, {}
            return q

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
# HyV3.__init__ — mock HuggingFace loading chain (needs real HYV3Config)
# ---------------------------------------------------------------------------

_FROM_PRETRAINED = "from_pretrained"


def _make_mock_args(model_path="/tmp/fake_model", quant_target=("moe",), **extra):
    base = {
        "model": model_path,
        "quant_target": list(quant_target),
        "device": "cpu",
        "data_dir": "/tmp/fake_data",
        "output_dir": "/tmp/fake_output",
        "quant_dtype": "int8",
        "w_bits": 8,
        "a_bits": 8,
        "algos": [],
        "cali_bsz": 2,
    }
    base.update(extra)
    return SimpleNamespace(**base)


def _mock_hf_loading_chain(monkeypatch, cfg):
    """Mock AutoConfig/AutoTokenizer/AutoModelForCausalLM/get_weight_mappings
    so that ``HyV3(args)`` runs ``__init__`` without real model files."""
    from accelerate import init_empty_weights
    from transformers import AutoModelForCausalLM

    fake_tokenizer = MagicMock()
    monkeypatch.setattr(
        "amct_pytorch.common.models.llm.common.base.AutoConfig",
        type("FakeAC", (), {_FROM_PRETRAINED: staticmethod(lambda *a, **kw: cfg)})(),
    )
    monkeypatch.setattr(
        "amct_pytorch.common.models.llm.common.base.AutoTokenizer",
        type("FakeAT", (), {_FROM_PRETRAINED: staticmethod(lambda *a, **kw: fake_tokenizer)})(),
    )
    monkeypatch.setattr(
        "amct_pytorch.common.models.llm.common.base.init_empty_weights",
        lambda: MagicMock(__enter__=lambda s: None, __exit__=lambda *a: None),
    )
    with init_empty_weights():
        empty_model = AutoModelForCausalLM.from_config(
            cfg, trust_remote_code=True, torch_dtype=torch.bfloat16
        )
    monkeypatch.setattr(
        "amct_pytorch.common.models.llm.common.base.AutoModelForCausalLM",
        type("FakeAMFCLM", (), {
            _FROM_PRETRAINED: staticmethod(lambda *a, **kw: empty_model),
            "from_config": staticmethod(lambda *a, **kw: empty_model),
        })(),
    )
    monkeypatch.setattr(
        "amct_pytorch.common.models.llm.common.base.get_weight_mappings",
        lambda path: {},
    )
    monkeypatch.setattr(
        "amct_pytorch.common.models.llm.hyv3.hyv3.get_weight_mappings",
        lambda path: {},
    )
    monkeypatch.setattr(
        "compressed_tensors.utils.safetensors_load.get_weight_mappings",
        lambda path: {},
    )


def _tiny_hyv3_config():
    return HYV3Config(
        num_hidden_layers=2,
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        rms_norm_eps=1e-6,
        vocab_size=100,
        max_position_embeddings=512,
        rope_theta=10000.0,
        tie_word_embeddings=False,
        num_experts=2,
        num_experts_per_tok=2,
        num_shared_experts=1,
        moe_intermediate_size=32,
    )


class TestHyV3Mocked:
    @staticmethod
    def test_init_creates_model(monkeypatch):
        _mock_hf_loading_chain(monkeypatch, _tiny_hyv3_config())
        model = HyV3(_make_mock_args(quant_target=["moe"]))
        assert model.num_layers == 2
        assert model.textconfig is HYV3Config
        assert model.cls is HYV3DecoderLayer
        assert model.quant_target == ["moe"]
        assert model.config.num_hidden_layers == 2
        assert getattr(model.config, "_attn_implementation") == "eager"
        assert model.tokenizer is not None
        assert model.model is not None

    @staticmethod
    def test_init_rejects_mlp_quant_target(monkeypatch):
        _mock_hf_loading_chain(monkeypatch, _tiny_hyv3_config())
        with pytest.raises(ValueError, match="does not support quant_target='mlp'"):
            HyV3(_make_mock_args(quant_target=["mlp"]))

    @staticmethod
    def test_init_get_layer_weight_prefix(monkeypatch):
        _mock_hf_loading_chain(monkeypatch, _tiny_hyv3_config())
        model = HyV3(_make_mock_args())
        assert model.get_layer_weight_prefix(0) == "model.layers.0."

    @staticmethod
    def test_init_num_layers_property(monkeypatch):
        _mock_hf_loading_chain(monkeypatch, _tiny_hyv3_config())
        model = HyV3(_make_mock_args())
        assert model.num_layers == 2

    @staticmethod
    def test_init_textconfig_property(monkeypatch):
        _mock_hf_loading_chain(monkeypatch, _tiny_hyv3_config())
        model = HyV3(_make_mock_args())
        assert model.textconfig is HYV3Config


# ---- HyV3 new methods (diff coverage) ------------------------------------


def _stub_hyv3(**attrs):
    """Build a minimal HyV3 stub for testing new tensorwise methods."""
    obj = HyV3.__new__(HyV3)
    obj.args = SimpleNamespace(
        quant_target=["moe"],
        w_bits=8,
        a_bits=8,
    )
    obj.quant_target = ["moe"]
    obj.quant_dtype = "int"
    obj.config = SimpleNamespace(
        num_hidden_layers=2,
        num_nextn_predict_layers=1,
        num_experts=2,
    )
    obj.num_layers = obj.config.num_hidden_layers
    for k, v in attrs.items():
        setattr(obj, k, v)
    return obj


def test_hyv3_block_size_returns_1():
    assert HyV3.block_size(torch.randn(4, 4)) == 1


def test_hyv3_generate_tensorwise_ignore_layers():
    stub = _stub_hyv3()
    ignore = stub.generate_tensorwise_ignore_layers()
    assert "model.layers.0.mlp.gate_proj" in ignore
    assert "model.layers.0.mlp.down_proj" in ignore
    assert "model.layers.0.mlp.up_proj" in ignore
    assert "model.layers.2.eh_proj" in ignore
    assert "model.embed_tokens" in ignore
    assert "lm_head" in ignore


def test_hyv3_cache_scheme_int():
    stub = _stub_hyv3()
    stub.quant_dtype = "int"
    scheme = stub.cache_scheme()
    assert scheme["kv_cache_scheme"]["type"] == "int"
    assert scheme["kv_cache_scheme"]["num_bits"] == 8


def test_hyv3_cache_scheme_float():
    stub = _stub_hyv3()
    stub.quant_dtype = "bf16"
    scheme = stub.cache_scheme()
    assert scheme["kv_cache_scheme"]["type"] == "float"


def test_hyv3_bits_scheme():
    stub = _stub_hyv3()
    result = stub.bits_scheme()
    assert len(result) == 2
    assert result[0]["targets"] == ["Linear"]
    assert result[1]["targets"] == ["MoEGMM"]
    assert result[0]["w_bits"] == 8
    assert result[0]["a_bits"] == 8


def test_hyv3_generate_tensorwise_quant_layers():
    stub = _stub_hyv3()
    layers = stub.generate_tensorwise_quant_layers()
    # Layer 0: only attn (no experts, no shared_mlp)
    assert "model.layers.0.self_attn.q_proj" in layers
    assert "model.layers.0.self_attn.k_proj" in layers
    assert "model.layers.0.self_attn.v_proj" in layers
    assert "model.layers.0.self_attn.o_proj" in layers
    # Layer 0 should NOT have experts or shared_mlp
    assert not any("model.layers.0.mlp.experts" in k for k in layers)
    assert not any("model.layers.0.mlp.shared_mlp" in k for k in layers)
    # Layer 1: attn + experts + shared_mlp
    assert "model.layers.1.self_attn.q_proj" in layers
    assert "model.layers.1.mlp.experts.0.gate_proj" in layers
    assert "model.layers.1.mlp.experts.1.down_proj" in layers
    assert "model.layers.1.mlp.shared_mlp.gate_proj" in layers
    # Layer 2 (nextn predict): attn + experts (with bit=8) + shared_mlp
    assert "model.layers.2.self_attn.q_proj" in layers
    assert "model.layers.2.mlp.experts.0.gate_proj" in layers
    # nextn predict layers use bit=8 for routed experts
    assert layers["model.layers.2.mlp.experts.0.gate_proj"] == 8


def test_hyv3_get_scale_name_uses_scale_suffix():
    stub = _stub_hyv3()
    prefix, inv_name = stub.get_scale_name("model.layers.0.self_attn.q_proj.weight")
    assert prefix == "_scale"
    assert inv_name == "model.layers.0.self_attn.q_proj.weight_scale"
