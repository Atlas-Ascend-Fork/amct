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

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------
"""Tests for model adapters with mocked HuggingFace model loading."""
import pytest
import torch
import torch.nn as nn
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from accelerate import init_empty_weights
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from transformers.models.qwen3.modeling_qwen3 import Qwen3Config, Qwen3DecoderLayer

from tests.unit_test.common.models.llm.common.test_base import _make_block_with_attn_and_mlp
from amct_pytorch.quantization.bit_policy import BitPolicy


def _make_qwen3_config(num_layers=2):
    return Qwen3Config(
        num_hidden_layers=num_layers,
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
    )


def _make_mock_args(model_path="/tmp/fake_model", quant_target=("mlp",), **extra):
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


@pytest.fixture
def mock_hf_model(monkeypatch):
    """Mock HuggingFace AutoConfig/AutoTokenizer/AutoModelForCausalLM."""
    cfg = _make_qwen3_config()
    fake_tokenizer = MagicMock()

    monkeypatch.setattr(
        "amct_pytorch.common.models.llm.common.base.AutoConfig",
        type("FakeAC", (), {"from_pretrained": staticmethod(lambda *a, **kw: cfg)})(),
    )
    monkeypatch.setattr(
        "amct_pytorch.common.models.llm.common.base.AutoTokenizer",
        type("FakeAT", (), {"from_pretrained": staticmethod(lambda *a, **kw: fake_tokenizer)})(),
    )
    monkeypatch.setattr(
        "amct_pytorch.common.models.llm.common.base.init_empty_weights",
        lambda: MagicMock(__enter__=lambda s: None, __exit__=lambda *a: None),
    )
    with init_empty_weights():
        empty_model = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True, torch_dtype=torch.bfloat16)
    monkeypatch.setattr(
        "amct_pytorch.common.models.llm.common.base.AutoModelForCausalLM",
        type("FakeAMFCLM", (), {
            "from_pretrained": staticmethod(lambda *a, **kw: empty_model),
            "from_config": staticmethod(lambda *a, **kw: empty_model),
        })(),
    )
    monkeypatch.setattr(
        "amct_pytorch.common.models.llm.common.base.get_weight_mappings",
        lambda path: {},
    )
    monkeypatch.setattr(
        "compressed_tensors.utils.safetensors_load.get_weight_mappings",
        lambda path: {},
    )


class TestQwen3Mocked:
    def test_qwen3_init_creates_model(self, mock_hf_model):
        from amct_pytorch.common.models.llm.qwen.qwen3.qwen3 import Qwen3
        args = _make_mock_args(quant_target=["mlp"])
        model = Qwen3(args)
        assert model.num_layers == 2
        assert model.textconfig is Qwen3Config
        assert model.cls is Qwen3DecoderLayer
        assert model.quant_target == ["mlp"]
        assert model.config.num_hidden_layers == 2
        assert model.tokenizer is not None

    def test_qwen3_parse_quant_mode_rejects_moe(self, mock_hf_model):
        from amct_pytorch.common.models.llm.qwen.qwen3.qwen3 import Qwen3
        args = _make_mock_args(quant_target=["moe"])
        with pytest.raises(ValueError, match="does not support quant_target='moe'"):
            Qwen3(args)

    def test_qwen3_get_layer_weight_prefix(self, mock_hf_model):
        from amct_pytorch.common.models.llm.qwen.qwen3.qwen3 import Qwen3
        model = Qwen3(_make_mock_args())
        assert model.get_layer_weight_prefix(0) == "model.layers.0."

    def test_qwen3_iter_deploy_bindings_called(self, mock_hf_model):
        from amct_pytorch.common.models.llm.qwen.qwen3.qwen3 import Qwen3
        model = Qwen3(_make_mock_args())
        block = nn.Module()
        bindings = list(model.iter_deploy_bindings(0, block))
        assert isinstance(bindings, list)

    def test_qwen3_iter_ptq_units_called(self, mock_hf_model):
        from amct_pytorch.common.models.llm.qwen.qwen3.qwen3 import Qwen3
        model = Qwen3(_make_mock_args(quant_target=["mlp"]))
        block = nn.Module()
        block.mlp = nn.Linear(4, 4)
        units = list(model.iter_ptq_units(0, block))
        assert len(units) == 1
        assert units[0].kind == "mlp"

    def test_qwen3_export_block_deploy_returns_tensors(self, mock_hf_model):
        from amct_pytorch.common.models.llm.qwen.qwen3.qwen3 import Qwen3
        from amct_pytorch.common.models.llm.common.deploy_export import export_block_deploy
        model = Qwen3(_make_mock_args(quant_target=["mlp"]))
        model.block = lambda layer_idx: nn.Linear(4, 4)
        model.load_selected_layer_ptq_params = lambda layer_idx, block, strict: None
        tensors, routes = export_block_deploy(model, 0, [])
        assert isinstance(tensors, dict)
        assert isinstance(routes, dict)

    def test_qwen3_init_cls_property(self, mock_hf_model):
        from amct_pytorch.common.models.llm.qwen.qwen3.qwen3 import Qwen3
        from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer
        model = Qwen3(_make_mock_args())
        assert model.cls is Qwen3DecoderLayer

    def test_qwen3_num_layers_property(self, mock_hf_model):
        from amct_pytorch.common.models.llm.qwen.qwen3.qwen3 import Qwen3
        model = Qwen3(_make_mock_args())
        assert model.num_layers == 2

    def test_qwen3_textconfig_property(self, mock_hf_model):
        from amct_pytorch.common.models.llm.qwen.qwen3.qwen3 import Qwen3
        from transformers.models.qwen3.modeling_qwen3 import Qwen3Config
        model = Qwen3(_make_mock_args())
        assert model.textconfig is Qwen3Config


class TestQwen3MoeMocked:
    def test_qwen3_moe_init_mocked(self, monkeypatch):
        from amct_pytorch.common.models.llm.qwen.qwen3.qwen3_moe import Qwen3Moe
        from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeDecoderLayer, Qwen3MoeConfig
        monkeypatch.setattr(
            "amct_pytorch.common.models.llm.qwen.qwen3.qwen3_moe.get_weight_mappings",
            lambda path: {},
        )
        cfg = Qwen3MoeConfig(
            num_hidden_layers=2, hidden_size=64, intermediate_size=128,
            num_attention_heads=4, num_key_value_heads=2, head_dim=16,
            rms_norm_eps=1e-6, vocab_size=100, max_position_embeddings=512,
            rope_theta=10000.0, tie_word_embeddings=False,
            num_experts=4, moe_intermediate_size=64,
            n_routed_experts=4, n_shared_experts=0,
        )
        monkeypatch.setattr(
            "amct_pytorch.common.models.llm.common.base.AutoConfig",
            type("FakeAC", (), {"from_pretrained": staticmethod(lambda *a, **kw: cfg)})(),
        )
        fake_tokenizer = MagicMock()
        monkeypatch.setattr(
            "amct_pytorch.common.models.llm.common.base.AutoTokenizer",
            type("FakeAT", (), {"from_pretrained": staticmethod(lambda *a, **kw: fake_tokenizer)})(),
        )
        monkeypatch.setattr(
            "amct_pytorch.common.models.llm.common.base.init_empty_weights",
            lambda: MagicMock(__enter__=lambda s: None, __exit__=lambda *a: None),
        )
        from transformers import AutoModelForCausalLM
        with init_empty_weights():
            empty_model = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True, torch_dtype=torch.bfloat16)
        monkeypatch.setattr(
            "amct_pytorch.common.models.llm.common.base.AutoModelForCausalLM",
            type("FakeAMFCLM", (), {
                "from_pretrained": staticmethod(lambda *a, **kw: empty_model),
                "from_config": staticmethod(lambda *a, **kw: empty_model),
            })(),
        )
        monkeypatch.setattr(
            "amct_pytorch.common.models.llm.common.base.get_weight_mappings",
            lambda path: {},
        )
        args = _make_mock_args(quant_target=["moe"])
        model = Qwen3Moe(args)
        assert model.num_layers == 2
        assert model.cls is Qwen3MoeDecoderLayer
        assert model.config.n_routed_experts > 0

    def test_qwen3_moe_parse_quant_mode_rejects_mlp(self, monkeypatch):
        from amct_pytorch.common.models.llm.qwen.qwen3.qwen3_moe import Qwen3Moe
        from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeConfig
        monkeypatch.setattr(
            "amct_pytorch.common.models.llm.qwen.qwen3.qwen3_moe.get_weight_mappings",
            lambda path: {},
        )
        cfg = Qwen3MoeConfig(
            num_hidden_layers=2, hidden_size=64, intermediate_size=128,
            num_attention_heads=4, num_key_value_heads=2, head_dim=16,
            rms_norm_eps=1e-6, vocab_size=100, max_position_embeddings=512,
            rope_theta=10000.0, tie_word_embeddings=False,
            num_experts=4, moe_intermediate_size=64,
            n_routed_experts=4, n_shared_experts=0,
        )
        monkeypatch.setattr(
            "amct_pytorch.common.models.llm.common.base.AutoConfig",
            type("FakeAC", (), {"from_pretrained": staticmethod(lambda *a, **kw: cfg)})(),
        )
        fake_tokenizer = MagicMock()
        monkeypatch.setattr(
            "amct_pytorch.common.models.llm.common.base.AutoTokenizer",
            type("FakeAT", (), {"from_pretrained": staticmethod(lambda *a, **kw: fake_tokenizer)})(),
        )
        monkeypatch.setattr(
            "amct_pytorch.common.models.llm.common.base.init_empty_weights",
            lambda: MagicMock(__enter__=lambda s: None, __exit__=lambda *a: None),
        )
        from transformers import AutoModelForCausalLM
        with init_empty_weights():
            empty_model = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True, torch_dtype=torch.bfloat16)
        monkeypatch.setattr(
            "amct_pytorch.common.models.llm.common.base.AutoModelForCausalLM",
            type("FakeAMFCLM", (), {
                "from_pretrained": staticmethod(lambda *a, **kw: empty_model),
                "from_config": staticmethod(lambda *a, **kw: empty_model),
            })(),
        )
        monkeypatch.setattr(
            "amct_pytorch.common.models.llm.common.base.get_weight_mappings",
            lambda path: {},
        )
        args = _make_mock_args(quant_target=["mlp"], moe_mlp_param_dir="")
        with pytest.raises(ValueError, match="does not support quant_target='mlp'"):
            Qwen3Moe(args)

    def test_qwen3_moe_num_experts(self, monkeypatch):
        from amct_pytorch.common.models.llm.qwen.qwen3.qwen3_moe import Qwen3Moe
        from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeConfig
        monkeypatch.setattr(
            "amct_pytorch.common.models.llm.qwen.qwen3.qwen3_moe.get_weight_mappings",
            lambda path: {},
        )
        cfg = Qwen3MoeConfig(
            num_hidden_layers=2, hidden_size=64, intermediate_size=128,
            num_attention_heads=4, num_key_value_heads=2, head_dim=16,
            rms_norm_eps=1e-6, vocab_size=100, max_position_embeddings=512,
            rope_theta=10000.0, tie_word_embeddings=False,
            num_experts=8, moe_intermediate_size=64,
            n_routed_experts=8, n_shared_experts=0,
        )
        _mock_base_deps(monkeypatch, cfg)
        args = _make_mock_args(quant_target=["moe"])
        model = Qwen3Moe(args)
        assert model.config.num_experts == 8

    def test_qwen3_moe_get_layer_weight_prefix(self, monkeypatch):
        from amct_pytorch.common.models.llm.qwen.qwen3.qwen3_moe import Qwen3Moe
        from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeConfig
        monkeypatch.setattr(
            "amct_pytorch.common.models.llm.qwen.qwen3.qwen3_moe.get_weight_mappings",
            lambda path: {},
        )
        cfg = Qwen3MoeConfig(
            num_hidden_layers=2, hidden_size=64, intermediate_size=128,
            num_attention_heads=4, num_key_value_heads=2, head_dim=16,
            rms_norm_eps=1e-6, vocab_size=100, max_position_embeddings=512,
            rope_theta=10000.0, tie_word_embeddings=False,
            num_experts=4, moe_intermediate_size=64,
            n_routed_experts=4, n_shared_experts=0,
        )
        _mock_base_deps(monkeypatch, cfg)
        args = _make_mock_args(quant_target=["moe"])
        model = Qwen3Moe(args)
        assert model.get_layer_weight_prefix(0) == "model.layers.0."

    def test_qwen3_moe_load_layer_weight_packs_experts(self, monkeypatch):
        from amct_pytorch.common.models.llm.qwen.qwen3.qwen3_moe import Qwen3Moe
        from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeConfig
        monkeypatch.setattr(
            "amct_pytorch.common.models.llm.qwen.qwen3.qwen3_moe.get_weight_mappings",
            lambda path: {},
        )
        cfg = Qwen3MoeConfig(
            num_hidden_layers=2, hidden_size=64, intermediate_size=128,
            num_attention_heads=4, num_key_value_heads=2, head_dim=16,
            rms_norm_eps=1e-6, vocab_size=100, max_position_embeddings=512,
            rope_theta=10000.0, tie_word_embeddings=False,
            num_experts=4, moe_intermediate_size=64,
            n_routed_experts=4, n_shared_experts=0,
        )
        _mock_base_deps(monkeypatch, cfg)
        args = _make_mock_args(quant_target=["moe"])
        model = Qwen3Moe(args)
        raw_sd = {
            "mlp.experts.0.gate_proj.weight": torch.randn(8, 8),
            "mlp.experts.0.up_proj.weight": torch.randn(8, 8),
            "mlp.experts.0.down_proj.weight": torch.randn(8, 8),
            "other.weight": torch.randn(4, 4),
        }
        monkeypatch.setattr(
            "amct_pytorch.common.models.llm.common.base.BaseModel.load_layer_weight",
            lambda self, prefix: dict(raw_sd),
        )
        result = model.load_layer_weight("model.layers.0.")
        assert "mlp.experts.gate_up_proj" in result
        assert "mlp.experts.down_proj" in result
        assert "other.weight" in result

    def test_qwen3_moe_build_quant_block_moe_with_experts(self, monkeypatch):
        from amct_pytorch.common.models.llm.qwen.qwen3.qwen3_moe import Qwen3Moe
        from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeConfig
        monkeypatch.setattr(
            "amct_pytorch.common.models.llm.qwen.qwen3.qwen3_moe.get_weight_mappings",
            lambda path: {},
        )
        cfg = Qwen3MoeConfig(
            num_hidden_layers=2, hidden_size=64, intermediate_size=128,
            num_attention_heads=4, num_key_value_heads=2, head_dim=16,
            rms_norm_eps=1e-6, vocab_size=100, max_position_embeddings=512,
            rope_theta=10000.0, tie_word_embeddings=False,
            num_experts=4, moe_intermediate_size=64,
            n_routed_experts=4, n_shared_experts=0,
        )
        _mock_base_deps(monkeypatch, cfg)
        args = _make_mock_args(quant_target=["moe"])
        model = Qwen3Moe(args)
        dl = nn.Module()
        mlp = nn.Module()
        mlp.experts = nn.Module()
        dl.mlp = mlp
        monkeypatch.setattr(model, "block", lambda idx: dl)
        monkeypatch.setattr(
            "amct_pytorch.common.models.llm.qwen.qwen3.qwen3_moe.QuantGatedExperts",
            lambda args_in, experts_in: experts_in,
        )
        result = model.build_quant_block(0)
        assert result is dl

    def test_qwen3_moe_build_quant_block_moe_no_experts_but_mlp(self, monkeypatch):
        from amct_pytorch.common.models.llm.qwen.qwen3.qwen3_moe import Qwen3Moe
        from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeConfig
        monkeypatch.setattr(
            "amct_pytorch.common.models.llm.qwen.qwen3.qwen3_moe.get_weight_mappings",
            lambda path: {},
        )
        cfg = Qwen3MoeConfig(
            num_hidden_layers=2, hidden_size=64, intermediate_size=128,
            num_attention_heads=4, num_key_value_heads=2, head_dim=16,
            rms_norm_eps=1e-6, vocab_size=100, max_position_embeddings=512,
            rope_theta=10000.0, tie_word_embeddings=False,
            num_experts=4, moe_intermediate_size=64,
            n_routed_experts=4, n_shared_experts=0,
        )
        _mock_base_deps(monkeypatch, cfg)
        args = _make_mock_args(quant_target=["moe"])
        model = Qwen3Moe(args)
        dl = nn.Module()
        dl.mlp = nn.Linear(4, 4)
        monkeypatch.setattr(model, "block", lambda idx: dl)
        captured = {}
        def _mock_mlp(args_in, module):
            captured["called"] = True
            captured["module"] = module
            return module
        monkeypatch.setattr(
            "amct_pytorch.common.models.llm.qwen.qwen3.qwen3_moe.QuantQwen3MLP",
            _mock_mlp,
        )
        result = model.build_quant_block(0)
        assert captured.get("called") is True

    def test_qwen3_moe_build_quant_block_moe_no_mlp(self, monkeypatch):
        from amct_pytorch.common.models.llm.qwen.qwen3.qwen3_moe import Qwen3Moe
        from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeConfig
        monkeypatch.setattr(
            "amct_pytorch.common.models.llm.qwen.qwen3.qwen3_moe.get_weight_mappings",
            lambda path: {},
        )
        cfg = Qwen3MoeConfig(
            num_hidden_layers=2, hidden_size=64, intermediate_size=128,
            num_attention_heads=4, num_key_value_heads=2, head_dim=16,
            rms_norm_eps=1e-6, vocab_size=100, max_position_embeddings=512,
            rope_theta=10000.0, tie_word_embeddings=False,
            num_experts=4, moe_intermediate_size=64,
            n_routed_experts=4, n_shared_experts=0,
        )
        _mock_base_deps(monkeypatch, cfg)
        args = _make_mock_args(quant_target=["moe"])
        model = Qwen3Moe(args)
        dl = nn.Module()
        monkeypatch.setattr(model, "block", lambda idx: dl)
        result = model.build_quant_block(0)
        assert result is dl

    def test_qwen3_moe_iter_deploy_bindings_expert_name(self, monkeypatch):
        from amct_pytorch.common.models.llm.qwen.qwen3.qwen3_moe import Qwen3Moe
        from amct_pytorch.quantization.modules.quant_linear import QuantLinear
        from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeConfig
        monkeypatch.setattr(
            "amct_pytorch.common.models.llm.qwen.qwen3.qwen3_moe.get_weight_mappings",
            lambda path: {},
        )
        cfg = Qwen3MoeConfig(
            num_hidden_layers=2, hidden_size=64, intermediate_size=128,
            num_attention_heads=4, num_key_value_heads=2, head_dim=16,
            rms_norm_eps=1e-6, vocab_size=100, max_position_embeddings=512,
            rope_theta=10000.0, tie_word_embeddings=False,
            num_experts=4, moe_intermediate_size=64,
            n_routed_experts=4, n_shared_experts=0,
        )
        _mock_base_deps(monkeypatch, cfg)
        args = _make_mock_args(quant_target=["moe"])
        model = Qwen3Moe(args)
        ql = QuantLinear.__new__(QuantLinear)
        ql._parameters = {}
        ql._modules = {}
        ql._buffers = {}
        block = nn.Module()
        block.mlp = nn.Module()
        block.mlp.experts = nn.Module()
        expert0 = nn.Module()
        expert1 = nn.Module()
        expert1.gate_proj = ql
        block.mlp.experts.expert_modules = nn.ModuleList([expert0, expert1])
        bindings = list(model.iter_deploy_bindings(0, block))
        assert len(bindings) == 1
        assert bindings[0][0] == "model.layers.0.mlp.experts.1.gate_proj.weight"

    def test_qwen3_moe_iter_deploy_bindings_normal(self, monkeypatch):
        from amct_pytorch.common.models.llm.qwen.qwen3.qwen3_moe import Qwen3Moe
        from amct_pytorch.quantization.modules.quant_linear import QuantLinear
        from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeConfig
        monkeypatch.setattr(
            "amct_pytorch.common.models.llm.qwen.qwen3.qwen3_moe.get_weight_mappings",
            lambda path: {},
        )
        cfg = Qwen3MoeConfig(
            num_hidden_layers=2, hidden_size=64, intermediate_size=128,
            num_attention_heads=4, num_key_value_heads=2, head_dim=16,
            rms_norm_eps=1e-6, vocab_size=100, max_position_embeddings=512,
            rope_theta=10000.0, tie_word_embeddings=False,
            num_experts=4, moe_intermediate_size=64,
            n_routed_experts=4, n_shared_experts=0,
        )
        _mock_base_deps(monkeypatch, cfg)
        args = _make_mock_args(quant_target=["moe"])
        model = Qwen3Moe(args)
        ql = QuantLinear.__new__(QuantLinear)
        ql._parameters = {}
        ql._modules = {}
        ql._buffers = {}
        block = nn.Module()
        block.q_proj = ql
        bindings = list(model.iter_deploy_bindings(0, block))
        assert len(bindings) == 1
        assert bindings[0][0] == "model.layers.0.q_proj.weight"

    def test_qwen3_moe_iter_deploy_bindings_invalid_expert_name(self, monkeypatch):
        from amct_pytorch.common.models.llm.qwen.qwen3.qwen3_moe import Qwen3Moe
        from amct_pytorch.quantization.modules.quant_linear import QuantLinear
        from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeConfig
        monkeypatch.setattr(
            "amct_pytorch.common.models.llm.qwen.qwen3.qwen3_moe.get_weight_mappings",
            lambda path: {},
        )
        cfg = Qwen3MoeConfig(
            num_hidden_layers=2, hidden_size=64, intermediate_size=128,
            num_attention_heads=4, num_key_value_heads=2, head_dim=16,
            rms_norm_eps=1e-6, vocab_size=100, max_position_embeddings=512,
            rope_theta=10000.0, tie_word_embeddings=False,
            num_experts=4, moe_intermediate_size=64,
            n_routed_experts=4, n_shared_experts=0,
        )
        _mock_base_deps(monkeypatch, cfg)
        args = _make_mock_args(quant_target=["moe"])
        model = Qwen3Moe(args)
        ql = QuantLinear.__new__(QuantLinear)
        ql._parameters = {}
        ql._modules = {}
        ql._buffers = {}
        block = nn.Module()
        block.mlp = nn.Module()
        block.mlp.experts = nn.Module()
        expert0 = nn.Module()
        expert0.extra = nn.Module()
        expert0.extra.proj = ql
        block.mlp.experts.expert_modules = nn.ModuleList([expert0])
        with pytest.raises(ValueError, match="Unexpected Qwen3 MoE expert module name"):
            list(model.iter_deploy_bindings(0, block))

    def test_qwen3_moe_skips_non_quant_linear(self, monkeypatch):
        from amct_pytorch.common.models.llm.qwen.qwen3.qwen3_moe import Qwen3Moe
        from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeConfig
        monkeypatch.setattr(
            "amct_pytorch.common.models.llm.qwen.qwen3.qwen3_moe.get_weight_mappings",
            lambda path: {},
        )
        cfg = Qwen3MoeConfig(
            num_hidden_layers=2, hidden_size=64, intermediate_size=128,
            num_attention_heads=4, num_key_value_heads=2, head_dim=16,
            rms_norm_eps=1e-6, vocab_size=100, max_position_embeddings=512,
            rope_theta=10000.0, tie_word_embeddings=False,
            num_experts=4, moe_intermediate_size=64,
            n_routed_experts=4, n_shared_experts=0,
        )
        _mock_base_deps(monkeypatch, cfg)
        args = _make_mock_args(quant_target=["moe"])
        model = Qwen3Moe(args)
        block = nn.Module()
        block.linear = nn.Linear(4, 4)
        bindings = list(model.iter_deploy_bindings(0, block))
        assert bindings == []

    def test_qwen3_moe_float_model_delegates(self, monkeypatch):
        from amct_pytorch.common.models.llm.qwen.qwen3.qwen3_moe import Qwen3Moe
        from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeConfig
        monkeypatch.setattr(
            "amct_pytorch.common.models.llm.qwen.qwen3.qwen3_moe.get_weight_mappings",
            lambda path: {},
        )
        cfg = Qwen3MoeConfig(
            num_hidden_layers=2, hidden_size=64, intermediate_size=128,
            num_attention_heads=4, num_key_value_heads=2, head_dim=16,
            rms_norm_eps=1e-6, vocab_size=100, max_position_embeddings=512,
            rope_theta=10000.0, tie_word_embeddings=False,
            num_experts=4, moe_intermediate_size=64,
            n_routed_experts=4, n_shared_experts=0,
        )
        _mock_base_deps(monkeypatch, cfg)
        model = Qwen3Moe(_make_mock_args(quant_target=["moe"]))
        mock_result = MagicMock()
        monkeypatch.setattr(
            "amct_pytorch.common.models.llm.common.base.BaseModel.float_model",
            lambda self: mock_result,
        )
        assert model.float_model() is mock_result

    def test_qwen3_moe_load_embed_state_dict_delegates(self, monkeypatch):
        from amct_pytorch.common.models.llm.qwen.qwen3.qwen3_moe import Qwen3Moe
        from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeConfig
        monkeypatch.setattr(
            "amct_pytorch.common.models.llm.qwen.qwen3.qwen3_moe.get_weight_mappings",
            lambda path: {},
        )
        cfg = Qwen3MoeConfig(
            num_hidden_layers=2, hidden_size=64, intermediate_size=128,
            num_attention_heads=4, num_key_value_heads=2, head_dim=16,
            rms_norm_eps=1e-6, vocab_size=100, max_position_embeddings=512,
            rope_theta=10000.0, tie_word_embeddings=False,
            num_experts=4, moe_intermediate_size=64,
            n_routed_experts=4, n_shared_experts=0,
        )
        _mock_base_deps(monkeypatch, cfg)
        model = Qwen3Moe(_make_mock_args(quant_target=["moe"]))
        monkeypatch.setattr(
            "amct_pytorch.common.models.llm.common.base.BaseModel.load_embed_state_dict",
            lambda self: None,
        )
        model.load_embed_state_dict()

    def test_qwen3_moe_block_delegates(self, monkeypatch):
        from amct_pytorch.common.models.llm.qwen.qwen3.qwen3_moe import Qwen3Moe
        from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeConfig
        monkeypatch.setattr(
            "amct_pytorch.common.models.llm.qwen.qwen3.qwen3_moe.get_weight_mappings",
            lambda path: {},
        )
        cfg = Qwen3MoeConfig(
            num_hidden_layers=2, hidden_size=64, intermediate_size=128,
            num_attention_heads=4, num_key_value_heads=2, head_dim=16,
            rms_norm_eps=1e-6, vocab_size=100, max_position_embeddings=512,
            rope_theta=10000.0, tie_word_embeddings=False,
            num_experts=4, moe_intermediate_size=64,
            n_routed_experts=4, n_shared_experts=0,
        )
        _mock_base_deps(monkeypatch, cfg)
        model = Qwen3Moe(_make_mock_args(quant_target=["moe"]))
        mock_result = MagicMock()
        monkeypatch.setattr(
            "amct_pytorch.common.models.llm.common.base.BaseModel.block",
            lambda self, idx: mock_result,
        )
        assert model.block(0) is mock_result

    def test_qwen3_moe_do_embedding_forward_delegates(self, monkeypatch):
        from amct_pytorch.common.models.llm.qwen.qwen3.qwen3_moe import Qwen3Moe
        from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeConfig
        monkeypatch.setattr(
            "amct_pytorch.common.models.llm.qwen.qwen3.qwen3_moe.get_weight_mappings",
            lambda path: {},
        )
        cfg = Qwen3MoeConfig(
            num_hidden_layers=2, hidden_size=64, intermediate_size=128,
            num_attention_heads=4, num_key_value_heads=2, head_dim=16,
            rms_norm_eps=1e-6, vocab_size=100, max_position_embeddings=512,
            rope_theta=10000.0, tie_word_embeddings=False,
            num_experts=4, moe_intermediate_size=64,
            n_routed_experts=4, n_shared_experts=0,
        )
        _mock_base_deps(monkeypatch, cfg)
        model = Qwen3Moe(_make_mock_args(quant_target=["moe"]))
        mock_result = [torch.randn(2, 4, 8)]
        monkeypatch.setattr(
            "amct_pytorch.common.models.llm.common.base.BaseModel.do_embedding_forward",
            lambda self, samples, dtype=torch.bfloat16, hook_name=None: mock_result,
        )
        result = model.do_embedding_forward([torch.randn(2, 4)])
        assert result is mock_result

    def test_qwen3_moe_do_block_forward_delegates(self, monkeypatch):
        from amct_pytorch.common.models.llm.qwen.qwen3.qwen3_moe import Qwen3Moe
        from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeConfig
        monkeypatch.setattr(
            "amct_pytorch.common.models.llm.qwen.qwen3.qwen3_moe.get_weight_mappings",
            lambda path: {},
        )
        cfg = Qwen3MoeConfig(
            num_hidden_layers=2, hidden_size=64, intermediate_size=128,
            num_attention_heads=4, num_key_value_heads=2, head_dim=16,
            rms_norm_eps=1e-6, vocab_size=100, max_position_embeddings=512,
            rope_theta=10000.0, tie_word_embeddings=False,
            num_experts=4, moe_intermediate_size=64,
            n_routed_experts=4, n_shared_experts=0,
        )
        _mock_base_deps(monkeypatch, cfg)
        model = Qwen3Moe(_make_mock_args(quant_target=["moe"]))
        mock_result = [torch.randn(2, 4, 8)]
        monkeypatch.setattr(
            "amct_pytorch.common.models.llm.common.base.BaseModel.do_block_forward",
            lambda self, layer_idx, samples, hook_name=None, use_quant_block=False,
            enable_quant=False: mock_result,
        )
        result = model.do_block_forward(0, [torch.randn(2, 4, 8)])
        assert result is mock_result

    def test_qwen3_moe_do_head_forward_delegates(self, monkeypatch):
        from amct_pytorch.common.models.llm.qwen.qwen3.qwen3_moe import Qwen3Moe
        from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeConfig
        monkeypatch.setattr(
            "amct_pytorch.common.models.llm.qwen.qwen3.qwen3_moe.get_weight_mappings",
            lambda path: {},
        )
        cfg = Qwen3MoeConfig(
            num_hidden_layers=2, hidden_size=64, intermediate_size=128,
            num_attention_heads=4, num_key_value_heads=2, head_dim=16,
            rms_norm_eps=1e-6, vocab_size=100, max_position_embeddings=512,
            rope_theta=10000.0, tie_word_embeddings=False,
            num_experts=4, moe_intermediate_size=64,
            n_routed_experts=4, n_shared_experts=0,
        )
        _mock_base_deps(monkeypatch, cfg)
        model = Qwen3Moe(_make_mock_args(quant_target=["moe"]))
        mock_result = [torch.randn(2, 4, 100)]
        monkeypatch.setattr(
            "amct_pytorch.common.models.llm.common.base.BaseModel.do_head_forward",
            lambda self, inps: mock_result,
        )
        result = model.do_head_forward([torch.randn(2, 4, 8)])
        assert result is mock_result

    def test_qwen3_moe_build_quant_block_attn_linear(self, monkeypatch):
        from amct_pytorch.common.models.llm.qwen.qwen3.qwen3_moe import Qwen3Moe
        from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeConfig
        monkeypatch.setattr(
            "amct_pytorch.common.models.llm.qwen.qwen3.qwen3_moe.get_weight_mappings",
            lambda path: {},
        )
        cfg = Qwen3MoeConfig(
            num_hidden_layers=2, hidden_size=64, intermediate_size=128,
            num_attention_heads=4, num_key_value_heads=2, head_dim=16,
            rms_norm_eps=1e-6, vocab_size=100, max_position_embeddings=512,
            rope_theta=10000.0, tie_word_embeddings=False,
            num_experts=4, moe_intermediate_size=64,
            n_routed_experts=4, n_shared_experts=0,
        )
        _mock_base_deps(monkeypatch, cfg)
        args = _make_mock_args(quant_target=["attn-linear"])
        model = Qwen3Moe(args)
        dl = nn.Module()
        monkeypatch.setattr(model, "block", lambda idx: dl)
        captured = {}
        def _mock_apply(args_in, dl_in, attn_cls):
            captured["called"] = True
        monkeypatch.setattr(
            "amct_pytorch.common.models.llm.qwen.qwen3.qwen3_moe.apply_quant_to_attn",
            _mock_apply,
        )
        result = model.build_quant_block(0)
        assert captured.get("called") is True
        assert result is dl

    def test_qwen3_moe_build_quant_block_attn_cache(self, monkeypatch):
        from amct_pytorch.common.models.llm.qwen.qwen3.qwen3_moe import Qwen3Moe
        from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeConfig
        monkeypatch.setattr(
            "amct_pytorch.common.models.llm.qwen.qwen3.qwen3_moe.get_weight_mappings",
            lambda path: {},
        )
        cfg = Qwen3MoeConfig(
            num_hidden_layers=2, hidden_size=64, intermediate_size=128,
            num_attention_heads=4, num_key_value_heads=2, head_dim=16,
            rms_norm_eps=1e-6, vocab_size=100, max_position_embeddings=512,
            rope_theta=10000.0, tie_word_embeddings=False,
            num_experts=4, moe_intermediate_size=64,
            n_routed_experts=4, n_shared_experts=0,
        )
        _mock_base_deps(monkeypatch, cfg)
        args = _make_mock_args(quant_target=["attn-cache"])
        model = Qwen3Moe(args)
        dl = nn.Module()
        monkeypatch.setattr(model, "block", lambda idx: dl)
        captured = {}
        def _mock_apply(args_in, dl_in, attn_cls):
            captured["called"] = True
        monkeypatch.setattr(
            "amct_pytorch.common.models.llm.qwen.qwen3.qwen3_moe.apply_quant_to_attn",
            _mock_apply,
        )
        result = model.build_quant_block(0)
        assert captured.get("called") is True
        assert result is dl

    def test_qwen3_moe_iter_ptq_units_delegates(self, monkeypatch):
        from amct_pytorch.common.models.llm.qwen.qwen3.qwen3_moe import Qwen3Moe
        from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeConfig
        monkeypatch.setattr(
            "amct_pytorch.common.models.llm.qwen.qwen3.qwen3_moe.get_weight_mappings",
            lambda path: {},
        )
        cfg = Qwen3MoeConfig(
            num_hidden_layers=2, hidden_size=64, intermediate_size=128,
            num_attention_heads=4, num_key_value_heads=2, head_dim=16,
            rms_norm_eps=1e-6, vocab_size=100, max_position_embeddings=512,
            rope_theta=10000.0, tie_word_embeddings=False,
            num_experts=4, moe_intermediate_size=64,
            n_routed_experts=4, n_shared_experts=0,
        )
        _mock_base_deps(monkeypatch, cfg)
        model = Qwen3Moe(_make_mock_args(quant_target=["moe"]))
        block = nn.Module()
        block.mlp = nn.Module()
        block.mlp.experts = nn.ModuleList([nn.Linear(4, 4)])
        units = list(model.iter_ptq_units(0, block))
        assert len(units) >= 1

    def test_qwen3_moe_load_unit_inputs_delegates(self, monkeypatch):
        from amct_pytorch.common.models.llm.qwen.qwen3.qwen3_moe import Qwen3Moe
        from amct_pytorch.common.models.llm.common.ptq_units import make_ptq_unit
        from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeConfig
        monkeypatch.setattr(
            "amct_pytorch.common.models.llm.qwen.qwen3.qwen3_moe.get_weight_mappings",
            lambda path: {},
        )
        cfg = Qwen3MoeConfig(
            num_hidden_layers=2, hidden_size=64, intermediate_size=128,
            num_attention_heads=4, num_key_value_heads=2, head_dim=16,
            rms_norm_eps=1e-6, vocab_size=100, max_position_embeddings=512,
            rope_theta=10000.0, tie_word_embeddings=False,
            num_experts=4, moe_intermediate_size=64,
            n_routed_experts=4, n_shared_experts=0,
        )
        _mock_base_deps(monkeypatch, cfg)
        model = Qwen3Moe(_make_mock_args(quant_target=["moe"]))
        unit = make_ptq_unit("moe", "mlp", layer_idx=0, module=nn.Linear(4, 4))
        fake_data = torch.randn(4, 4)
        monkeypatch.setattr(
            "amct_pytorch.common.models.llm.common.base.load_ptq_inps",
            lambda data_dir, kind, layer_idx: fake_data,
        )
        result = model.load_unit_inputs("/tmp/fake", unit)
        assert result is fake_data


def _mock_base_deps(monkeypatch, cfg):
    """Mock BaseModel HF dependencies with a given config."""
    fake_tokenizer = MagicMock()
    monkeypatch.setattr(
        "amct_pytorch.common.models.llm.common.base.AutoConfig",
        type("FakeAC", (), {"from_pretrained": staticmethod(lambda *a, **kw: cfg)})(),
    )
    monkeypatch.setattr(
        "amct_pytorch.common.models.llm.common.base.AutoTokenizer",
        type("FakeAT", (), {"from_pretrained": staticmethod(lambda *a, **kw: fake_tokenizer)})(),
    )
    monkeypatch.setattr(
        "amct_pytorch.common.models.llm.common.base.init_empty_weights",
        lambda: MagicMock(__enter__=lambda s: None, __exit__=lambda *a: None),
    )
    from transformers import AutoModelForCausalLM
    with init_empty_weights():
        empty_model = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True, torch_dtype=torch.bfloat16)
    monkeypatch.setattr(
        "amct_pytorch.common.models.llm.common.base.AutoModelForCausalLM",
        type("FakeAMFCLM", (), {
            "from_pretrained": staticmethod(lambda *a, **kw: empty_model),
            "from_config": staticmethod(lambda *a, **kw: empty_model),
        })(),
    )
    monkeypatch.setattr(
        "amct_pytorch.common.models.llm.common.base.get_weight_mappings",
        lambda path: {},
    )


class TestQwen3NextMocked:
    def test_qwen3_next_init(self, monkeypatch):
        from amct_pytorch.common.models.llm.qwen.qwen3_next.qwen3_next import Qwen3Next
        from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextConfig, Qwen3NextDecoderLayer
        monkeypatch.setattr(
            "amct_pytorch.common.models.llm.qwen.qwen3_next.qwen3_next.get_weight_mappings",
            lambda path: {},
        )
        cfg = Qwen3NextConfig(
            num_hidden_layers=2, hidden_size=64, intermediate_size=128,
            num_attention_heads=4, num_key_value_heads=2,
            head_dim=16, rms_norm_eps=1e-6, vocab_size=100,
            max_position_embeddings=512, rope_theta=10000.0, tie_word_embeddings=False,
        )
        _mock_base_deps(monkeypatch, cfg)
        args = _make_mock_args(quant_target=["moe"])
        model = Qwen3Next(args)
        assert model.num_layers == 2
        assert model.cls is Qwen3NextDecoderLayer

    def test_qwen3_next_parse_quant_mode_rejects_mlp(self, monkeypatch):
        from amct_pytorch.common.models.llm.qwen.qwen3_next.qwen3_next import Qwen3Next
        from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextConfig
        monkeypatch.setattr(
            "amct_pytorch.common.models.llm.qwen.qwen3_next.qwen3_next.get_weight_mappings",
            lambda path: {},
        )
        cfg = Qwen3NextConfig(
            num_hidden_layers=2, hidden_size=64, intermediate_size=128,
            num_attention_heads=4, num_key_value_heads=2,
            head_dim=16, rms_norm_eps=1e-6, vocab_size=100,
            max_position_embeddings=512, rope_theta=10000.0, tie_word_embeddings=False,
        )
        _mock_base_deps(monkeypatch, cfg)
        args = _make_mock_args(quant_target=["mlp"])
        with pytest.raises(ValueError, match="does not support quant_target='mlp'"):
            Qwen3Next(args)

    def test_qwen3_next_export_block_deploy(self, monkeypatch):
        from amct_pytorch.common.models.llm.qwen.qwen3_next.qwen3_next import Qwen3Next
        from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextConfig
        from amct_pytorch.common.models.llm.common.deploy_export import export_block_deploy
        monkeypatch.setattr(
            "amct_pytorch.common.models.llm.qwen.qwen3_next.qwen3_next.get_weight_mappings",
            lambda path: {},
        )
        cfg = Qwen3NextConfig(
            num_hidden_layers=2, hidden_size=64, intermediate_size=128,
            num_attention_heads=4, num_key_value_heads=2,
            head_dim=16, rms_norm_eps=1e-6, vocab_size=100,
            max_position_embeddings=512, rope_theta=10000.0, tie_word_embeddings=False,
        )
        _mock_base_deps(monkeypatch, cfg)
        args = _make_mock_args(quant_target=["moe"])
        model = Qwen3Next(args)
        model.block = lambda layer_idx: nn.Linear(4, 4)
        model.load_selected_layer_ptq_params = lambda layer_idx, block, strict: None
        tensors, routes = export_block_deploy(model, 0, [])
        assert isinstance(tensors, dict)

    def test_qwen3_next_get_layer_weight_prefix(self, monkeypatch):
        from amct_pytorch.common.models.llm.qwen.qwen3_next.qwen3_next import Qwen3Next
        from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextConfig
        monkeypatch.setattr(
            "amct_pytorch.common.models.llm.qwen.qwen3_next.qwen3_next.get_weight_mappings",
            lambda path: {},
        )
        cfg = Qwen3NextConfig(
            num_hidden_layers=2, hidden_size=64, intermediate_size=128,
            num_attention_heads=4, num_key_value_heads=2,
            head_dim=16, rms_norm_eps=1e-6, vocab_size=100,
            max_position_embeddings=512, rope_theta=10000.0, tie_word_embeddings=False,
        )
        _mock_base_deps(monkeypatch, cfg)
        args = _make_mock_args(quant_target=["moe"])
        model = Qwen3Next(args)
        assert model.get_layer_weight_prefix(0) == "model.layers.0."

    def test_qwen3_next_load_layer_weight_packs_experts(self, monkeypatch):
        from amct_pytorch.common.models.llm.qwen.qwen3_next.qwen3_next import Qwen3Next
        from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextConfig
        monkeypatch.setattr(
            "amct_pytorch.common.models.llm.qwen.qwen3_next.qwen3_next.get_weight_mappings",
            lambda path: {},
        )
        cfg = Qwen3NextConfig(
            num_hidden_layers=2, hidden_size=64, intermediate_size=128,
            num_attention_heads=4, num_key_value_heads=2,
            head_dim=16, rms_norm_eps=1e-6, vocab_size=100,
            max_position_embeddings=512, rope_theta=10000.0, tie_word_embeddings=False,
        )
        _mock_base_deps(monkeypatch, cfg)
        args = _make_mock_args(quant_target=["moe"])
        model = Qwen3Next(args)
        raw_sd = {
            "mlp.experts.0.gate_proj.weight": torch.randn(8, 8),
            "mlp.experts.0.up_proj.weight": torch.randn(8, 8),
            "mlp.experts.0.down_proj.weight": torch.randn(8, 8),
            "mlp.experts.1.gate_proj.weight": torch.randn(8, 8),
            "mlp.experts.1.up_proj.weight": torch.randn(8, 8),
            "mlp.experts.1.down_proj.weight": torch.randn(8, 8),
            "linear_attn.rotary_emb.inv_freq": torch.randn(4),
        }
        monkeypatch.setattr(
            "amct_pytorch.common.models.llm.common.base.BaseModel.load_layer_weight",
            lambda self, prefix: dict(raw_sd),
        )
        result = model.load_layer_weight("model.layers.0.")
        assert "mlp.experts.gate_up_proj" in result
        assert "mlp.experts.down_proj" in result
        assert "linear_attn.rotary_emb.inv_freq" not in result

    def test_qwen3_next_block(self, monkeypatch):
        from amct_pytorch.common.models.llm.qwen.qwen3_next.qwen3_next import Qwen3Next
        from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextConfig
        monkeypatch.setattr(
            "amct_pytorch.common.models.llm.qwen.qwen3_next.qwen3_next.get_weight_mappings",
            lambda path: {},
        )
        cfg = Qwen3NextConfig(
            num_hidden_layers=2, hidden_size=64, intermediate_size=128,
            num_attention_heads=4, num_key_value_heads=2,
            head_dim=16, rms_norm_eps=1e-6, vocab_size=100,
            max_position_embeddings=512, rope_theta=10000.0, tie_word_embeddings=False,
        )
        _mock_base_deps(monkeypatch, cfg)
        args = _make_mock_args(quant_target=["moe"])
        model = Qwen3Next(args)
        mock_layer = MagicMock()
        mock_layer.eval.return_value = mock_layer
        mock_layer.bfloat16.return_value = mock_layer
        monkeypatch.setattr(model, "cls", lambda cfg_in, idx: mock_layer)
        monkeypatch.setattr(model, "load_layer_weight", lambda prefix: {})
        block = model.block(0)
        assert block is mock_layer

    def test_qwen3_next_build_quant_block_moe(self, monkeypatch):
        from amct_pytorch.common.models.llm.qwen.qwen3_next.qwen3_next import Qwen3Next
        from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextConfig
        monkeypatch.setattr(
            "amct_pytorch.common.models.llm.qwen.qwen3_next.qwen3_next.get_weight_mappings",
            lambda path: {},
        )
        cfg = Qwen3NextConfig(
            num_hidden_layers=2, hidden_size=64, intermediate_size=128,
            num_attention_heads=4, num_key_value_heads=2,
            head_dim=16, rms_norm_eps=1e-6, vocab_size=100,
            max_position_embeddings=512, rope_theta=10000.0, tie_word_embeddings=False,
        )
        _mock_base_deps(monkeypatch, cfg)
        args = _make_mock_args(quant_target=["moe"])
        model = Qwen3Next(args)
        dl = nn.Module()
        moe = nn.Module()
        moe.experts = type("PE", (), {"gate_up_proj": None, "down_proj": None, "num_experts": 4})()
        dl.mlp = moe
        monkeypatch.setattr(model, "block", lambda idx: dl)
        monkeypatch.setattr(
            "amct_pytorch.common.models.llm.qwen.qwen3_next.qwen3_next.find_moe_module",
            lambda dl_in: dl_in.mlp,
        )
        monkeypatch.setattr(
            "amct_pytorch.common.models.llm.qwen.qwen3_next.qwen3_next.QuantGatedExperts",
            lambda args_in, experts, **kwargs: experts,
        )
        result = model.build_quant_block(0)
        assert result is dl

    def test_qwen3_next_build_quant_block_moe_with_shared_expert(self, monkeypatch):
        from amct_pytorch.common.models.llm.qwen.qwen3_next.qwen3_next import Qwen3Next
        from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextConfig
        monkeypatch.setattr(
            "amct_pytorch.common.models.llm.qwen.qwen3_next.qwen3_next.get_weight_mappings",
            lambda path: {},
        )
        cfg = Qwen3NextConfig(
            num_hidden_layers=2, hidden_size=64, intermediate_size=128,
            num_attention_heads=4, num_key_value_heads=2,
            head_dim=16, rms_norm_eps=1e-6, vocab_size=100,
            max_position_embeddings=512, rope_theta=10000.0, tie_word_embeddings=False,
        )
        _mock_base_deps(monkeypatch, cfg)
        args = _make_mock_args(quant_target=["moe"])
        model = Qwen3Next(args)
        dl = nn.Module()
        moe = nn.Module()
        moe.experts = type("PE", (), {"gate_up_proj": None, "down_proj": None, "num_experts": 4})()
        moe.shared_expert = nn.Linear(4, 4)
        dl.mlp = moe
        monkeypatch.setattr(model, "block", lambda idx: dl)
        monkeypatch.setattr(
            "amct_pytorch.common.models.llm.qwen.qwen3_next.qwen3_next.find_moe_module",
            lambda dl_in: dl_in.mlp,
        )
        monkeypatch.setattr(
            "amct_pytorch.common.models.llm.qwen.qwen3_next.qwen3_next.QuantGatedExperts",
            lambda args_in, experts, **kwargs: experts,
        )
        captured = {}
        class _MockQwen3NextMLP:
            def __new__(cls, args_in, module, **kwargs):
                captured["called"] = True
                return module
        monkeypatch.setattr(
            "amct_pytorch.common.models.llm.qwen.qwen3_next.qwen3_next.QuantQwen3NextMLP",
            _MockQwen3NextMLP,
        )
        result = model.build_quant_block(0)
        assert captured.get("called") is True

    def test_qwen3_next_build_quant_block_moe_no_experts(self, monkeypatch):
        from amct_pytorch.common.models.llm.qwen.qwen3_next.qwen3_next import Qwen3Next
        from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextConfig
        monkeypatch.setattr(
            "amct_pytorch.common.models.llm.qwen.qwen3_next.qwen3_next.get_weight_mappings",
            lambda path: {},
        )
        cfg = Qwen3NextConfig(
            num_hidden_layers=2, hidden_size=64, intermediate_size=128,
            num_attention_heads=4, num_key_value_heads=2,
            head_dim=16, rms_norm_eps=1e-6, vocab_size=100,
            max_position_embeddings=512, rope_theta=10000.0, tie_word_embeddings=False,
        )
        _mock_base_deps(monkeypatch, cfg)
        args = _make_mock_args(quant_target=["moe"])
        model = Qwen3Next(args)
        dl = nn.Linear(4, 4)
        monkeypatch.setattr(model, "block", lambda idx: dl)
        monkeypatch.setattr(
            "amct_pytorch.common.models.llm.qwen.qwen3_next.qwen3_next.find_moe_module",
            lambda dl_in: None,
        )
        result = model.build_quant_block(0)
        assert result is dl

    def test_qwen3_next_apply_quant_attn_linear_attention(self, monkeypatch):
        from amct_pytorch.common.models.llm.qwen.qwen3_next.qwen3_next import Qwen3Next
        from amct_pytorch.common.models.llm.qwen.qwen3_next.quant_module import QuantQwen3NextLinearAttn
        from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextConfig
        monkeypatch.setattr(
            "amct_pytorch.common.models.llm.qwen.qwen3_next.qwen3_next.get_weight_mappings",
            lambda path: {},
        )
        cfg = Qwen3NextConfig(
            num_hidden_layers=2, hidden_size=64, intermediate_size=128,
            num_attention_heads=4, num_key_value_heads=2,
            head_dim=16, rms_norm_eps=1e-6, vocab_size=100,
            max_position_embeddings=512, rope_theta=10000.0, tie_word_embeddings=False,
        )
        _mock_base_deps(monkeypatch, cfg)
        args = _make_mock_args(quant_target=["attn-linear"])
        model = Qwen3Next(args)
        dl = nn.Module()
        dl.layer_type = "linear_attention"
        dl.linear_attn = nn.Module()
        captured = {}
        def _mock(args_in, dl_in, attn_cls):
            captured["attn_cls"] = attn_cls
        monkeypatch.setattr(
            "amct_pytorch.common.models.llm.qwen.qwen3_next.qwen3_next.apply_quant_to_attn", _mock
        )
        model.apply_quant_attn(dl)
        assert captured["attn_cls"] is QuantQwen3NextLinearAttn

    def test_qwen3_next_apply_quant_attn_default(self, monkeypatch):
        from amct_pytorch.common.models.llm.qwen.qwen3_next.qwen3_next import Qwen3Next
        from amct_pytorch.common.models.llm.qwen.qwen3_next.quant_module import QuantQwen3NextAttn
        from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextConfig
        monkeypatch.setattr(
            "amct_pytorch.common.models.llm.qwen.qwen3_next.qwen3_next.get_weight_mappings",
            lambda path: {},
        )
        cfg = Qwen3NextConfig(
            num_hidden_layers=2, hidden_size=64, intermediate_size=128,
            num_attention_heads=4, num_key_value_heads=2,
            head_dim=16, rms_norm_eps=1e-6, vocab_size=100,
            max_position_embeddings=512, rope_theta=10000.0, tie_word_embeddings=False,
        )
        _mock_base_deps(monkeypatch, cfg)
        args = _make_mock_args(quant_target=["attn-linear"])
        model = Qwen3Next(args)
        dl = nn.Module()
        dl.layer_type = "default_attention"
        captured = {}
        def _mock(args_in, dl_in, attn_cls):
            captured["attn_cls"] = attn_cls
        monkeypatch.setattr(
            "amct_pytorch.common.models.llm.qwen.qwen3_next.qwen3_next.apply_quant_to_attn", _mock
        )
        model.apply_quant_attn(dl)
        assert captured["attn_cls"] is QuantQwen3NextAttn

    def test_qwen3_next_iter_deploy_bindings_expert_name(self, monkeypatch):
        from amct_pytorch.common.models.llm.qwen.qwen3_next.qwen3_next import Qwen3Next
        from amct_pytorch.quantization.modules.quant_linear import QuantLinear
        from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextConfig
        monkeypatch.setattr(
            "amct_pytorch.common.models.llm.qwen.qwen3_next.qwen3_next.get_weight_mappings",
            lambda path: {},
        )
        cfg = Qwen3NextConfig(
            num_hidden_layers=2, hidden_size=64, intermediate_size=128,
            num_attention_heads=4, num_key_value_heads=2,
            head_dim=16, rms_norm_eps=1e-6, vocab_size=100,
            max_position_embeddings=512, rope_theta=10000.0, tie_word_embeddings=False,
        )
        _mock_base_deps(monkeypatch, cfg)
        args = _make_mock_args(quant_target=["moe"])
        model = Qwen3Next(args)
        ql = QuantLinear.__new__(QuantLinear)
        ql._parameters = {}
        ql._modules = {}
        ql._buffers = {}
        block = nn.Module()
        block.mlp = nn.Module()
        block.mlp.experts = nn.Module()
        expert0 = nn.Module()
        expert1 = nn.Module()
        expert1.up_proj = ql
        block.mlp.experts.expert_modules = nn.ModuleList([expert0, expert1])
        bindings = list(model.iter_deploy_bindings(0, block))
        assert len(bindings) == 1
        assert bindings[0][0] == "model.layers.0.mlp.experts.1.up_proj.weight"

    def test_qwen3_next_iter_deploy_bindings_normal(self, monkeypatch):
        from amct_pytorch.common.models.llm.qwen.qwen3_next.qwen3_next import Qwen3Next
        from amct_pytorch.quantization.modules.quant_linear import QuantLinear
        from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextConfig
        monkeypatch.setattr(
            "amct_pytorch.common.models.llm.qwen.qwen3_next.qwen3_next.get_weight_mappings",
            lambda path: {},
        )
        cfg = Qwen3NextConfig(
            num_hidden_layers=2, hidden_size=64, intermediate_size=128,
            num_attention_heads=4, num_key_value_heads=2,
            head_dim=16, rms_norm_eps=1e-6, vocab_size=100,
            max_position_embeddings=512, rope_theta=10000.0, tie_word_embeddings=False,
        )
        _mock_base_deps(monkeypatch, cfg)
        args = _make_mock_args(quant_target=["moe"])
        model = Qwen3Next(args)
        ql = QuantLinear.__new__(QuantLinear)
        ql._parameters = {}
        ql._modules = {}
        ql._buffers = {}
        block = nn.Module()
        block.q_proj = ql
        bindings = list(model.iter_deploy_bindings(0, block))
        assert len(bindings) == 1
        assert bindings[0][0] == "model.layers.0.q_proj.weight"

    def test_qwen3_next_iter_deploy_bindings_invalid_expert_name(self, monkeypatch):
        from amct_pytorch.common.models.llm.qwen.qwen3_next.qwen3_next import Qwen3Next
        from amct_pytorch.quantization.modules.quant_linear import QuantLinear
        from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextConfig
        monkeypatch.setattr(
            "amct_pytorch.common.models.llm.qwen.qwen3_next.qwen3_next.get_weight_mappings",
            lambda path: {},
        )
        cfg = Qwen3NextConfig(
            num_hidden_layers=2, hidden_size=64, intermediate_size=128,
            num_attention_heads=4, num_key_value_heads=2,
            head_dim=16, rms_norm_eps=1e-6, vocab_size=100,
            max_position_embeddings=512, rope_theta=10000.0, tie_word_embeddings=False,
        )
        _mock_base_deps(monkeypatch, cfg)
        args = _make_mock_args(quant_target=["moe"])
        model = Qwen3Next(args)
        ql = QuantLinear.__new__(QuantLinear)
        ql._parameters = {}
        ql._modules = {}
        ql._buffers = {}
        block = nn.Module()
        block.mlp = nn.Module()
        block.mlp.experts = nn.Module()
        expert0 = nn.Module()
        expert0.extra = nn.Module()
        expert0.extra.proj = ql
        block.mlp.experts.expert_modules = nn.ModuleList([expert0])
        with pytest.raises(ValueError, match="Unexpected Qwen3-Next expert module name"):
            list(model.iter_deploy_bindings(0, block))

    def test_qwen3_next_skips_non_quant_linear(self, monkeypatch):
        from amct_pytorch.common.models.llm.qwen.qwen3_next.qwen3_next import Qwen3Next
        from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextConfig
        monkeypatch.setattr(
            "amct_pytorch.common.models.llm.qwen.qwen3_next.qwen3_next.get_weight_mappings",
            lambda path: {},
        )
        cfg = Qwen3NextConfig(
            num_hidden_layers=2, hidden_size=64, intermediate_size=128,
            num_attention_heads=4, num_key_value_heads=2,
            head_dim=16, rms_norm_eps=1e-6, vocab_size=100,
            max_position_embeddings=512, rope_theta=10000.0, tie_word_embeddings=False,
        )
        _mock_base_deps(monkeypatch, cfg)
        args = _make_mock_args(quant_target=["moe"])
        model = Qwen3Next(args)
        block = nn.Module()
        block.linear = nn.Linear(4, 4)
        bindings = list(model.iter_deploy_bindings(0, block))
        assert bindings == []

    def test_qwen3_next_float_model_delegates(self, monkeypatch):
        from amct_pytorch.common.models.llm.qwen.qwen3_next.qwen3_next import Qwen3Next
        from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextConfig
        monkeypatch.setattr(
            "amct_pytorch.common.models.llm.qwen.qwen3_next.qwen3_next.get_weight_mappings",
            lambda path: {},
        )
        cfg = Qwen3NextConfig(
            num_hidden_layers=2, hidden_size=64, intermediate_size=128,
            num_attention_heads=4, num_key_value_heads=2,
            head_dim=16, rms_norm_eps=1e-6, vocab_size=100,
            max_position_embeddings=512, rope_theta=10000.0, tie_word_embeddings=False,
        )
        _mock_base_deps(monkeypatch, cfg)
        model = Qwen3Next(_make_mock_args(quant_target=["moe"]))
        mock_result = MagicMock()
        monkeypatch.setattr(
            "amct_pytorch.common.models.llm.common.base.BaseModel.float_model",
            lambda self: mock_result,
        )
        assert model.float_model() is mock_result

    def test_qwen3_next_load_embed_state_dict_delegates(self, monkeypatch):
        from amct_pytorch.common.models.llm.qwen.qwen3_next.qwen3_next import Qwen3Next
        from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextConfig
        monkeypatch.setattr(
            "amct_pytorch.common.models.llm.qwen.qwen3_next.qwen3_next.get_weight_mappings",
            lambda path: {},
        )
        cfg = Qwen3NextConfig(
            num_hidden_layers=2, hidden_size=64, intermediate_size=128,
            num_attention_heads=4, num_key_value_heads=2,
            head_dim=16, rms_norm_eps=1e-6, vocab_size=100,
            max_position_embeddings=512, rope_theta=10000.0, tie_word_embeddings=False,
        )
        _mock_base_deps(monkeypatch, cfg)
        model = Qwen3Next(_make_mock_args(quant_target=["moe"]))
        monkeypatch.setattr(
            "amct_pytorch.common.models.llm.common.base.BaseModel.load_embed_state_dict",
            lambda self: None,
        )
        model.load_embed_state_dict()

    def test_qwen3_next_do_embedding_forward_delegates(self, monkeypatch):
        from amct_pytorch.common.models.llm.qwen.qwen3_next.qwen3_next import Qwen3Next
        from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextConfig
        monkeypatch.setattr(
            "amct_pytorch.common.models.llm.qwen.qwen3_next.qwen3_next.get_weight_mappings",
            lambda path: {},
        )
        cfg = Qwen3NextConfig(
            num_hidden_layers=2, hidden_size=64, intermediate_size=128,
            num_attention_heads=4, num_key_value_heads=2,
            head_dim=16, rms_norm_eps=1e-6, vocab_size=100,
            max_position_embeddings=512, rope_theta=10000.0, tie_word_embeddings=False,
        )
        _mock_base_deps(monkeypatch, cfg)
        model = Qwen3Next(_make_mock_args(quant_target=["moe"]))
        mock_result = [torch.randn(2, 4, 8)]
        monkeypatch.setattr(
            "amct_pytorch.common.models.llm.common.base.BaseModel.do_embedding_forward",
            lambda self, samples, dtype=torch.bfloat16, hook_name=None: mock_result,
        )
        result = model.do_embedding_forward([torch.randn(2, 4)])
        assert result is mock_result

    def test_qwen3_next_do_block_forward_delegates(self, monkeypatch):
        from amct_pytorch.common.models.llm.qwen.qwen3_next.qwen3_next import Qwen3Next
        from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextConfig
        monkeypatch.setattr(
            "amct_pytorch.common.models.llm.qwen.qwen3_next.qwen3_next.get_weight_mappings",
            lambda path: {},
        )
        cfg = Qwen3NextConfig(
            num_hidden_layers=2, hidden_size=64, intermediate_size=128,
            num_attention_heads=4, num_key_value_heads=2,
            head_dim=16, rms_norm_eps=1e-6, vocab_size=100,
            max_position_embeddings=512, rope_theta=10000.0, tie_word_embeddings=False,
        )
        _mock_base_deps(monkeypatch, cfg)
        model = Qwen3Next(_make_mock_args(quant_target=["moe"]))
        mock_result = [torch.randn(2, 4, 8)]
        monkeypatch.setattr(
            "amct_pytorch.common.models.llm.common.base.BaseModel.do_block_forward",
            lambda self, layer_idx, samples, hook_name=None, use_quant_block=False,
            enable_quant=False: mock_result,
        )
        result = model.do_block_forward(0, [torch.randn(2, 4, 8)])
        assert result is mock_result

    def test_qwen3_next_do_head_forward_delegates(self, monkeypatch):
        from amct_pytorch.common.models.llm.qwen.qwen3_next.qwen3_next import Qwen3Next
        from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextConfig
        monkeypatch.setattr(
            "amct_pytorch.common.models.llm.qwen.qwen3_next.qwen3_next.get_weight_mappings",
            lambda path: {},
        )
        cfg = Qwen3NextConfig(
            num_hidden_layers=2, hidden_size=64, intermediate_size=128,
            num_attention_heads=4, num_key_value_heads=2,
            head_dim=16, rms_norm_eps=1e-6, vocab_size=100,
            max_position_embeddings=512, rope_theta=10000.0, tie_word_embeddings=False,
        )
        _mock_base_deps(monkeypatch, cfg)
        model = Qwen3Next(_make_mock_args(quant_target=["moe"]))
        mock_result = [torch.randn(2, 4, 100)]
        monkeypatch.setattr(
            "amct_pytorch.common.models.llm.common.base.BaseModel.do_head_forward",
            lambda self, inps: mock_result,
        )
        result = model.do_head_forward([torch.randn(2, 4, 8)])
        assert result is mock_result

    def test_qwen3_next_build_quant_block_attn_linear(self, monkeypatch):
        from amct_pytorch.common.models.llm.qwen.qwen3_next.qwen3_next import Qwen3Next
        from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextConfig
        monkeypatch.setattr(
            "amct_pytorch.common.models.llm.qwen.qwen3_next.qwen3_next.get_weight_mappings",
            lambda path: {},
        )
        cfg = Qwen3NextConfig(
            num_hidden_layers=2, hidden_size=64, intermediate_size=128,
            num_attention_heads=4, num_key_value_heads=2,
            head_dim=16, rms_norm_eps=1e-6, vocab_size=100,
            max_position_embeddings=512, rope_theta=10000.0, tie_word_embeddings=False,
        )
        _mock_base_deps(monkeypatch, cfg)
        args = _make_mock_args(quant_target=["attn-linear"])
        model = Qwen3Next(args)
        dl = nn.Module()
        monkeypatch.setattr(model, "block", lambda idx: dl)
        monkeypatch.setattr(model, "apply_quant_attn", lambda dl_in: setattr(dl_in, "_a", True))
        result = model.build_quant_block(0)
        assert hasattr(result, "_a")

    def test_qwen3_next_load_unit_inputs_delegates(self, monkeypatch):
        from amct_pytorch.common.models.llm.qwen.qwen3_next.qwen3_next import Qwen3Next
        from amct_pytorch.common.models.llm.common.ptq_units import make_ptq_unit
        from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextConfig
        monkeypatch.setattr(
            "amct_pytorch.common.models.llm.qwen.qwen3_next.qwen3_next.get_weight_mappings",
            lambda path: {},
        )
        cfg = Qwen3NextConfig(
            num_hidden_layers=2, hidden_size=64, intermediate_size=128,
            num_attention_heads=4, num_key_value_heads=2,
            head_dim=16, rms_norm_eps=1e-6, vocab_size=100,
            max_position_embeddings=512, rope_theta=10000.0, tie_word_embeddings=False,
        )
        _mock_base_deps(monkeypatch, cfg)
        model = Qwen3Next(_make_mock_args(quant_target=["moe"]))
        unit = make_ptq_unit("moe", "mlp", layer_idx=0, module=nn.Linear(4, 4))
        fake_data = torch.randn(4, 4)
        monkeypatch.setattr(
            "amct_pytorch.common.models.llm.common.base.load_ptq_inps",
            lambda data_dir, kind, layer_idx: fake_data,
        )
        result = model.load_unit_inputs("/tmp/fake", unit)
        assert result is fake_data

    def test_qwen3_next_iter_ptq_units(self, monkeypatch):
        from amct_pytorch.common.models.llm.qwen.qwen3_next.qwen3_next import Qwen3Next
        from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextConfig
        monkeypatch.setattr(
            "amct_pytorch.common.models.llm.qwen.qwen3_next.qwen3_next.get_weight_mappings",
            lambda path: {},
        )
        cfg = Qwen3NextConfig(
            num_hidden_layers=2, hidden_size=64, intermediate_size=128,
            num_attention_heads=4, num_key_value_heads=2,
            head_dim=16, rms_norm_eps=1e-6, vocab_size=100,
            max_position_embeddings=512, rope_theta=10000.0, tie_word_embeddings=False,
        )
        _mock_base_deps(monkeypatch, cfg)
        args = _make_mock_args(quant_target=["moe"])
        model = Qwen3Next(args)
        block = nn.Module()
        block.mlp = nn.Module()
        block.mlp.experts = nn.ModuleList([nn.Linear(4, 4), nn.Linear(4, 4)])
        units = list(model.iter_ptq_units(0, block))
        assert len(units) == 2
        assert all(u.kind == "moe" for u in units)


def _mock_base_deps_wrapper(monkeypatch, cfg):
    """Mock BaseModel HF dependencies with a wrapper config that has text_config."""
    fake_tokenizer = MagicMock()
    monkeypatch.setattr(
        "amct_pytorch.common.models.llm.common.base.AutoConfig",
        type("FakeAC", (), {"from_pretrained": staticmethod(lambda *a, **kw: cfg)})(),
    )
    monkeypatch.setattr(
        "amct_pytorch.common.models.llm.common.base.AutoTokenizer",
        type("FakeAT", (), {"from_pretrained": staticmethod(lambda *a, **kw: fake_tokenizer)})(),
    )
    monkeypatch.setattr(
        "amct_pytorch.common.models.llm.common.base.init_empty_weights",
        lambda: MagicMock(__enter__=lambda s: None, __exit__=lambda *a: None),
    )
    from transformers import AutoModelForCausalLM
    with init_empty_weights():
        empty_model = AutoModelForCausalLM.from_config(cfg.text_config, trust_remote_code=True, torch_dtype=torch.bfloat16)
    monkeypatch.setattr(
        "amct_pytorch.common.models.llm.common.base.AutoModelForCausalLM",
        type("FakeAMFCLM", (), {
            "from_pretrained": staticmethod(lambda *a, **kw: empty_model),
            "from_config": staticmethod(lambda *a, **kw: empty_model),
        })(),
    )
    monkeypatch.setattr(
        "amct_pytorch.common.models.llm.common.base.get_weight_mappings",
        lambda path: {},
    )


class TestQwen3_5Mocked:
    def test_qwen3_5_init(self, monkeypatch):
        from amct_pytorch.common.models.llm.qwen.qwen3_5.qwen3_5 import Qwen3_5
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Config, Qwen3_5DecoderLayer
        cfg = Qwen3_5Config(text_config={
            "num_hidden_layers": 2, "hidden_size": 64, "intermediate_size": 128,
            "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16,
            "rms_norm_eps": 1e-6, "vocab_size": 100, "max_position_embeddings": 512,
            "rope_theta": 10000.0, "tie_word_embeddings": False,
        })
        _mock_base_deps_wrapper(monkeypatch, cfg)
        args = _make_mock_args(quant_target=["mlp"])
        model = Qwen3_5(args)
        assert model.num_layers == 2
        assert model.cls is Qwen3_5DecoderLayer
        assert model.quant_target == ["mlp"]

    def test_qwen3_5_parse_quant_mode_rejects_moe(self, monkeypatch):
        from amct_pytorch.common.models.llm.qwen.qwen3_5.qwen3_5 import Qwen3_5
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Config
        cfg = Qwen3_5Config(text_config={
            "num_hidden_layers": 2, "hidden_size": 64, "intermediate_size": 128,
            "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16,
            "rms_norm_eps": 1e-6, "vocab_size": 100, "max_position_embeddings": 512,
            "rope_theta": 10000.0, "tie_word_embeddings": False,
        })
        _mock_base_deps_wrapper(monkeypatch, cfg)
        args = _make_mock_args(quant_target=["moe"])
        with pytest.raises(ValueError, match="does not support quant_target='moe'"):
            Qwen3_5(args)

    def test_qwen3_5_get_layer_weight_prefix(self, monkeypatch):
        from amct_pytorch.common.models.llm.qwen.qwen3_5.qwen3_5 import Qwen3_5
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Config
        cfg = Qwen3_5Config(text_config={
            "num_hidden_layers": 2, "hidden_size": 64, "intermediate_size": 128,
            "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16,
            "rms_norm_eps": 1e-6, "vocab_size": 100, "max_position_embeddings": 512,
            "rope_theta": 10000.0, "tie_word_embeddings": False,
        })
        _mock_base_deps_wrapper(monkeypatch, cfg)
        model = Qwen3_5(_make_mock_args(quant_target=["mlp"]))
        assert model.get_layer_weight_prefix(0) == "model.language_model.layers.0."

    def test_qwen3_5_embed_base_prefix(self, monkeypatch):
        from amct_pytorch.common.models.llm.qwen.qwen3_5.qwen3_5 import Qwen3_5
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Config
        cfg = Qwen3_5Config(text_config={
            "num_hidden_layers": 2, "hidden_size": 64, "intermediate_size": 128,
            "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16,
            "rms_norm_eps": 1e-6, "vocab_size": 100, "max_position_embeddings": 512,
            "rope_theta": 10000.0, "tie_word_embeddings": False,
        })
        _mock_base_deps_wrapper(monkeypatch, cfg)
        model = Qwen3_5(_make_mock_args(quant_target=["mlp"]))
        assert model._embed_base_prefix() == "model.language_model."

    def test_qwen3_5_get_block_attention_mask(self, monkeypatch):
        from amct_pytorch.common.models.llm.qwen.qwen3_5.qwen3_5 import Qwen3_5
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Config
        cfg = Qwen3_5Config(text_config={
            "num_hidden_layers": 2, "hidden_size": 64, "intermediate_size": 128,
            "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16,
            "rms_norm_eps": 1e-6, "vocab_size": 100, "max_position_embeddings": 512,
            "rope_theta": 10000.0, "tie_word_embeddings": False,
        })
        _mock_base_deps_wrapper(monkeypatch, cfg)
        model = Qwen3_5(_make_mock_args(quant_target=["mlp"]))
        assert model.get_block_attention_mask(nn.Module()) is None

    def test_qwen3_5_build_quant_block_mlp(self, monkeypatch):
        from amct_pytorch.common.models.llm.qwen.qwen3_5.qwen3_5 import Qwen3_5
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Config
        cfg = Qwen3_5Config(text_config={
            "num_hidden_layers": 2, "hidden_size": 64, "intermediate_size": 128,
            "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16,
            "rms_norm_eps": 1e-6, "vocab_size": 100, "max_position_embeddings": 512,
            "rope_theta": 10000.0, "tie_word_embeddings": False,
        })
        _mock_base_deps_wrapper(monkeypatch, cfg)
        model = Qwen3_5(_make_mock_args(quant_target=["mlp"]))
        mock_block = nn.Linear(4, 4)
        monkeypatch.setattr(model, "block", lambda idx: mock_block)
        monkeypatch.setattr(model, "apply_quant_moe_mlp", lambda dl: setattr(dl, "_q", True))
        result = model.build_quant_block(0)
        assert hasattr(result, "_q")

    def test_qwen3_5_build_quant_block_attn_linear(self, monkeypatch):
        from amct_pytorch.common.models.llm.qwen.qwen3_5.qwen3_5 import Qwen3_5
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Config
        cfg = Qwen3_5Config(text_config={
            "num_hidden_layers": 2, "hidden_size": 64, "intermediate_size": 128,
            "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16,
            "rms_norm_eps": 1e-6, "vocab_size": 100, "max_position_embeddings": 512,
            "rope_theta": 10000.0, "tie_word_embeddings": False,
        })
        _mock_base_deps_wrapper(monkeypatch, cfg)
        model = Qwen3_5(_make_mock_args(quant_target=["attn-linear"]))
        mock_block = nn.Linear(4, 4)
        monkeypatch.setattr(model, "block", lambda idx: mock_block)
        monkeypatch.setattr(model, "apply_quant_attn", lambda dl: setattr(dl, "_a", True))
        result = model.build_quant_block(0)
        assert hasattr(result, "_a")

    def test_qwen3_5_apply_quant_attn_linear_attention(self, monkeypatch):
        from amct_pytorch.common.models.llm.qwen.qwen3_5.qwen3_5 import Qwen3_5
        from amct_pytorch.common.models.llm.qwen.qwen3_5.quant_module import QuantQwen35LinearAttn
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Config
        cfg = Qwen3_5Config(text_config={
            "num_hidden_layers": 2, "hidden_size": 64, "intermediate_size": 128,
            "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16,
            "rms_norm_eps": 1e-6, "vocab_size": 100, "max_position_embeddings": 512,
            "rope_theta": 10000.0, "tie_word_embeddings": False,
        })
        _mock_base_deps_wrapper(monkeypatch, cfg)
        model = Qwen3_5(_make_mock_args(quant_target=["attn-linear"]))
        dl = nn.Module()
        dl.layer_type = "linear_attention"
        dl.linear_attn = nn.Module()
        captured = {}
        def _mock(args, dl_in, attn_cls):
            captured["attn_cls"] = attn_cls
        monkeypatch.setattr(
            "amct_pytorch.common.models.llm.qwen.qwen3_5.qwen3_5.apply_quant_to_attn", _mock
        )
        model.apply_quant_attn(dl)
        assert captured["attn_cls"] is QuantQwen35LinearAttn

    def test_qwen3_5_apply_quant_attn_default(self, monkeypatch):
        from amct_pytorch.common.models.llm.qwen.qwen3_5.qwen3_5 import Qwen3_5
        from amct_pytorch.common.models.llm.qwen.qwen3_5.quant_module import QuantQwen35Attn
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Config
        cfg = Qwen3_5Config(text_config={
            "num_hidden_layers": 2, "hidden_size": 64, "intermediate_size": 128,
            "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16,
            "rms_norm_eps": 1e-6, "vocab_size": 100, "max_position_embeddings": 512,
            "rope_theta": 10000.0, "tie_word_embeddings": False,
        })
        _mock_base_deps_wrapper(monkeypatch, cfg)
        model = Qwen3_5(_make_mock_args(quant_target=["attn-linear"]))
        dl = nn.Module()
        dl.layer_type = "default_attention"
        captured = {}
        def _mock(args, dl_in, attn_cls):
            captured["attn_cls"] = attn_cls
        monkeypatch.setattr(
            "amct_pytorch.common.models.llm.qwen.qwen3_5.qwen3_5.apply_quant_to_attn", _mock
        )
        model.apply_quant_attn(dl)
        assert captured["attn_cls"] is QuantQwen35Attn

    def test_qwen3_5_apply_quant_moe_mlp(self, monkeypatch):
        from amct_pytorch.common.models.llm.qwen.qwen3_5.qwen3_5 import Qwen3_5
        from amct_pytorch.common.models.llm.qwen.qwen3_5.quant_module import QuantQwen35MLP
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Config
        cfg = Qwen3_5Config(text_config={
            "num_hidden_layers": 2, "hidden_size": 64, "intermediate_size": 128,
            "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16,
            "rms_norm_eps": 1e-6, "vocab_size": 100, "max_position_embeddings": 512,
            "rope_theta": 10000.0, "tie_word_embeddings": False,
        })
        _mock_base_deps_wrapper(monkeypatch, cfg)
        model = Qwen3_5(_make_mock_args(quant_target=["mlp"]))
        dl = nn.Linear(4, 4)
        captured = {}
        def _mock(args, dl_in, **kwargs):
            captured["cls"] = kwargs.get("cls")
        monkeypatch.setattr(
            "amct_pytorch.common.models.llm.qwen.qwen3_5.qwen3_5.apply_quant_to_moe_mlp", _mock
        )
        model.apply_quant_moe_mlp(dl)
        assert captured["cls"] is QuantQwen35MLP

    def test_qwen3_5_iter_ptq_units_delegates(self, monkeypatch):
        from amct_pytorch.common.models.llm.qwen.qwen3_5.qwen3_5 import Qwen3_5
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Config
        cfg = Qwen3_5Config(text_config={
            "num_hidden_layers": 2, "hidden_size": 64, "intermediate_size": 128,
            "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16,
            "rms_norm_eps": 1e-6, "vocab_size": 100, "max_position_embeddings": 512,
            "rope_theta": 10000.0, "tie_word_embeddings": False,
        })
        _mock_base_deps_wrapper(monkeypatch, cfg)
        model = Qwen3_5(_make_mock_args(quant_target=["mlp"]))
        block = _make_block_with_attn_and_mlp()
        units = list(model.iter_ptq_units(0, block))
        assert len(units) == 1
        assert units[0].kind == "mlp"

    def test_qwen3_5_iter_deploy_bindings(self, monkeypatch):
        from amct_pytorch.common.models.llm.qwen.qwen3_5.qwen3_5 import Qwen3_5
        from amct_pytorch.quantization.modules.quant_linear import QuantLinear
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Config
        cfg = Qwen3_5Config(text_config={
            "num_hidden_layers": 2, "hidden_size": 64, "intermediate_size": 128,
            "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16,
            "rms_norm_eps": 1e-6, "vocab_size": 100, "max_position_embeddings": 512,
            "rope_theta": 10000.0, "tie_word_embeddings": False,
        })
        _mock_base_deps_wrapper(monkeypatch, cfg)
        model = Qwen3_5(_make_mock_args(quant_target=["mlp"]))
        ql = QuantLinear.__new__(QuantLinear)
        ql._parameters = {}
        ql._modules = {}
        ql._buffers = {}
        block = nn.Module()
        block.linear = ql
        bindings = list(model.iter_deploy_bindings(0, block))
        assert len(bindings) == 1

    def test_qwen3_5_float_model_delegates(self, monkeypatch):
        from amct_pytorch.common.models.llm.qwen.qwen3_5.qwen3_5 import Qwen3_5
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Config
        cfg = Qwen3_5Config(text_config={
            "num_hidden_layers": 2, "hidden_size": 64, "intermediate_size": 128,
            "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16,
            "rms_norm_eps": 1e-6, "vocab_size": 100, "max_position_embeddings": 512,
            "rope_theta": 10000.0, "tie_word_embeddings": False,
        })
        _mock_base_deps_wrapper(monkeypatch, cfg)
        model = Qwen3_5(_make_mock_args(quant_target=["mlp"]))
        mock_result = MagicMock()
        monkeypatch.setattr(
            "amct_pytorch.common.models.llm.common.base.BaseModel.float_model",
            lambda self: mock_result,
        )
        assert model.float_model() is mock_result

    def test_qwen3_5_load_layer_weight_delegates(self, monkeypatch):
        from amct_pytorch.common.models.llm.qwen.qwen3_5.qwen3_5 import Qwen3_5
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Config
        cfg = Qwen3_5Config(text_config={
            "num_hidden_layers": 2, "hidden_size": 64, "intermediate_size": 128,
            "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16,
            "rms_norm_eps": 1e-6, "vocab_size": 100, "max_position_embeddings": 512,
            "rope_theta": 10000.0, "tie_word_embeddings": False,
        })
        _mock_base_deps_wrapper(monkeypatch, cfg)
        model = Qwen3_5(_make_mock_args(quant_target=["mlp"]))
        mock_result = {"some.weight": torch.randn(4, 4)}
        monkeypatch.setattr(
            "amct_pytorch.common.models.llm.common.base.BaseModel.load_layer_weight",
            lambda self, prefix: mock_result,
        )
        assert model.load_layer_weight("model.language_model.layers.0.") is mock_result

    def test_qwen3_5_load_embed_state_dict_delegates(self, monkeypatch):
        from amct_pytorch.common.models.llm.qwen.qwen3_5.qwen3_5 import Qwen3_5
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Config
        cfg = Qwen3_5Config(text_config={
            "num_hidden_layers": 2, "hidden_size": 64, "intermediate_size": 128,
            "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16,
            "rms_norm_eps": 1e-6, "vocab_size": 100, "max_position_embeddings": 512,
            "rope_theta": 10000.0, "tie_word_embeddings": False,
        })
        _mock_base_deps_wrapper(monkeypatch, cfg)
        model = Qwen3_5(_make_mock_args(quant_target=["mlp"]))
        monkeypatch.setattr(
            "amct_pytorch.common.models.llm.common.base.BaseModel.load_embed_state_dict",
            lambda self: None,
        )
        model.load_embed_state_dict()

    def test_qwen3_5_block_delegates(self, monkeypatch):
        from amct_pytorch.common.models.llm.qwen.qwen3_5.qwen3_5 import Qwen3_5
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Config
        cfg = Qwen3_5Config(text_config={
            "num_hidden_layers": 2, "hidden_size": 64, "intermediate_size": 128,
            "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16,
            "rms_norm_eps": 1e-6, "vocab_size": 100, "max_position_embeddings": 512,
            "rope_theta": 10000.0, "tie_word_embeddings": False,
        })
        _mock_base_deps_wrapper(monkeypatch, cfg)
        model = Qwen3_5(_make_mock_args(quant_target=["mlp"]))
        mock_result = MagicMock()
        monkeypatch.setattr(
            "amct_pytorch.common.models.llm.common.base.BaseModel.block",
            lambda self, idx: mock_result,
        )
        assert model.block(0) is mock_result

    def test_qwen3_5_do_embedding_forward_delegates(self, monkeypatch):
        from amct_pytorch.common.models.llm.qwen.qwen3_5.qwen3_5 import Qwen3_5
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Config
        cfg = Qwen3_5Config(text_config={
            "num_hidden_layers": 2, "hidden_size": 64, "intermediate_size": 128,
            "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16,
            "rms_norm_eps": 1e-6, "vocab_size": 100, "max_position_embeddings": 512,
            "rope_theta": 10000.0, "tie_word_embeddings": False,
        })
        _mock_base_deps_wrapper(monkeypatch, cfg)
        model = Qwen3_5(_make_mock_args(quant_target=["mlp"]))
        mock_result = [torch.randn(2, 4, 8)]
        monkeypatch.setattr(
            "amct_pytorch.common.models.llm.common.base.BaseModel.do_embedding_forward",
            lambda self, samples, dtype=torch.bfloat16, hook_name=None: mock_result,
        )
        result = model.do_embedding_forward([torch.randn(2, 4)])
        assert result is mock_result

    def test_qwen3_5_do_block_forward_delegates(self, monkeypatch):
        from amct_pytorch.common.models.llm.qwen.qwen3_5.qwen3_5 import Qwen3_5
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Config
        cfg = Qwen3_5Config(text_config={
            "num_hidden_layers": 2, "hidden_size": 64, "intermediate_size": 128,
            "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16,
            "rms_norm_eps": 1e-6, "vocab_size": 100, "max_position_embeddings": 512,
            "rope_theta": 10000.0, "tie_word_embeddings": False,
        })
        _mock_base_deps_wrapper(monkeypatch, cfg)
        model = Qwen3_5(_make_mock_args(quant_target=["mlp"]))
        mock_result = [torch.randn(2, 4, 8)]
        monkeypatch.setattr(
            "amct_pytorch.common.models.llm.common.base.BaseModel.do_block_forward",
            lambda self, layer_idx, samples, hook_name=None, use_quant_block=False,
            enable_quant=False: mock_result,
        )
        result = model.do_block_forward(0, [torch.randn(2, 4, 8)])
        assert result is mock_result

    def test_qwen3_5_do_head_forward_delegates(self, monkeypatch):
        from amct_pytorch.common.models.llm.qwen.qwen3_5.qwen3_5 import Qwen3_5
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Config
        cfg = Qwen3_5Config(text_config={
            "num_hidden_layers": 2, "hidden_size": 64, "intermediate_size": 128,
            "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16,
            "rms_norm_eps": 1e-6, "vocab_size": 100, "max_position_embeddings": 512,
            "rope_theta": 10000.0, "tie_word_embeddings": False,
        })
        _mock_base_deps_wrapper(monkeypatch, cfg)
        model = Qwen3_5(_make_mock_args(quant_target=["mlp"]))
        mock_result = [torch.randn(2, 4, 100)]
        monkeypatch.setattr(
            "amct_pytorch.common.models.llm.common.base.BaseModel.do_head_forward",
            lambda self, inps: mock_result,
        )
        result = model.do_head_forward([torch.randn(2, 4, 8)])
        assert result is mock_result

    def test_qwen3_5_load_unit_inputs_delegates(self, monkeypatch):
        from amct_pytorch.common.models.llm.qwen.qwen3_5.qwen3_5 import Qwen3_5
        from amct_pytorch.common.models.llm.common.ptq_units import make_ptq_unit
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Config
        cfg = Qwen3_5Config(text_config={
            "num_hidden_layers": 2, "hidden_size": 64, "intermediate_size": 128,
            "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16,
            "rms_norm_eps": 1e-6, "vocab_size": 100, "max_position_embeddings": 512,
            "rope_theta": 10000.0, "tie_word_embeddings": False,
        })
        _mock_base_deps_wrapper(monkeypatch, cfg)
        model = Qwen3_5(_make_mock_args(quant_target=["mlp"]))
        unit = make_ptq_unit("mlp", "mlp", layer_idx=0, module=nn.Linear(4, 4))
        fake_data = torch.randn(4, 4)
        monkeypatch.setattr(
            "amct_pytorch.common.models.llm.common.base.load_ptq_inps",
            lambda data_dir, kind, layer_idx: fake_data,
        )
        result = model.load_unit_inputs("/tmp/fake", unit)
        assert result is fake_data


class TestQwen3_5MoeMocked:
    def test_qwen3_5_moe_init(self, monkeypatch):
        from amct_pytorch.common.models.llm.qwen.qwen3_5.qwen3_5_moe import Qwen3_5Moe
        from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeConfig, Qwen3_5MoeDecoderLayer
        monkeypatch.setattr(
            "amct_pytorch.common.models.llm.qwen.qwen3_5.qwen3_5_moe.get_weight_mappings",
            lambda path: {},
        )
        cfg = Qwen3_5MoeConfig(text_config={
            "num_hidden_layers": 2, "hidden_size": 64, "intermediate_size": 128,
            "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16,
            "rms_norm_eps": 1e-6, "vocab_size": 100, "max_position_embeddings": 512,
            "rope_theta": 10000.0, "tie_word_embeddings": False,
            "num_experts": 4, "moe_intermediate_size": 64,
            "n_routed_experts": 4, "n_shared_experts": 0,
        })
        _mock_base_deps_wrapper(monkeypatch, cfg)
        args = _make_mock_args(quant_target=["moe"])
        model = Qwen3_5Moe(args)
        assert model.num_layers == 2
        assert model.cls is Qwen3_5MoeDecoderLayer

    def test_qwen3_5_moe_parse_quant_mode_rejects_mlp(self, monkeypatch):
        from amct_pytorch.common.models.llm.qwen.qwen3_5.qwen3_5_moe import Qwen3_5Moe
        from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeConfig
        monkeypatch.setattr(
            "amct_pytorch.common.models.llm.qwen.qwen3_5.qwen3_5_moe.get_weight_mappings",
            lambda path: {},
        )
        cfg = Qwen3_5MoeConfig(text_config={
            "num_hidden_layers": 2, "hidden_size": 64, "intermediate_size": 128,
            "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16,
            "rms_norm_eps": 1e-6, "vocab_size": 100, "max_position_embeddings": 512,
            "rope_theta": 10000.0, "tie_word_embeddings": False,
            "num_experts": 4, "moe_intermediate_size": 64,
            "n_routed_experts": 4, "n_shared_experts": 0,
        })
        _mock_base_deps_wrapper(monkeypatch, cfg)
        args = _make_mock_args(quant_target=["mlp"], moe_mlp_param_dir="")
        with pytest.raises(ValueError, match="does not support quant_target='mlp'"):
            Qwen3_5Moe(args)

    def test_qwen3_5_moe_get_layer_weight_prefix(self, monkeypatch):
        from amct_pytorch.common.models.llm.qwen.qwen3_5.qwen3_5_moe import Qwen3_5Moe
        from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeConfig
        monkeypatch.setattr(
            "amct_pytorch.common.models.llm.qwen.qwen3_5.qwen3_5_moe.get_weight_mappings",
            lambda path: {},
        )
        cfg = Qwen3_5MoeConfig(text_config={
            "num_hidden_layers": 2, "hidden_size": 64, "intermediate_size": 128,
            "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16,
            "rms_norm_eps": 1e-6, "vocab_size": 100, "max_position_embeddings": 512,
            "rope_theta": 10000.0, "tie_word_embeddings": False,
            "num_experts": 4, "moe_intermediate_size": 64,
            "n_routed_experts": 4, "n_shared_experts": 0,
        })
        _mock_base_deps_wrapper(monkeypatch, cfg)
        args = _make_mock_args(quant_target=["moe"])
        model = Qwen3_5Moe(args)
        assert model.get_layer_weight_prefix(0) == "model.language_model.layers.0."

    def test_qwen3_5_moe_num_experts(self, monkeypatch):
        from amct_pytorch.common.models.llm.qwen.qwen3_5.qwen3_5_moe import Qwen3_5Moe
        from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeConfig
        monkeypatch.setattr(
            "amct_pytorch.common.models.llm.qwen.qwen3_5.qwen3_5_moe.get_weight_mappings",
            lambda path: {},
        )
        cfg = Qwen3_5MoeConfig(text_config={
            "num_hidden_layers": 2, "hidden_size": 64, "intermediate_size": 128,
            "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16,
            "rms_norm_eps": 1e-6, "vocab_size": 100, "max_position_embeddings": 512,
            "rope_theta": 10000.0, "tie_word_embeddings": False,
            "num_experts": 8, "moe_intermediate_size": 64,
            "n_routed_experts": 8, "n_shared_experts": 0,
        })
        _mock_base_deps_wrapper(monkeypatch, cfg)
        args = _make_mock_args(quant_target=["moe"])
        model = Qwen3_5Moe(args)
        assert model.config.num_experts == 8

    def test_qwen3_5_moe_build_quant_block_moe(self, monkeypatch):
        from amct_pytorch.common.models.llm.qwen.qwen3_5.qwen3_5_moe import Qwen3_5Moe
        from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeConfig
        monkeypatch.setattr(
            "amct_pytorch.common.models.llm.qwen.qwen3_5.qwen3_5_moe.get_weight_mappings",
            lambda path: {},
        )
        cfg = Qwen3_5MoeConfig(text_config={
            "num_hidden_layers": 2, "hidden_size": 64, "intermediate_size": 128,
            "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16,
            "rms_norm_eps": 1e-6, "vocab_size": 100, "max_position_embeddings": 512,
            "rope_theta": 10000.0, "tie_word_embeddings": False,
            "num_experts": 4, "moe_intermediate_size": 64,
            "n_routed_experts": 4, "n_shared_experts": 0,
        })
        _mock_base_deps_wrapper(monkeypatch, cfg)
        args = _make_mock_args(quant_target=["moe"])
        model = Qwen3_5Moe(args)
        dl = nn.Module()
        moe = nn.Module()
        moe.experts = type("PE", (), {"gate_up_proj": None, "down_proj": None, "num_experts": 4})()
        dl.mlp = moe
        monkeypatch.setattr(model, "block", lambda idx: dl)
        monkeypatch.setattr(
            "amct_pytorch.common.models.llm.qwen.qwen3_5.qwen3_5_moe.find_moe_module",
            lambda dl_in: dl_in.mlp,
        )
        monkeypatch.setattr(
            "amct_pytorch.common.models.llm.qwen.qwen3_5.qwen3_5_moe.QuantGatedExperts",
            lambda args_in, experts, **kwargs: experts,
        )
        result = model.build_quant_block(0)
        assert result is dl

    def test_qwen3_5_moe_build_quant_block_moe_with_shared_expert(self, monkeypatch):
        from amct_pytorch.common.models.llm.qwen.qwen3_5.qwen3_5_moe import Qwen3_5Moe
        from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeConfig
        monkeypatch.setattr(
            "amct_pytorch.common.models.llm.qwen.qwen3_5.qwen3_5_moe.get_weight_mappings",
            lambda path: {},
        )
        cfg = Qwen3_5MoeConfig(text_config={
            "num_hidden_layers": 2, "hidden_size": 64, "intermediate_size": 128,
            "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16,
            "rms_norm_eps": 1e-6, "vocab_size": 100, "max_position_embeddings": 512,
            "rope_theta": 10000.0, "tie_word_embeddings": False,
            "num_experts": 4, "moe_intermediate_size": 64,
            "n_routed_experts": 4, "n_shared_experts": 1,
        })
        _mock_base_deps_wrapper(monkeypatch, cfg)
        args = _make_mock_args(quant_target=["moe"])
        model = Qwen3_5Moe(args)
        dl = nn.Module()
        moe = nn.Module()
        moe.experts = type("PE", (), {"gate_up_proj": None, "down_proj": None, "num_experts": 4})()
        shared = nn.Linear(4, 4)
        moe.shared_expert = shared
        dl.mlp = moe
        monkeypatch.setattr(model, "block", lambda idx: dl)
        monkeypatch.setattr(
            "amct_pytorch.common.models.llm.qwen.qwen3_5.qwen3_5_moe.find_moe_module",
            lambda dl_in: dl_in.mlp,
        )
        monkeypatch.setattr(
            "amct_pytorch.common.models.llm.qwen.qwen3_5.qwen3_5_moe.QuantGatedExperts",
            lambda args_in, experts, **kwargs: experts,
        )
        captured = {}
        class _MockQwen35MLP:
            def __new__(cls, args_in, module, **kwargs):
                captured["called"] = True
                captured["module"] = module
                return module
        monkeypatch.setattr(
            "amct_pytorch.common.models.llm.qwen.qwen3_5.qwen3_5_moe.QuantQwen35MLP",
            _MockQwen35MLP,
        )
        result = model.build_quant_block(0)
        assert captured.get("called") is True

    def test_qwen3_5_moe_build_quant_block_moe_no_experts(self, monkeypatch):
        from amct_pytorch.common.models.llm.qwen.qwen3_5.qwen3_5_moe import Qwen3_5Moe
        from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeConfig
        monkeypatch.setattr(
            "amct_pytorch.common.models.llm.qwen.qwen3_5.qwen3_5_moe.get_weight_mappings",
            lambda path: {},
        )
        cfg = Qwen3_5MoeConfig(text_config={
            "num_hidden_layers": 2, "hidden_size": 64, "intermediate_size": 128,
            "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16,
            "rms_norm_eps": 1e-6, "vocab_size": 100, "max_position_embeddings": 512,
            "rope_theta": 10000.0, "tie_word_embeddings": False,
            "num_experts": 4, "moe_intermediate_size": 64,
            "n_routed_experts": 4, "n_shared_experts": 0,
        })
        _mock_base_deps_wrapper(monkeypatch, cfg)
        args = _make_mock_args(quant_target=["moe"])
        model = Qwen3_5Moe(args)
        dl = nn.Linear(4, 4)
        monkeypatch.setattr(model, "block", lambda idx: dl)
        monkeypatch.setattr(
            "amct_pytorch.common.models.llm.qwen.qwen3_5.qwen3_5_moe.find_moe_module",
            lambda dl_in: None,
        )
        result = model.build_quant_block(0)
        assert result is dl

    def test_qwen3_5_moe_build_quant_block_attn_and_moe(self, monkeypatch):
        from amct_pytorch.common.models.llm.qwen.qwen3_5.qwen3_5_moe import Qwen3_5Moe
        from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeConfig
        monkeypatch.setattr(
            "amct_pytorch.common.models.llm.qwen.qwen3_5.qwen3_5_moe.get_weight_mappings",
            lambda path: {},
        )
        cfg = Qwen3_5MoeConfig(text_config={
            "num_hidden_layers": 2, "hidden_size": 64, "intermediate_size": 128,
            "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16,
            "rms_norm_eps": 1e-6, "vocab_size": 100, "max_position_embeddings": 512,
            "rope_theta": 10000.0, "tie_word_embeddings": False,
            "num_experts": 4, "moe_intermediate_size": 64,
            "n_routed_experts": 4, "n_shared_experts": 0,
        })
        _mock_base_deps_wrapper(monkeypatch, cfg)
        args = _make_mock_args(quant_target=["attn-linear", "moe"])
        model = Qwen3_5Moe(args)
        dl = nn.Module()
        moe = nn.Module()
        moe.experts = type("PE", (), {"gate_up_proj": None, "down_proj": None, "num_experts": 4})()
        dl.mlp = moe
        monkeypatch.setattr(model, "block", lambda idx: dl)
        monkeypatch.setattr(model, "apply_quant_attn", lambda dl_in: setattr(dl_in, "_a", True))
        monkeypatch.setattr(
            "amct_pytorch.common.models.llm.qwen.qwen3_5.qwen3_5_moe.find_moe_module",
            lambda dl_in: dl_in.mlp,
        )
        monkeypatch.setattr(
            "amct_pytorch.common.models.llm.qwen.qwen3_5.qwen3_5_moe.QuantGatedExperts",
            lambda args_in, experts, **kwargs: experts,
        )
        result = model.build_quant_block(0)
        assert hasattr(result, "_a")

    def test_qwen3_5_moe_iter_deploy_bindings_expert_name(self, monkeypatch):
        from amct_pytorch.common.models.llm.qwen.qwen3_5.qwen3_5_moe import Qwen3_5Moe
        from amct_pytorch.quantization.modules.quant_linear import QuantLinear
        from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeConfig
        monkeypatch.setattr(
            "amct_pytorch.common.models.llm.qwen.qwen3_5.qwen3_5_moe.get_weight_mappings",
            lambda path: {},
        )
        cfg = Qwen3_5MoeConfig(text_config={
            "num_hidden_layers": 2, "hidden_size": 64, "intermediate_size": 128,
            "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16,
            "rms_norm_eps": 1e-6, "vocab_size": 100, "max_position_embeddings": 512,
            "rope_theta": 10000.0, "tie_word_embeddings": False,
            "num_experts": 4, "moe_intermediate_size": 64,
            "n_routed_experts": 4, "n_shared_experts": 0,
        })
        _mock_base_deps_wrapper(monkeypatch, cfg)
        args = _make_mock_args(quant_target=["moe"])
        model = Qwen3_5Moe(args)
        ql = QuantLinear.__new__(QuantLinear)
        ql._parameters = {}
        ql._modules = {}
        ql._buffers = {}
        block = nn.Module()
        block.mlp = nn.Module()
        block.mlp.experts = nn.Module()
        expert0 = nn.Module()
        expert1 = nn.Module()
        expert1.gate_proj = ql
        block.mlp.experts.expert_modules = nn.ModuleList([expert0, expert1])
        bindings = list(model.iter_deploy_bindings(0, block))
        assert len(bindings) == 1
        assert bindings[0][0] == "model.language_model.layers.0.mlp.experts.expert_modules.1.gate_proj.weight"

    def test_qwen3_5_moe_iter_ptq_units(self, monkeypatch):
        from amct_pytorch.common.models.llm.qwen.qwen3_5.qwen3_5_moe import Qwen3_5Moe
        from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeConfig
        monkeypatch.setattr(
            "amct_pytorch.common.models.llm.qwen.qwen3_5.qwen3_5_moe.get_weight_mappings",
            lambda path: {},
        )
        cfg = Qwen3_5MoeConfig(text_config={
            "num_hidden_layers": 2, "hidden_size": 64, "intermediate_size": 128,
            "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16,
            "rms_norm_eps": 1e-6, "vocab_size": 100, "max_position_embeddings": 512,
            "rope_theta": 10000.0, "tie_word_embeddings": False,
            "num_experts": 4, "moe_intermediate_size": 64,
            "n_routed_experts": 4, "n_shared_experts": 0,
        })
        _mock_base_deps_wrapper(monkeypatch, cfg)
        args = _make_mock_args(quant_target=["moe"])
        model = Qwen3_5Moe(args)
        block = nn.Module()
        block.mlp = nn.Module()
        block.mlp.experts = nn.ModuleList([nn.Linear(4, 4), nn.Linear(4, 4)])
        units = list(model.iter_ptq_units(0, block))
        assert len(units) == 2
        assert all(u.kind == "moe" for u in units)

    def test_qwen3_5_moe_float_model_delegates(self, monkeypatch):
        from amct_pytorch.common.models.llm.qwen.qwen3_5.qwen3_5_moe import Qwen3_5Moe
        from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeConfig
        monkeypatch.setattr(
            "amct_pytorch.common.models.llm.qwen.qwen3_5.qwen3_5_moe.get_weight_mappings",
            lambda path: {},
        )
        cfg = Qwen3_5MoeConfig(text_config={
            "num_hidden_layers": 2, "hidden_size": 64, "intermediate_size": 128,
            "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16,
            "rms_norm_eps": 1e-6, "vocab_size": 100, "max_position_embeddings": 512,
            "rope_theta": 10000.0, "tie_word_embeddings": False,
            "num_experts": 4, "moe_intermediate_size": 64,
            "n_routed_experts": 4, "n_shared_experts": 0,
        })
        _mock_base_deps_wrapper(monkeypatch, cfg)
        model = Qwen3_5Moe(_make_mock_args(quant_target=["moe"]))
        mock_result = MagicMock()
        monkeypatch.setattr(
            "amct_pytorch.common.models.llm.common.base.BaseModel.float_model",
            lambda self: mock_result,
        )
        assert model.float_model() is mock_result

    def test_qwen3_5_moe_load_layer_weight_delegates(self, monkeypatch):
        from amct_pytorch.common.models.llm.qwen.qwen3_5.qwen3_5_moe import Qwen3_5Moe
        from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeConfig
        monkeypatch.setattr(
            "amct_pytorch.common.models.llm.qwen.qwen3_5.qwen3_5_moe.get_weight_mappings",
            lambda path: {},
        )
        cfg = Qwen3_5MoeConfig(text_config={
            "num_hidden_layers": 2, "hidden_size": 64, "intermediate_size": 128,
            "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16,
            "rms_norm_eps": 1e-6, "vocab_size": 100, "max_position_embeddings": 512,
            "rope_theta": 10000.0, "tie_word_embeddings": False,
            "num_experts": 4, "moe_intermediate_size": 64,
            "n_routed_experts": 4, "n_shared_experts": 0,
        })
        _mock_base_deps_wrapper(monkeypatch, cfg)
        model = Qwen3_5Moe(_make_mock_args(quant_target=["moe"]))
        mock_result = {"some.weight": torch.randn(4, 4)}
        monkeypatch.setattr(
            "amct_pytorch.common.models.llm.common.base.BaseModel.load_layer_weight",
            lambda self, prefix: mock_result,
        )
        assert model.load_layer_weight("model.language_model.layers.0.") is mock_result

    def test_qwen3_5_moe_load_embed_state_dict_delegates(self, monkeypatch):
        from amct_pytorch.common.models.llm.qwen.qwen3_5.qwen3_5_moe import Qwen3_5Moe
        from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeConfig
        monkeypatch.setattr(
            "amct_pytorch.common.models.llm.qwen.qwen3_5.qwen3_5_moe.get_weight_mappings",
            lambda path: {},
        )
        cfg = Qwen3_5MoeConfig(text_config={
            "num_hidden_layers": 2, "hidden_size": 64, "intermediate_size": 128,
            "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16,
            "rms_norm_eps": 1e-6, "vocab_size": 100, "max_position_embeddings": 512,
            "rope_theta": 10000.0, "tie_word_embeddings": False,
            "num_experts": 4, "moe_intermediate_size": 64,
            "n_routed_experts": 4, "n_shared_experts": 0,
        })
        _mock_base_deps_wrapper(monkeypatch, cfg)
        model = Qwen3_5Moe(_make_mock_args(quant_target=["moe"]))
        monkeypatch.setattr(
            "amct_pytorch.common.models.llm.common.base.BaseModel.load_embed_state_dict",
            lambda self: None,
        )
        model.load_embed_state_dict()

    def test_qwen3_5_moe_block_delegates(self, monkeypatch):
        from amct_pytorch.common.models.llm.qwen.qwen3_5.qwen3_5_moe import Qwen3_5Moe
        from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeConfig
        monkeypatch.setattr(
            "amct_pytorch.common.models.llm.qwen.qwen3_5.qwen3_5_moe.get_weight_mappings",
            lambda path: {},
        )
        cfg = Qwen3_5MoeConfig(text_config={
            "num_hidden_layers": 2, "hidden_size": 64, "intermediate_size": 128,
            "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16,
            "rms_norm_eps": 1e-6, "vocab_size": 100, "max_position_embeddings": 512,
            "rope_theta": 10000.0, "tie_word_embeddings": False,
            "num_experts": 4, "moe_intermediate_size": 64,
            "n_routed_experts": 4, "n_shared_experts": 0,
        })
        _mock_base_deps_wrapper(monkeypatch, cfg)
        model = Qwen3_5Moe(_make_mock_args(quant_target=["moe"]))
        mock_result = MagicMock()
        monkeypatch.setattr(
            "amct_pytorch.common.models.llm.common.base.BaseModel.block",
            lambda self, idx: mock_result,
        )
        assert model.block(0) is mock_result

    def test_qwen3_5_moe_do_embedding_forward_delegates(self, monkeypatch):
        from amct_pytorch.common.models.llm.qwen.qwen3_5.qwen3_5_moe import Qwen3_5Moe
        from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeConfig
        monkeypatch.setattr(
            "amct_pytorch.common.models.llm.qwen.qwen3_5.qwen3_5_moe.get_weight_mappings",
            lambda path: {},
        )
        cfg = Qwen3_5MoeConfig(text_config={
            "num_hidden_layers": 2, "hidden_size": 64, "intermediate_size": 128,
            "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16,
            "rms_norm_eps": 1e-6, "vocab_size": 100, "max_position_embeddings": 512,
            "rope_theta": 10000.0, "tie_word_embeddings": False,
            "num_experts": 4, "moe_intermediate_size": 64,
            "n_routed_experts": 4, "n_shared_experts": 0,
        })
        _mock_base_deps_wrapper(monkeypatch, cfg)
        model = Qwen3_5Moe(_make_mock_args(quant_target=["moe"]))
        mock_result = [torch.randn(2, 4, 8)]
        monkeypatch.setattr(
            "amct_pytorch.common.models.llm.common.base.BaseModel.do_embedding_forward",
            lambda self, samples, dtype=torch.bfloat16, hook_name=None: mock_result,
        )
        result = model.do_embedding_forward([torch.randn(2, 4)])
        assert result is mock_result

    def test_qwen3_5_moe_do_block_forward_delegates(self, monkeypatch):
        from amct_pytorch.common.models.llm.qwen.qwen3_5.qwen3_5_moe import Qwen3_5Moe
        from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeConfig
        monkeypatch.setattr(
            "amct_pytorch.common.models.llm.qwen.qwen3_5.qwen3_5_moe.get_weight_mappings",
            lambda path: {},
        )
        cfg = Qwen3_5MoeConfig(text_config={
            "num_hidden_layers": 2, "hidden_size": 64, "intermediate_size": 128,
            "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16,
            "rms_norm_eps": 1e-6, "vocab_size": 100, "max_position_embeddings": 512,
            "rope_theta": 10000.0, "tie_word_embeddings": False,
            "num_experts": 4, "moe_intermediate_size": 64,
            "n_routed_experts": 4, "n_shared_experts": 0,
        })
        _mock_base_deps_wrapper(monkeypatch, cfg)
        model = Qwen3_5Moe(_make_mock_args(quant_target=["moe"]))
        mock_result = [torch.randn(2, 4, 8)]
        monkeypatch.setattr(
            "amct_pytorch.common.models.llm.common.base.BaseModel.do_block_forward",
            lambda self, layer_idx, samples, hook_name=None, use_quant_block=False,
            enable_quant=False: mock_result,
        )
        result = model.do_block_forward(0, [torch.randn(2, 4, 8)])
        assert result is mock_result

    def test_qwen3_5_moe_do_head_forward_delegates(self, monkeypatch):
        from amct_pytorch.common.models.llm.qwen.qwen3_5.qwen3_5_moe import Qwen3_5Moe
        from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeConfig
        monkeypatch.setattr(
            "amct_pytorch.common.models.llm.qwen.qwen3_5.qwen3_5_moe.get_weight_mappings",
            lambda path: {},
        )
        cfg = Qwen3_5MoeConfig(text_config={
            "num_hidden_layers": 2, "hidden_size": 64, "intermediate_size": 128,
            "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16,
            "rms_norm_eps": 1e-6, "vocab_size": 100, "max_position_embeddings": 512,
            "rope_theta": 10000.0, "tie_word_embeddings": False,
            "num_experts": 4, "moe_intermediate_size": 64,
            "n_routed_experts": 4, "n_shared_experts": 0,
        })
        _mock_base_deps_wrapper(monkeypatch, cfg)
        model = Qwen3_5Moe(_make_mock_args(quant_target=["moe"]))
        mock_result = [torch.randn(2, 4, 100)]
        monkeypatch.setattr(
            "amct_pytorch.common.models.llm.common.base.BaseModel.do_head_forward",
            lambda self, inps: mock_result,
        )
        result = model.do_head_forward([torch.randn(2, 4, 8)])
        assert result is mock_result

    def test_qwen3_5_moe_load_unit_inputs_delegates(self, monkeypatch):
        from amct_pytorch.common.models.llm.qwen.qwen3_5.qwen3_5_moe import Qwen3_5Moe
        from amct_pytorch.common.models.llm.common.ptq_units import make_ptq_unit
        from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeConfig
        monkeypatch.setattr(
            "amct_pytorch.common.models.llm.qwen.qwen3_5.qwen3_5_moe.get_weight_mappings",
            lambda path: {},
        )
        cfg = Qwen3_5MoeConfig(text_config={
            "num_hidden_layers": 2, "hidden_size": 64, "intermediate_size": 128,
            "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16,
            "rms_norm_eps": 1e-6, "vocab_size": 100, "max_position_embeddings": 512,
            "rope_theta": 10000.0, "tie_word_embeddings": False,
            "num_experts": 4, "moe_intermediate_size": 64,
            "n_routed_experts": 4, "n_shared_experts": 0,
        })
        _mock_base_deps_wrapper(monkeypatch, cfg)
        model = Qwen3_5Moe(_make_mock_args(quant_target=["moe"]))
        unit = make_ptq_unit("moe", "mlp", layer_idx=0, module=nn.Linear(4, 4))
        fake_data = torch.randn(4, 4)
        monkeypatch.setattr(
            "amct_pytorch.common.models.llm.common.base.load_ptq_inps",
            lambda data_dir, kind, layer_idx: fake_data,
        )
        result = model.load_unit_inputs("/tmp/fake", unit)
        assert result is fake_data


def _make_glm5_cfg(num_layers=2, **extra):
    from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import GlmMoeDsaConfig

    cfg = GlmMoeDsaConfig(
        vocab_size=100, hidden_size=64, intermediate_size=128,
        num_hidden_layers=num_layers, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=512, rms_norm_eps=1e-6, tie_word_embeddings=False,
        **extra,
    )
    # Ensure required attributes exist (may not be part of HF config defaults)
    if not hasattr(cfg, "first_k_dense_replace"):
        cfg.first_k_dense_replace = 3
    if not hasattr(cfg, "moe_layer_freq"):
        cfg.moe_layer_freq = 1
    return cfg


def _setup_glm5(monkeypatch, cfg, quant_target=("mlp",)):
    from amct_pytorch.common.models.llm.glm.glm5.glm5 import GLM5

    _mock_base_deps(monkeypatch, cfg)
    args = _make_mock_args(quant_target=list(quant_target))
    return GLM5(args)


class TestGLM5Mocked:
    def test_glm5_init(self, monkeypatch):
        from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import GlmMoeDsaDecoderLayer

        cfg = _make_glm5_cfg()
        model = _setup_glm5(monkeypatch, cfg)
        assert model.num_layers == 2
        assert model.cls is GlmMoeDsaDecoderLayer
        assert model.quant_target == ["mlp"]

    def test_glm5_init_moe(self, monkeypatch):
        cfg = _make_glm5_cfg()
        model = _setup_glm5(monkeypatch, cfg, quant_target=["moe"])
        assert model.quant_target == ["moe"]
        assert model.num_layers == 2

    def test_glm5_init_attn_linear(self, monkeypatch):
        cfg = _make_glm5_cfg()
        model = _setup_glm5(monkeypatch, cfg, quant_target=["attn-linear"])
        assert model.quant_target == ["attn-linear"]

    # ---- get_layer_weight_prefix / _is_moe_layer ----------------

    def test_glm5_get_layer_weight_prefix(self, monkeypatch):
        model = _setup_glm5(monkeypatch, _make_glm5_cfg())
        assert model.get_layer_weight_prefix(3) == "model.layers.3."

    def test_glm5_is_moe_layer_dense_and_sparse(self, monkeypatch):
        cfg = _make_glm5_cfg(num_layers=6)
        cfg.first_k_dense_replace = 2
        cfg.moe_layer_freq = 2
        model = _setup_glm5(monkeypatch, cfg)
        assert model._is_moe_layer(0) is False  # 0 < 2
        assert model._is_moe_layer(1) is False  # 1 < 2
        assert model._is_moe_layer(2) is True   # 2 >= 2, 0%2=0
        assert model._is_moe_layer(3) is False  # 3 >= 2, 1%2=1
        assert model._is_moe_layer(4) is True   # 4 >= 2, 2%2=0
        assert model._is_moe_layer(5) is False  # 5 >= 2, 3%2=1

    def test_glm5_is_moe_layer_defaults_use_or_zero(self, monkeypatch):
        """When first_k_dense_replace=0, first dense layer index is 0, so all are moe."""
        cfg = _make_glm5_cfg(num_layers=3)
        cfg.first_k_dense_replace = 0
        cfg.moe_layer_freq = 1
        model = _setup_glm5(monkeypatch, cfg)
        # fd=0, mf=1 => all layers moe
        assert model._is_moe_layer(0) is True
        assert model._is_moe_layer(2) is True

    # ---- _ensure_glm5_config -------------------------------------

    def test_glm5_ensure_config_generates_mlp_types(self, monkeypatch):
        from amct_pytorch.common.models.llm.glm.glm5.glm5 import GLM5

        cfg = _make_glm5_cfg(num_layers=4)
        cfg.first_k_dense_replace = 1
        cfg.moe_layer_freq = 2
        # _mock_base_deps creates empty model which needs valid mlp_layer_types
        _mock_base_deps(monkeypatch, cfg)
        # Now clear mlp_layer_types so _ensure_glm5_config regenerates it
        cfg.mlp_layer_types = None
        args = _make_mock_args(quant_target=["mlp"])
        GLM5(args)
        assert cfg.mlp_layer_types == ["dense", "sparse", "dense", "sparse"]

    def test_glm5_ensure_config_skips_when_present(self, monkeypatch):
        cfg = _make_glm5_cfg()
        cfg.mlp_layer_types = ["dense", "dense"]
        model = _setup_glm5(monkeypatch, cfg)
        assert cfg.mlp_layer_types == ["dense", "dense"]

    # ---- block ---------------------------------------------------

    def test_glm5_block_dense_layer(self, monkeypatch):
        model = _setup_glm5(monkeypatch, _make_glm5_cfg())
        mock_layer = MagicMock()
        mock_layer.eval.return_value = mock_layer
        mock_layer.bfloat16.return_value = mock_layer
        monkeypatch.setattr(model, "cls", lambda cfg, idx: mock_layer)
        monkeypatch.setattr(model, "load_layer_weight", lambda prefix: {"some.weight": torch.randn(4, 8)})
        block = model.block(0)
        assert block is mock_layer

    def test_glm5_block_moe_layer_packs_expert_weights(self, monkeypatch):
        cfg = _make_glm5_cfg(num_layers=1)
        cfg.first_k_dense_replace = 0
        cfg.moe_layer_freq = 1
        model = _setup_glm5(monkeypatch, cfg)
        mock_layer = MagicMock()
        mock_layer.eval.return_value = mock_layer
        mock_layer.bfloat16.return_value = mock_layer
        monkeypatch.setattr(model, "cls", lambda cfg, idx: mock_layer)
        # State dict with expert weights that should be packed
        sd = {
            "mlp.experts.0.gate_proj.weight": torch.randn(8, 8),
            "mlp.experts.0.up_proj.weight": torch.randn(8, 8),
            "mlp.experts.0.down_proj.weight": torch.randn(8, 8),
            "mlp.experts.1.gate_proj.weight": torch.randn(8, 8),
            "mlp.experts.1.up_proj.weight": torch.randn(8, 8),
            "mlp.experts.1.down_proj.weight": torch.randn(8, 8),
        }
        monkeypatch.setattr(model, "load_layer_weight", lambda prefix: sd)
        block = model.block(0)
        assert block is mock_layer
        # Verify pack_gated_expert_weights was called: the packed keys should
        # have replaced the individual expert weight keys
        called_sd = mock_layer.load_state_dict.call_args[0][0]
        assert "mlp.experts.gate_up_proj" in called_sd
        assert "mlp.experts.down_proj" in called_sd

    # ---- build_quant_block ---------------------------------------

    def test_glm5_build_quant_block_attn_cache_not_impl(self, monkeypatch):
        model = _setup_glm5(monkeypatch, _make_glm5_cfg(), quant_target=["attn-cache"])
        # Mock block to avoid real decoder layer construction
        mock_block = nn.Module()
        monkeypatch.setattr(model, "block", lambda idx: mock_block)
        with pytest.raises(NotImplementedError, match="does not support quant_target='attn-cache'"):
            model.build_quant_block(0)

    def test_glm5_build_quant_block_attn_linear(self, monkeypatch):
        from amct_pytorch.common.models.llm.glm.glm5 import glm5 as glm5_mod

        model = _setup_glm5(monkeypatch, _make_glm5_cfg(), quant_target=["attn-linear"])
        mock_block = nn.Module()
        mock_block.self_attn = nn.Linear(4, 4)
        mock_block.mlp = nn.Linear(4, 4)
        monkeypatch.setattr(model, "block", lambda idx: mock_block)

        # Mock QuantGLM5Attn constructor to avoid deep dependencies
        captured = {}
        def _mock_attn(args, module):
            captured["called"] = True
            captured["module"] = module
            return module

        monkeypatch.setattr(glm5_mod, "QuantGLM5Attn", _mock_attn)
        monkeypatch.setattr(glm5_mod, "QuantGLM5MLP", lambda args, module: module)

        result = model.build_quant_block(0)
        assert captured.get("called") is True
        assert isinstance(result, nn.Module)

    def test_glm5_build_quant_block_attn_linear_also_mlp(self, monkeypatch):
        from amct_pytorch.common.models.llm.glm.glm5 import glm5 as glm5_mod

        model = _setup_glm5(monkeypatch, _make_glm5_cfg(), quant_target=["attn-linear", "mlp"])
        mock_block = nn.Module()
        mock_block.self_attn = nn.Linear(4, 4)
        mock_block.mlp = nn.Linear(4, 4)
        monkeypatch.setattr(model, "block", lambda idx: mock_block)
        monkeypatch.setattr(glm5_mod, "QuantGLM5Attn", lambda args, module: module)
        captured_mlp = {}
        def _mock_mlp(args, module):
            captured_mlp["called"] = True
            return module
        monkeypatch.setattr(glm5_mod, "QuantGLM5MLP", _mock_mlp)
        result = model.build_quant_block(0)
        assert captured_mlp.get("called") is True

    def test_glm5_build_quant_block_no_mlp(self, monkeypatch):
        model = _setup_glm5(monkeypatch, _make_glm5_cfg(), quant_target=["mlp"])
        mock_block = nn.Module()  # no mlp attribute
        monkeypatch.setattr(model, "block", lambda idx: mock_block)
        result = model.build_quant_block(0)
        assert result is mock_block

    def test_glm5_build_quant_block_mlp_only(self, monkeypatch):
        from amct_pytorch.common.models.llm.glm.glm5 import glm5 as glm5_mod

        model = _setup_glm5(monkeypatch, _make_glm5_cfg(), quant_target=["mlp"])
        mock_block = nn.Module()
        mock_block.mlp = nn.Linear(4, 4)
        monkeypatch.setattr(model, "block", lambda idx: mock_block)
        captured = {}
        def _mock_mlp(args, module):
            captured["called"] = True
            captured["module"] = module
            return module
        monkeypatch.setattr(glm5_mod, "QuantGLM5MLP", _mock_mlp)
        result = model.build_quant_block(0)
        assert captured.get("called") is True

    def test_glm5_build_quant_block_moe_not_packed(self, monkeypatch):
        model = _setup_glm5(monkeypatch, _make_glm5_cfg(), quant_target=["moe"])
        mock_block = nn.Module()
        mock_block.mlp = nn.Module()
        # experts without gate_up_proj/down_proj attrs -> not packed
        mock_block.mlp.experts = nn.Linear(4, 4)
        monkeypatch.setattr(model, "block", lambda idx: mock_block)
        with pytest.raises(TypeError, match="Expected packed experts"):
            model.build_quant_block(0)

    def test_glm5_build_quant_block_moe_packed_experts(self, monkeypatch):
        from amct_pytorch.common.models.llm.glm.glm5 import glm5 as glm5_mod

        model = _setup_glm5(monkeypatch, _make_glm5_cfg(), quant_target=["moe"])
        mock_block = nn.Module()
        mock_block.mlp = nn.Module()
        # packed experts must have gate_up_proj and down_proj
        mock_block.mlp.experts = type("PE", (), {"gate_up_proj": None, "down_proj": None, "num_experts": 2})()
        monkeypatch.setattr(model, "block", lambda idx: mock_block)

        captured = {}
        def _mock_ge(args, experts_module):
            captured["called"] = True
            captured["experts"] = experts_module
            return experts_module
        monkeypatch.setattr(glm5_mod, "QuantGatedExperts", _mock_ge)
        result = model.build_quant_block(0)
        assert captured.get("called") is True

    def test_glm5_build_quant_block_moe_and_mlp_shared_experts(self, monkeypatch):
        from amct_pytorch.common.models.llm.glm.glm5 import glm5 as glm5_mod

        model = _setup_glm5(monkeypatch, _make_glm5_cfg(), quant_target=["moe", "mlp"])
        mock_block = nn.Module()
        mock_block.mlp = nn.Module()
        mock_block.mlp.experts = type("PE", (), {"gate_up_proj": None, "down_proj": None, "num_experts": 2})()
        mock_block.mlp.shared_experts = nn.Linear(4, 4)
        monkeypatch.setattr(model, "block", lambda idx: mock_block)
        ge_captured = {}
        def _mock_ge(args, experts_module):
            ge_captured["called"] = True
            return experts_module
        monkeypatch.setattr(glm5_mod, "QuantGatedExperts", _mock_ge)
        se_captured = {}
        def _mock_se(args, module):
            se_captured["called"] = True
            return module
        monkeypatch.setattr(glm5_mod, "QuantGLM5MLP", _mock_se)
        result = model.build_quant_block(0)
        assert ge_captured.get("called") is True
        assert se_captured.get("called") is True

    # ---- iter_ptq_units -------------------------------------------

    def test_glm5_iter_ptq_units_mlp(self, monkeypatch):
        model = _setup_glm5(monkeypatch, _make_glm5_cfg(), quant_target=["mlp"])
        block = _make_block_with_attn_and_mlp()
        units = list(model.iter_ptq_units(0, block))
        assert len(units) == 1
        assert units[0].kind == "mlp"
        assert units[0].name == "mlp"

    def test_glm5_iter_ptq_units_attn_linear(self, monkeypatch):
        model = _setup_glm5(monkeypatch, _make_glm5_cfg(), quant_target=["attn-linear"])
        block = _make_block_with_attn_and_mlp()
        units = list(model.iter_ptq_units(0, block))
        assert len(units) == 1
        assert units[0].kind == "attn"
        assert units[0].name == "self_attn"

    def test_glm5_iter_ptq_units_attn_cache(self, monkeypatch):
        model = _setup_glm5(monkeypatch, _make_glm5_cfg(), quant_target=["attn-cache"])
        block = _make_block_with_attn_and_mlp()
        units = list(model.iter_ptq_units(0, block))
        assert len(units) == 1
        assert units[0].kind == "attn"

    def test_glm5_iter_ptq_units_attn_no_self_attn_fallback(self, monkeypatch):
        model = _setup_glm5(monkeypatch, _make_glm5_cfg(), quant_target=["attn-linear"])
        block = nn.Module()  # no self_attn, no mlp
        block.mlp = nn.Linear(4, 4)  # super will use this
        # The GLM5 code warns and falls back to super().iter_ptq_units
        # BaseModel's iter_ptq_units will yield attn unit from block itself
        units = list(model.iter_ptq_units(0, block))
        # BaseModel yields an attn unit from block.self_attn (None)
        assert len(units) >= 1
        assert units[0].kind == "attn"

    def test_glm5_iter_ptq_units_no_mlp_raises(self, monkeypatch):
        model = _setup_glm5(monkeypatch, _make_glm5_cfg(), quant_target=["unknown"])
        block = nn.Module()  # no mlp
        with pytest.raises(ValueError, match="Unsupported quant target"):
            list(model.iter_ptq_units(0, block))

    def test_glm5_iter_ptq_units_moe_no_experts(self, monkeypatch):
        model = _setup_glm5(monkeypatch, _make_glm5_cfg(), quant_target=["moe"])
        block = nn.Module()
        block.mlp = nn.Module()  # no experts attr
        units = list(model.iter_ptq_units(0, block))
        assert units == []

    def test_glm5_iter_ptq_units_moe_not_quant_gated_raises(self, monkeypatch):
        model = _setup_glm5(monkeypatch, _make_glm5_cfg(), quant_target=["moe"])
        block = nn.Module()
        block.mlp = nn.Module()
        block.mlp.experts = nn.Linear(4, 4)  # not QuantGatedExperts
        with pytest.raises(TypeError, match="Expected QuantGatedExperts"):
            list(model.iter_ptq_units(0, block))

    def test_glm5_iter_ptq_units_moe_with_quant_gated_experts(self, monkeypatch):
        from amct_pytorch.common.models.llm.qwen.moe_common import QuantGatedExperts

        model = _setup_glm5(monkeypatch, _make_glm5_cfg(), quant_target=["moe"])
        block = nn.Module()
        block.mlp = nn.Module()
        expert0 = nn.Linear(4, 4)
        expert1 = nn.Linear(4, 4)
        # Create QuantGatedExperts via __new__ + manual Module.__init__
        experts = QuantGatedExperts.__new__(QuantGatedExperts)
        nn.Module.__init__(experts)
        experts.expert_modules = nn.ModuleList([expert0, expert1])
        block.mlp.experts = experts
        units = list(model.iter_ptq_units(0, block))
        assert len(units) == 2
        assert all(u.kind == "moe" for u in units)
        assert units[0].name == "expert_0"
        assert units[1].name == "expert_1"
        assert units[0].metadata == {"expert_idx": 0}

    def test_glm5_iter_ptq_units_mlp_with_shared_experts(self, monkeypatch):
        model = _setup_glm5(monkeypatch, _make_glm5_cfg(), quant_target=["mlp"])
        block = nn.Module()
        block.mlp = nn.Module()
        shared = nn.Linear(4, 4)
        block.mlp.shared_experts = shared
        units = list(model.iter_ptq_units(0, block))
        assert len(units) == 1
        assert units[0].name == "shared_experts"
        assert units[0].kind == "mlp"

    # ---- iter_deploy_bindings -------------------------------------

    def test_glm5_iter_deploy_bindings_regular(self, monkeypatch):
        from amct_pytorch.quantization.modules.quant_linear import QuantLinear

        model = _setup_glm5(monkeypatch, _make_glm5_cfg())
        ql = QuantLinear.__new__(QuantLinear)
        ql._parameters = {}
        ql._buffers = {}
        ql._modules = {}
        block = nn.Module()
        block.q_proj = ql
        bindings = list(model.iter_deploy_bindings(0, block))
        assert len(bindings) == 1
        assert bindings[0][0] == "model.layers.0.q_proj.weight"
        assert bindings[0][1] is ql

    def test_glm5_iter_deploy_bindings_expert_name(self, monkeypatch):
        from amct_pytorch.quantization.modules.quant_linear import QuantLinear

        model = _setup_glm5(monkeypatch, _make_glm5_cfg())
        ql = QuantLinear.__new__(QuantLinear)
        ql._parameters = {}
        ql._buffers = {}
        ql._modules = {}
        block = nn.Module()
        block.mlp = nn.Module()
        block.mlp.experts = nn.Module()
        # Create ModuleList with 3 modules (indices 0,1,2)
        expert0 = nn.Module()
        expert1 = nn.Module()
        expert2 = nn.Module()
        expert2.up_proj = ql
        block.mlp.experts.expert_modules = nn.ModuleList([expert0, expert1, expert2])
        bindings = list(model.iter_deploy_bindings(0, block))
        assert len(bindings) == 1
        assert bindings[0][0] == "model.layers.0.mlp.experts.2.up_proj.weight"

    def test_glm5_iter_deploy_bindings_invalid_expert_name(self, monkeypatch):
        from amct_pytorch.quantization.modules.quant_linear import QuantLinear

        model = _setup_glm5(monkeypatch, _make_glm5_cfg())
        ql = QuantLinear.__new__(QuantLinear)
        ql._parameters = {}
        ql._buffers = {}
        ql._modules = {}
        block = nn.Module()
        block.mlp = nn.Module()
        block.mlp.experts = nn.Module()
        expert0 = nn.Module()
        # Name has wrong number of parts (6 instead of 5): mlp.experts.expert_modules.0.extra.proj
        expert0.extra = nn.Module()
        expert0.extra.proj = ql
        block.mlp.experts.expert_modules = nn.ModuleList([expert0])
        with pytest.raises(ValueError, match="Unexpected GLM5 expert module name"):
            list(model.iter_deploy_bindings(0, block))

    def test_glm5_iter_deploy_bindings_skips_non_quant_linear(self, monkeypatch):
        model = _setup_glm5(monkeypatch, _make_glm5_cfg())
        block = nn.Module()
        block.linear = nn.Linear(4, 4)
        bindings = list(model.iter_deploy_bindings(0, block))
        assert bindings == []

    # ---- load_unit_inputs -----------------------------------------

    def test_glm5_load_unit_inputs(self, monkeypatch):
        from amct_pytorch.common.models.llm.common.ptq_units import make_ptq_unit

        model = _setup_glm5(monkeypatch, _make_glm5_cfg())
        unit = make_ptq_unit("mlp", "mlp", layer_idx=2, module=nn.Linear(4, 4))
        fake_data = torch.randn(4, 4)
        monkeypatch.setattr(
            "amct_pytorch.common.models.llm.common.base.load_ptq_inps",
            lambda data_dir, kind, layer_idx: fake_data,
        )
        result = model.load_unit_inputs("/tmp/fake", unit)
        assert result is fake_data

    # ---- delegation methods ----------------------------------------

    def test_glm5_float_model_delegates_to_super(self, monkeypatch):
        model = _setup_glm5(monkeypatch, _make_glm5_cfg())
        mock_result = MagicMock()
        monkeypatch.setattr(
            "transformers.AutoModelForCausalLM.from_pretrained",
            lambda *a, **kw: mock_result,
        )
        import transformers
        original = transformers.AutoModelForCausalLM.from_pretrained
        try:
            transformers.AutoModelForCausalLM.from_pretrained = lambda *a, **kw: mock_result
            assert model.float_model() is mock_result
        finally:
            transformers.AutoModelForCausalLM.from_pretrained = original

    def test_glm5_empty_weights_model_delegates_to_super(self, monkeypatch):
        model = _setup_glm5(monkeypatch, _make_glm5_cfg())
        mock_result = MagicMock()
        monkeypatch.setattr(
            "amct_pytorch.common.models.llm.common.base.BaseModel.empty_weights_model",
            lambda self: mock_result,
        )
        assert model.empty_weights_model() is mock_result

    def test_glm5_load_layer_weight_delegates_to_super(self, monkeypatch):
        model = _setup_glm5(monkeypatch, _make_glm5_cfg())
        mock_result = {"some.weight": torch.randn(4, 4)}
        monkeypatch.setattr(
            "amct_pytorch.common.models.llm.common.base.BaseModel.load_layer_weight",
            lambda self, prefix: mock_result,
        )
        assert model.load_layer_weight("model.layers.0.") is mock_result

    def test_glm5_load_embed_state_dict_delegates_to_super(self, monkeypatch):
        model = _setup_glm5(monkeypatch, _make_glm5_cfg())
        monkeypatch.setattr(
            "amct_pytorch.common.models.llm.common.base.BaseModel.load_embed_state_dict",
            lambda self: None,
        )
        model.load_embed_state_dict()

    def test_glm5_do_embedding_forward_delegates_to_super(self, monkeypatch):
        model = _setup_glm5(monkeypatch, _make_glm5_cfg())
        mock_result = [torch.randn(2, 4, 8)]
        monkeypatch.setattr(
            "amct_pytorch.common.models.llm.common.base.BaseModel.do_embedding_forward",
            lambda self, samples, dtype=torch.bfloat16, hook_name=None: mock_result,
        )
        result = model.do_embedding_forward([torch.randn(2, 4)])
        assert result is mock_result

    def test_glm5_do_block_forward_delegates_to_super(self, monkeypatch):
        model = _setup_glm5(monkeypatch, _make_glm5_cfg())
        mock_result = [torch.randn(2, 4, 8)]
        monkeypatch.setattr(
            "amct_pytorch.common.models.llm.common.base.BaseModel.do_block_forward",
            lambda self, layer_idx, samples, hook_name=None, use_quant_block=False,
            enable_quant=False: mock_result,
        )
        result = model.do_block_forward(0, [torch.randn(2, 4, 8)])
        assert result is mock_result

    def test_glm5_do_head_forward_delegates_to_super(self, monkeypatch):
        model = _setup_glm5(monkeypatch, _make_glm5_cfg())
        mock_result = [torch.randn(2, 4, 100)]
        monkeypatch.setattr(
            "amct_pytorch.common.models.llm.common.base.BaseModel.do_head_forward",
            lambda self, inps: mock_result,
        )
        result = model.do_head_forward([torch.randn(2, 4, 8)])
        assert result is mock_result


def _mock_pangu_deps(monkeypatch, num_layers=2):
    """Mock all HF dependencies for Pangu 7B (trust_remote_code, not in transformers core)."""
    cfg = SimpleNamespace()
    cfg.num_hidden_layers = num_layers
    cfg.hidden_size = 64
    cfg._attn_implementation = None

    fake_tokenizer = MagicMock()

    monkeypatch.setattr(
        "amct_pytorch.common.models.llm.common.base.AutoConfig",
        type("FakeAC", (), {"from_pretrained": staticmethod(lambda *a, **kw: cfg)})(),
    )
    monkeypatch.setattr(
        "amct_pytorch.common.models.llm.common.base.AutoTokenizer",
        type("FakeAT", (), {"from_pretrained": staticmethod(lambda *a, **kw: fake_tokenizer)})(),
    )
    monkeypatch.setattr(
        "amct_pytorch.common.models.llm.common.base.init_empty_weights",
        lambda: MagicMock(__enter__=lambda s: None, __exit__=lambda *a: None),
    )

    mock_layer = nn.Linear(4, 4)
    mock_model = MagicMock()
    mock_model.model = MagicMock()
    mock_model.model.layers = [mock_layer]
    mock_model.model.norm = nn.Module()
    mock_model.model.embed_tokens = nn.Module()
    mock_model.lm_head = nn.Module()

    monkeypatch.setattr(
        "amct_pytorch.common.models.llm.common.base.AutoModelForCausalLM",
        type("FakeAMFCLM", (), {
            "from_pretrained": staticmethod(lambda *a, **kw: mock_model),
            "from_config": staticmethod(lambda *a, **kw: mock_model),
        })(),
    )
    monkeypatch.setattr(
        "amct_pytorch.common.models.llm.common.base.get_weight_mappings",
        lambda path: {},
    )


class TestPangu7BMocked:
    def test_pangu_init(self, monkeypatch):
        from amct_pytorch.common.models.llm.pangu.pangu_7b.pangu_7b import Pangu7B
        _mock_pangu_deps(monkeypatch)
        args = _make_mock_args(quant_target=["mlp"])
        model = Pangu7B(args)
        assert model.num_layers == 2
        assert model.quant_target == ["mlp"]

    def test_pangu_iter_ptq_units_mlp(self, monkeypatch):
        from amct_pytorch.common.models.llm.pangu.pangu_7b.pangu_7b import Pangu7B
        _mock_pangu_deps(monkeypatch)
        args = _make_mock_args(quant_target=["mlp"])
        model = Pangu7B(args)
        block = _make_block_with_attn_and_mlp()
        units = list(model.iter_ptq_units(0, block))
        assert len(units) == 1
        assert units[0].kind == "mlp"

    def test_pangu_get_layer_weight_prefix(self, monkeypatch):
        from amct_pytorch.common.models.llm.pangu.pangu_7b.pangu_7b import Pangu7B
        _mock_pangu_deps(monkeypatch)
        args = _make_mock_args(quant_target=["mlp"])
        model = Pangu7B(args)
        assert model.get_layer_weight_prefix(0) == "model.layers.0."

    def test_pangu_parse_quant_mode_rejects_moe(self, monkeypatch):
        from amct_pytorch.common.models.llm.pangu.pangu_7b.pangu_7b import Pangu7B
        _mock_pangu_deps(monkeypatch)
        args = _make_mock_args(quant_target=["moe"])
        with pytest.raises(ValueError, match="does not support quant_target='moe'"):
            Pangu7B(args)

    def test_pangu_build_quant_block_mlp(self, monkeypatch):
        from amct_pytorch.common.models.llm.pangu.pangu_7b.pangu_7b import Pangu7B
        _mock_pangu_deps(monkeypatch)
        args = _make_mock_args(quant_target=["mlp"])
        model = Pangu7B(args)
        mock_block = nn.Linear(4, 4)
        monkeypatch.setattr(model, "block", lambda idx: mock_block)
        monkeypatch.setattr(
            "amct_pytorch.common.models.llm.pangu.pangu_7b.pangu_7b.apply_quant_to_moe_mlp",
            lambda args_in, dl, **kwargs: dl,
        )
        result = model.build_quant_block(0)
        assert result is mock_block

    def test_pangu_build_quant_block_attn_linear(self, monkeypatch):
        from amct_pytorch.common.models.llm.pangu.pangu_7b.pangu_7b import Pangu7B
        _mock_pangu_deps(monkeypatch)
        args = _make_mock_args(quant_target=["attn-linear"])
        model = Pangu7B(args)
        mock_block = nn.Linear(4, 4)
        monkeypatch.setattr(model, "block", lambda idx: mock_block)
        monkeypatch.setattr(
            "amct_pytorch.common.models.llm.pangu.pangu_7b.pangu_7b.apply_quant_to_attn",
            lambda args_in, dl, cls: dl,
        )
        result = model.build_quant_block(0)
        assert result is mock_block

    def test_pangu_build_quant_block_both(self, monkeypatch):
        from amct_pytorch.common.models.llm.pangu.pangu_7b.pangu_7b import Pangu7B
        _mock_pangu_deps(monkeypatch)
        args = _make_mock_args(quant_target=["attn-cache", "mlp"])
        model = Pangu7B(args)
        mock_block = nn.Linear(4, 4)
        monkeypatch.setattr(model, "block", lambda idx: mock_block)
        captured = {}
        def _mock_attn(args_in, dl, cls):
            captured["attn"] = True
            return dl
        def _mock_mlp(args_in, dl, **kwargs):
            captured["mlp"] = True
            return dl
        monkeypatch.setattr(
            "amct_pytorch.common.models.llm.pangu.pangu_7b.pangu_7b.apply_quant_to_attn", _mock_attn
        )
        monkeypatch.setattr(
            "amct_pytorch.common.models.llm.pangu.pangu_7b.pangu_7b.apply_quant_to_moe_mlp", _mock_mlp
        )
        result = model.build_quant_block(0)
        assert captured.get("attn") is True
        assert captured.get("mlp") is True

    def test_pangu_iter_ptq_units_attn(self, monkeypatch):
        from amct_pytorch.common.models.llm.pangu.pangu_7b.pangu_7b import Pangu7B
        _mock_pangu_deps(monkeypatch)
        args = _make_mock_args(quant_target=["attn-linear"])
        model = Pangu7B(args)
        block = _make_block_with_attn_and_mlp()
        units = list(model.iter_ptq_units(0, block))
        assert len(units) == 1
        assert units[0].kind == "attn"
        assert units[0].name == "self_attn"

    def test_pangu_iter_ptq_units_attn_cache(self, monkeypatch):
        from amct_pytorch.common.models.llm.pangu.pangu_7b.pangu_7b import Pangu7B
        _mock_pangu_deps(monkeypatch)
        args = _make_mock_args(quant_target=["attn-cache"])
        model = Pangu7B(args)
        block = _make_block_with_attn_and_mlp()
        units = list(model.iter_ptq_units(0, block))
        assert len(units) == 1
        assert units[0].kind == "attn"

    def test_pangu_iter_deploy_bindings_delegates(self, monkeypatch):
        from amct_pytorch.common.models.llm.pangu.pangu_7b.pangu_7b import Pangu7B
        from amct_pytorch.quantization.modules.quant_linear import QuantLinear
        _mock_pangu_deps(monkeypatch)
        args = _make_mock_args(quant_target=["mlp"])
        model = Pangu7B(args)
        ql = QuantLinear.__new__(QuantLinear)
        ql._parameters = {}
        ql._modules = {}
        ql._buffers = {}
        block = nn.Module()
        block.q_proj = ql
        bindings = list(model.iter_deploy_bindings(0, block))
        assert len(bindings) == 1

    def test_pangu_block(self, monkeypatch):
        from amct_pytorch.common.models.llm.pangu.pangu_7b.pangu_7b import Pangu7B
        _mock_pangu_deps(monkeypatch)
        args = _make_mock_args(quant_target=["mlp"])
        model = Pangu7B(args)
        mock_layer = MagicMock()
        mock_layer.eval.return_value = mock_layer
        mock_layer.bfloat16.return_value = mock_layer
        monkeypatch.setattr(model, "cls", lambda cfg_in, idx: mock_layer)
        monkeypatch.setattr(model, "load_layer_weight", lambda prefix: {})
        block = model.block(0)
        assert block is mock_layer

    def test_pangu_float_model(self, monkeypatch):
        from amct_pytorch.common.models.llm.pangu.pangu_7b.pangu_7b import Pangu7B
        _mock_pangu_deps(monkeypatch)
        args = _make_mock_args(quant_target=["mlp"])
        model = Pangu7B(args)
        mock_model = MagicMock()
        monkeypatch.setattr(
            "amct_pytorch.common.models.llm.pangu.pangu_7b.pangu_7b.AutoModelForCausalLM",
            type("FakeAM", (), {
                "from_pretrained": staticmethod(lambda *a, **kw: mock_model),
            })(),
        )
        monkeypatch.setattr(
            "amct_pytorch.common.models.llm.pangu.pangu_7b.pangu_7b._ensure_transformers_compat",
            lambda: None,
        )
        monkeypatch.setattr(
            "amct_pytorch.common.models.llm.pangu.pangu_7b.pangu_7b._disable_npu_fused_attention",
            lambda m: None,
        )
        result = model.float_model()
        assert result is mock_model

    def test_pangu_load_layer_weight_delegates(self, monkeypatch):
        from amct_pytorch.common.models.llm.pangu.pangu_7b.pangu_7b import Pangu7B
        _mock_pangu_deps(monkeypatch)
        model = Pangu7B(_make_mock_args(quant_target=["mlp"]))
        mock_result = {"some.weight": torch.randn(4, 4)}
        monkeypatch.setattr(
            "amct_pytorch.common.models.llm.common.base.BaseModel.load_layer_weight",
            lambda self, prefix: mock_result,
        )
        assert model.load_layer_weight("model.layers.0.") is mock_result

    def test_pangu_load_embed_state_dict_delegates(self, monkeypatch):
        from amct_pytorch.common.models.llm.pangu.pangu_7b.pangu_7b import Pangu7B
        _mock_pangu_deps(monkeypatch)
        model = Pangu7B(_make_mock_args(quant_target=["mlp"]))
        monkeypatch.setattr(
            "amct_pytorch.common.models.llm.common.base.BaseModel.load_embed_state_dict",
            lambda self: None,
        )
        model.load_embed_state_dict()

    def test_pangu_do_embedding_forward_delegates(self, monkeypatch):
        from amct_pytorch.common.models.llm.pangu.pangu_7b.pangu_7b import Pangu7B
        _mock_pangu_deps(monkeypatch)
        model = Pangu7B(_make_mock_args(quant_target=["mlp"]))
        mock_result = [torch.randn(2, 4, 8)]
        monkeypatch.setattr(
            "amct_pytorch.common.models.llm.common.base.BaseModel.do_embedding_forward",
            lambda self, samples, dtype=torch.bfloat16, hook_name=None: mock_result,
        )
        result = model.do_embedding_forward([torch.randn(2, 4)])
        assert result is mock_result

    def test_pangu_do_block_forward_delegates(self, monkeypatch):
        from amct_pytorch.common.models.llm.pangu.pangu_7b.pangu_7b import Pangu7B
        _mock_pangu_deps(monkeypatch)
        model = Pangu7B(_make_mock_args(quant_target=["mlp"]))
        mock_result = [torch.randn(2, 4, 8)]
        monkeypatch.setattr(
            "amct_pytorch.common.models.llm.common.base.BaseModel.do_block_forward",
            lambda self, layer_idx, samples, hook_name=None, use_quant_block=False,
            enable_quant=False: mock_result,
        )
        result = model.do_block_forward(0, [torch.randn(2, 4, 8)])
        assert result is mock_result

    def test_pangu_do_head_forward_delegates(self, monkeypatch):
        from amct_pytorch.common.models.llm.pangu.pangu_7b.pangu_7b import Pangu7B
        _mock_pangu_deps(monkeypatch)
        model = Pangu7B(_make_mock_args(quant_target=["mlp"]))
        mock_result = [torch.randn(2, 4, 100)]
        monkeypatch.setattr(
            "amct_pytorch.common.models.llm.common.base.BaseModel.do_head_forward",
            lambda self, inps: mock_result,
        )
        result = model.do_head_forward([torch.randn(2, 4, 8)])
        assert result is mock_result

    def test_pangu_load_unit_inputs_delegates(self, monkeypatch):
        from amct_pytorch.common.models.llm.pangu.pangu_7b.pangu_7b import Pangu7B
        from amct_pytorch.common.models.llm.common.ptq_units import make_ptq_unit
        _mock_pangu_deps(monkeypatch)
        model = Pangu7B(_make_mock_args(quant_target=["mlp"]))
        unit = make_ptq_unit("mlp", "mlp", layer_idx=0, module=nn.Linear(4, 4))
        fake_data = torch.randn(4, 4)
        monkeypatch.setattr(
            "amct_pytorch.common.models.llm.common.base.load_ptq_inps",
            lambda data_dir, kind, layer_idx: fake_data,
        )
        result = model.load_unit_inputs("/tmp/fake", unit)
        assert result is fake_data


class TestLongcatNextMocked:
    def test_longcat_next_init(self, monkeypatch):
        from amct_pytorch.common.models.llm.longcat.longcat_next.longcat_next import LongcatNext
        from transformers.models.longcat_flash.configuration_longcat_flash import LongcatFlashConfig
        monkeypatch.setattr(
            LongcatNext,
            "empty_weights_model",
            lambda self: AutoModelForCausalLM.from_config(
                self.config, trust_remote_code=True, torch_dtype=torch.bfloat16),
        )
        cfg = LongcatFlashConfig(
            num_hidden_layers=2, num_layers=2, hidden_size=64, intermediate_size=128,
            num_attention_heads=4, num_key_value_heads=2, head_dim=16,
            rms_norm_eps=1e-6, vocab_size=100, max_position_embeddings=512,
            tie_word_embeddings=False,
        )
        _mock_base_deps(monkeypatch, cfg)
        args = _make_mock_args(quant_target=["mlp"])
        model = LongcatNext(args)
        assert model.num_layers == 2

    def test_longcat_next_get_layer_weight_prefix(self, monkeypatch):
        from amct_pytorch.common.models.llm.longcat.longcat_next.longcat_next import LongcatNext
        from transformers.models.longcat_flash.configuration_longcat_flash import LongcatFlashConfig
        monkeypatch.setattr(
            LongcatNext,
            "empty_weights_model",
            lambda self: AutoModelForCausalLM.from_config(
                self.config, trust_remote_code=True, torch_dtype=torch.bfloat16),
        )
        cfg = LongcatFlashConfig(
            num_hidden_layers=2, num_layers=2, hidden_size=64, intermediate_size=128,
            num_attention_heads=4, num_key_value_heads=2, head_dim=16,
            rms_norm_eps=1e-6, vocab_size=100, max_position_embeddings=512,
            tie_word_embeddings=False,
        )
        _mock_base_deps(monkeypatch, cfg)
        args = _make_mock_args(quant_target=["mlp"])
        model = LongcatNext(args)
        assert model.get_layer_weight_prefix(0) == "model.layers.0."


class TestLongcatLiteMocked:
    def test_longcat_lite_init(self, monkeypatch):
        from amct_pytorch.common.models.llm.longcat.longcat_lite.longcat_lite import LongcatLite
        from transformers.models.longcat_flash.configuration_longcat_flash import LongcatFlashConfig
        cfg = LongcatFlashConfig(
            num_hidden_layers=2, num_layers=2, hidden_size=64, intermediate_size=128,
            num_attention_heads=4, num_key_value_heads=2, head_dim=16,
            rms_norm_eps=1e-6, vocab_size=100, max_position_embeddings=512,
            tie_word_embeddings=False,
        )
        _mock_base_deps(monkeypatch, cfg)
        args = _make_mock_args(quant_target=["mlp"])
        model = LongcatLite(args)
        assert model.num_layers == 2

    def test_longcat_lite_iter_ptq_units_mlp(self, monkeypatch):
        from amct_pytorch.common.models.llm.longcat.longcat_lite.longcat_lite import LongcatLite
        from transformers.models.longcat_flash.configuration_longcat_flash import LongcatFlashConfig
        cfg = LongcatFlashConfig(
            num_hidden_layers=2, num_layers=2, hidden_size=64, intermediate_size=128,
            num_attention_heads=4, num_key_value_heads=2, head_dim=16,
            rms_norm_eps=1e-6, vocab_size=100, max_position_embeddings=512,
            tie_word_embeddings=False,
        )
        _mock_base_deps(monkeypatch, cfg)
        args = _make_mock_args(quant_target=["mlp"])
        model = LongcatLite(args)
        block = nn.Module()
        block.mlps = nn.ModuleList([nn.Linear(4, 4)])
        units = list(model.iter_ptq_units(0, block))
        assert len(units) == 1
        assert units[0].kind == "mlp"


class TestMoeCommon:
    """Tests for amct_pytorch/common/models/llm/qwen/moe_common.py"""

    def test_is_packed_experts_true(self):
        from amct_pytorch.common.models.llm.qwen.moe_common import is_packed_experts
        experts = type("PE", (), {"gate_up_proj": None, "down_proj": None})()
        assert is_packed_experts(experts) is True

    def test_is_packed_experts_false(self):
        from amct_pytorch.common.models.llm.qwen.moe_common import is_packed_experts
        experts = nn.Linear(4, 4)
        assert is_packed_experts(experts) is False
        assert is_packed_experts(nn.Module()) is False

    def test_pack_gated_expert_weights(self):
        from amct_pytorch.common.models.llm.qwen.moe_common import pack_gated_expert_weights
        sd = {
            "mlp.experts.0.gate_proj.weight": torch.randn(8, 8),
            "mlp.experts.0.up_proj.weight": torch.randn(8, 8),
            "mlp.experts.0.down_proj.weight": torch.randn(8, 8),
            "mlp.experts.1.gate_proj.weight": torch.randn(8, 8),
            "mlp.experts.1.up_proj.weight": torch.randn(8, 8),
            "mlp.experts.1.down_proj.weight": torch.randn(8, 8),
            "other.weight": torch.randn(4, 4),
        }
        result = pack_gated_expert_weights(dict(sd))
        assert "mlp.experts.gate_up_proj" in result
        assert result["mlp.experts.gate_up_proj"].shape == (2, 16, 8)
        assert result["mlp.experts.down_proj"].shape == (2, 8, 8)
        assert "other.weight" in result

    def test_pack_gated_expert_weights_no_experts(self):
        from amct_pytorch.common.models.llm.qwen.moe_common import pack_gated_expert_weights
        sd = {"other.weight": torch.randn(4, 4)}
        result = pack_gated_expert_weights(dict(sd))
        assert result == sd

    def test_pack_gated_expert_weights_inconsistent(self):
        from amct_pytorch.common.models.llm.qwen.moe_common import pack_gated_expert_weights
        sd = {
            "mlp.experts.0.gate_proj.weight": torch.randn(8, 8),
            "mlp.experts.0.up_proj.weight": torch.randn(8, 8),
            "mlp.experts.1.gate_proj.weight": torch.randn(8, 8),
            "mlp.experts.1.up_proj.weight": torch.randn(8, 8),
            "mlp.experts.1.down_proj.weight": torch.randn(8, 8),
        }
        with pytest.raises(KeyError, match="Inconsistent expert weights"):
            pack_gated_expert_weights(dict(sd))


class TestPanguQuantModule:
    """Tests for amct_pytorch/common/models/llm/pangu/pangu_7b/quant_module.py"""

    def test_rotate_half(self):
        from amct_pytorch.common.models.llm.pangu.pangu_7b.quant_module import rotate_half
        x = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])
        result = rotate_half(x)
        expected = torch.tensor([[[-3.0, -4.0, 1.0, 2.0]]])
        assert torch.allclose(result, expected)

    def test_apply_rotary_pos_emb(self):
        from amct_pytorch.common.models.llm.pangu.pangu_7b.quant_module import apply_rotary_pos_emb
        q = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])
        k = torch.tensor([[[5.0, 6.0, 7.0, 8.0]]])
        cos = torch.tensor([[[0.5, 0.5, 0.5, 0.5]]])
        sin = torch.tensor([[[0.5, 0.5, 0.5, 0.5]]])
        q_embed, k_embed = apply_rotary_pos_emb(q, k, cos, sin)
        assert q_embed.shape == (1, 1, 1, 4)
        assert k_embed.shape == (1, 1, 1, 4)

    def test_ensure_transformers_compat(self):
        from amct_pytorch.common.models.llm.pangu.pangu_7b.quant_module import _ensure_transformers_compat
        _ensure_transformers_compat()
        import transformers.utils as transformers_utils
        from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
        assert hasattr(transformers_utils, "LossKwargs")
        assert hasattr(transformers_utils, "auto_docstring")
        assert hasattr(transformers_utils, "can_return_tuple")
        assert "default" in ROPE_INIT_FUNCTIONS

    def test_quant_pangu_attn_construct_with_attn_linear(self, monkeypatch):
        from amct_pytorch.common.models.llm.pangu.pangu_7b.quant_module import QuantPanguAttn
        monkeypatch.setattr(
            "amct_pytorch.common.models.llm.pangu.pangu_7b.quant_module.QuantLinear",
            lambda args, module, w_bits=None, name=None: module,
        )
        monkeypatch.setattr(
            "amct_pytorch.common.models.llm.pangu.pangu_7b.quant_module.QuantizedMatmul",
            lambda *a, **kw: nn.Identity(),
        )
        monkeypatch.setattr(
            "amct_pytorch.common.models.llm.pangu.pangu_7b.quant_module.ActivationQuantizer",
            lambda args, bits: nn.Identity(),
        )
        monkeypatch.setattr(
            "amct_pytorch.common.models.llm.pangu.pangu_7b.quant_module.build_algorithms_by_target",
            lambda args, target, ctx: None,
        )
        args = _make_mock_args(
            quant_target=["attn-linear"], q_bits=8, k_bits=8, p_bits=8, v_bits=8,
            bit_policy=BitPolicy({
                "attn-linear": {
                    "q_proj": {"w_bits": 8, "a_bits": 8},
                    "k_proj": {"w_bits": 8, "a_bits": 8},
                    "v_proj": {"w_bits": 8, "a_bits": 8},
                    "o_proj": {"w_bits": 8, "a_bits": 8},
                },
                "attn-cache": {"q": 8, "k": 8, "p": 8, "v": 8},
            }),
        )
        attn_module = MagicMock()
        attn_module.config = SimpleNamespace(hidden_size=64)
        attn_module.layer_idx = 0
        attn_module.head_dim = 16
        attn_module.num_heads = 4
        attn_module.num_key_value_heads = 2
        attn_module.num_key_value_groups = 2
        attn_module.scaling = 0.25
        attn_module.attention_dropout = 0.0
        attn_module.is_causal = True
        attn_module.hidden_size = 64
        attn_module.q_proj = nn.Linear(4, 4)
        attn_module.k_proj = nn.Linear(4, 4)
        attn_module.v_proj = nn.Linear(4, 4)
        attn_module.o_proj = nn.Linear(4, 4)
        qa = QuantPanguAttn(args, attn_module)
        assert qa.enable_attn_linear is True
        assert qa.inp_afq is not None

    def test_quant_pangu_attn_construct_no_attn_linear(self, monkeypatch):
        from amct_pytorch.common.models.llm.pangu.pangu_7b.quant_module import QuantPanguAttn
        monkeypatch.setattr(
            "amct_pytorch.common.models.llm.pangu.pangu_7b.quant_module.QuantLinear",
            lambda args, module, w_bits=None, name=None: module,
        )
        monkeypatch.setattr(
            "amct_pytorch.common.models.llm.pangu.pangu_7b.quant_module.QuantizedMatmul",
            lambda *a, **kw: nn.Identity(),
        )
        monkeypatch.setattr(
            "amct_pytorch.common.models.llm.pangu.pangu_7b.quant_module.ActivationQuantizer",
            lambda args, bits: nn.Identity(),
        )
        monkeypatch.setattr(
            "amct_pytorch.common.models.llm.pangu.pangu_7b.quant_module.build_algorithms_by_target",
            lambda args, target, ctx: None,
        )
        args = _make_mock_args(
            quant_target=["mlp"], q_bits=8, k_bits=8, p_bits=8, v_bits=8,
            bit_policy=BitPolicy({
                "attn-linear": {
                    "q_proj": {"w_bits": 8, "a_bits": 8},
                    "k_proj": {"w_bits": 8, "a_bits": 8},
                    "v_proj": {"w_bits": 8, "a_bits": 8},
                    "o_proj": {"w_bits": 8, "a_bits": 8},
                },
                "attn-cache": {"q": 8, "k": 8, "p": 8, "v": 8},
            }),
        )
        attn_module = MagicMock()
        attn_module.config = SimpleNamespace(hidden_size=64)
        attn_module.layer_idx = 0
        attn_module.head_dim = 16
        attn_module.num_heads = 4
        attn_module.num_key_value_heads = 2
        attn_module.num_key_value_groups = 2
        attn_module.scaling = 0.25
        attn_module.attention_dropout = 0.0
        attn_module.is_causal = True
        attn_module.hidden_size = 64
        attn_module.q_proj = nn.Linear(4, 4)
        attn_module.k_proj = nn.Linear(4, 4)
        attn_module.v_proj = nn.Linear(4, 4)
        attn_module.o_proj = nn.Linear(4, 4)
        qa = QuantPanguAttn(args, attn_module)
        assert qa.enable_attn_linear is False
        assert isinstance(qa.q_proj, nn.Module)
