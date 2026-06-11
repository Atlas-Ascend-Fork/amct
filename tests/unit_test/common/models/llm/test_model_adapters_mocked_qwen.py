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
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn
from accelerate import init_empty_weights
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen3.modeling_qwen3 import Qwen3Config, Qwen3DecoderLayer

from amct_pytorch.quantization.bit_policy import BitPolicy
from tests.unit_test.common.models.llm.common.test_base import (
    _make_block_with_attn_and_mlp,
)

FROM_PRETRAINED = 'from_pretrained'
QUANT_TARGET_MLP = 'mlp'
W_BITS = 'w_bits'
A_BITS = 'a_bits'

DENSE = 'dense'


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


def _make_mock_args(model_path="/tmp/fake_model", quant_target=(QUANT_TARGET_MLP,), **extra):
    base = {
        "model": model_path,
        "quant_target": list(quant_target),
        "device": "cpu",
        "data_dir": "/tmp/fake_data",
        "output_dir": "/tmp/fake_output",
        "quant_dtype": "int8",
        W_BITS: 8,
        A_BITS: 8,
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
        type("FakeAC", (), {FROM_PRETRAINED: staticmethod(lambda *a, **kw: cfg)})(),
    )
    monkeypatch.setattr(
        "amct_pytorch.common.models.llm.common.base.AutoTokenizer",
        type("FakeAT", (), {FROM_PRETRAINED: staticmethod(lambda *a, **kw: fake_tokenizer)})(),
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
            FROM_PRETRAINED: staticmethod(lambda *a, **kw: empty_model),
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
        args = _make_mock_args(quant_target=[QUANT_TARGET_MLP])
        model = Qwen3(args)
        assert model.num_layers == 2
        assert model.textconfig is Qwen3Config
        assert model.cls is Qwen3DecoderLayer
        assert model.quant_target == [QUANT_TARGET_MLP]
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
        model = Qwen3(_make_mock_args(quant_target=[QUANT_TARGET_MLP]))
        block = nn.Module()
        block.mlp = nn.Linear(4, 4)
        units = list(model.iter_ptq_units(0, block))
        assert len(units) == 1
        assert units[0].kind == QUANT_TARGET_MLP

    def test_qwen3_export_block_deploy_returns_tensors(self, mock_hf_model):
        from amct_pytorch.common.models.llm.common.deploy_export import (
            export_block_deploy,
        )
        from amct_pytorch.common.models.llm.qwen.qwen3.qwen3 import Qwen3
        model = Qwen3(_make_mock_args(quant_target=[QUANT_TARGET_MLP]))

        def block(layer_idx):
            return nn.Linear(4, 4)

        model.block = block

        def load_selected_layer_ptq_params(layer_idx, block, strict):
            return None

        model.load_selected_layer_ptq_params = load_selected_layer_ptq_params
        tensors, routes = export_block_deploy(model, 0, [])
        assert isinstance(tensors, dict)
        assert isinstance(routes, dict)

    def test_qwen3_init_cls_property(self, mock_hf_model):
        from transformers.models.qwen3.modeling_qwen3 import (
            Qwen3DecoderLayer as qwen3_decoder_layer,
        )

        from amct_pytorch.common.models.llm.qwen.qwen3.qwen3 import Qwen3
        model = Qwen3(_make_mock_args())
        assert model.cls is qwen3_decoder_layer

    def test_qwen3_num_layers_property(self, mock_hf_model):
        from amct_pytorch.common.models.llm.qwen.qwen3.qwen3 import Qwen3
        model = Qwen3(_make_mock_args())
        assert model.num_layers == 2

    def test_qwen3_textconfig_property(self, mock_hf_model):
        from transformers.models.qwen3.modeling_qwen3 import Qwen3Config as qwen3_config

        from amct_pytorch.common.models.llm.qwen.qwen3.qwen3 import Qwen3
        model = Qwen3(_make_mock_args())
        assert model.textconfig is qwen3_config


class TestQwen3MoeMocked:
    def test_qwen3_moe_init_mocked(self, monkeypatch):
        from transformers.models.qwen3_moe.modeling_qwen3_moe import (
            Qwen3MoeConfig,
            Qwen3MoeDecoderLayer,
        )

        from amct_pytorch.common.models.llm.qwen.qwen3.qwen3_moe import Qwen3Moe
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
            type("FakeAC", (), {FROM_PRETRAINED: staticmethod(lambda *a, **kw: cfg)})(),
        )
        fake_tokenizer = MagicMock()
        monkeypatch.setattr(
            "amct_pytorch.common.models.llm.common.base.AutoTokenizer",
            type("FakeAT", (), {FROM_PRETRAINED: staticmethod(lambda *a, **kw: fake_tokenizer)})(),
        )
        monkeypatch.setattr(
            "amct_pytorch.common.models.llm.common.base.init_empty_weights",
            lambda: MagicMock(__enter__=lambda s: None, __exit__=lambda *a: None),
        )
        from transformers import AutoModelForCausalLM as auto_model_for_causal_lm
        with init_empty_weights():
            empty_model = auto_model_for_causal_lm.from_config(cfg, trust_remote_code=True, torch_dtype=torch.bfloat16)
        monkeypatch.setattr(
            "amct_pytorch.common.models.llm.common.base.AutoModelForCausalLM",
            type("FakeAMFCLM", (), {
                FROM_PRETRAINED: staticmethod(lambda *a, **kw: empty_model),
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
        from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeConfig

        from amct_pytorch.common.models.llm.qwen.qwen3.qwen3_moe import Qwen3Moe
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
            type("FakeAC", (), {FROM_PRETRAINED: staticmethod(lambda *a, **kw: cfg)})(),
        )
        fake_tokenizer = MagicMock()
        monkeypatch.setattr(
            "amct_pytorch.common.models.llm.common.base.AutoTokenizer",
            type("FakeAT", (), {FROM_PRETRAINED: staticmethod(lambda *a, **kw: fake_tokenizer)})(),
        )
        monkeypatch.setattr(
            "amct_pytorch.common.models.llm.common.base.init_empty_weights",
            lambda: MagicMock(__enter__=lambda s: None, __exit__=lambda *a: None),
        )
        from transformers import AutoModelForCausalLM as auto_model_for_causal_lm
        with init_empty_weights():
            empty_model = auto_model_for_causal_lm.from_config(cfg, trust_remote_code=True, torch_dtype=torch.bfloat16)
        monkeypatch.setattr(
            "amct_pytorch.common.models.llm.common.base.AutoModelForCausalLM",
            type("FakeAMFCLM", (), {
                FROM_PRETRAINED: staticmethod(lambda *a, **kw: empty_model),
                "from_config": staticmethod(lambda *a, **kw: empty_model),
            })(),
        )
        monkeypatch.setattr(
            "amct_pytorch.common.models.llm.common.base.get_weight_mappings",
            lambda path: {},
        )
        args = _make_mock_args(quant_target=[QUANT_TARGET_MLP], moe_mlp_param_dir="")
        with pytest.raises(ValueError, match=f"does not support quant_target='{QUANT_TARGET_MLP}'"):
            Qwen3Moe(args)

    def test_qwen3_moe_num_experts(self, monkeypatch):
        from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeConfig

        from amct_pytorch.common.models.llm.qwen.qwen3.qwen3_moe import Qwen3Moe
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
        from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeConfig

        from amct_pytorch.common.models.llm.qwen.qwen3.qwen3_moe import Qwen3Moe
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
        from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeConfig

        from amct_pytorch.common.models.llm.qwen.qwen3.qwen3_moe import Qwen3Moe
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
        from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeConfig

        from amct_pytorch.common.models.llm.qwen.qwen3.qwen3_moe import Qwen3Moe
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
        from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeConfig

        from amct_pytorch.common.models.llm.qwen.qwen3.qwen3_moe import Qwen3Moe
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
        from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeConfig

        from amct_pytorch.common.models.llm.qwen.qwen3.qwen3_moe import Qwen3Moe
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
        from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeConfig

        from amct_pytorch.common.models.llm.qwen.qwen3.qwen3_moe import Qwen3Moe
        from amct_pytorch.quantization.modules.quant_linear import QuantLinear
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
        from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeConfig

        from amct_pytorch.common.models.llm.qwen.qwen3.qwen3_moe import Qwen3Moe
        from amct_pytorch.quantization.modules.quant_linear import QuantLinear
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
        from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeConfig

        from amct_pytorch.common.models.llm.qwen.qwen3.qwen3_moe import Qwen3Moe
        from amct_pytorch.quantization.modules.quant_linear import QuantLinear
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
        from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeConfig

        from amct_pytorch.common.models.llm.qwen.qwen3.qwen3_moe import Qwen3Moe
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
        assert not bindings

    def test_qwen3_moe_float_model_delegates(self, monkeypatch):
        from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeConfig

        from amct_pytorch.common.models.llm.qwen.qwen3.qwen3_moe import Qwen3Moe
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
        from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeConfig

        from amct_pytorch.common.models.llm.qwen.qwen3.qwen3_moe import Qwen3Moe
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
        from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeConfig

        from amct_pytorch.common.models.llm.qwen.qwen3.qwen3_moe import Qwen3Moe
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
        from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeConfig

        from amct_pytorch.common.models.llm.qwen.qwen3.qwen3_moe import Qwen3Moe
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
        from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeConfig

        from amct_pytorch.common.models.llm.qwen.qwen3.qwen3_moe import Qwen3Moe
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
        fake_block_result = mock_result

        def _mock_do_block_forward(self, layer_idx, samples, hook_name=None,
                                   use_quant_block=False, enable_quant=False):
            return fake_block_result

        monkeypatch.setattr(
            "amct_pytorch.common.models.llm.common.base.BaseModel.do_block_forward",
            _mock_do_block_forward,
        )
        result = model.do_block_forward(0, [torch.randn(2, 4, 8)])
        assert result is mock_result

    def test_qwen3_moe_do_head_forward_delegates(self, monkeypatch):
        from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeConfig

        from amct_pytorch.common.models.llm.qwen.qwen3.qwen3_moe import Qwen3Moe
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
        from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeConfig

        from amct_pytorch.common.models.llm.qwen.qwen3.qwen3_moe import Qwen3Moe
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
        from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeConfig

        from amct_pytorch.common.models.llm.qwen.qwen3.qwen3_moe import Qwen3Moe
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
        from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeConfig

        from amct_pytorch.common.models.llm.qwen.qwen3.qwen3_moe import Qwen3Moe
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
        from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeConfig

        from amct_pytorch.common.models.llm.common.ptq_units import make_ptq_unit
        from amct_pytorch.common.models.llm.qwen.qwen3.qwen3_moe import Qwen3Moe
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
        unit = make_ptq_unit("moe", QUANT_TARGET_MLP, layer_idx=0, module=nn.Linear(4, 4))
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
        type("FakeAC", (), {FROM_PRETRAINED: staticmethod(lambda *a, **kw: cfg)})(),
    )
    monkeypatch.setattr(
        "amct_pytorch.common.models.llm.common.base.AutoTokenizer",
        type("FakeAT", (), {FROM_PRETRAINED: staticmethod(lambda *a, **kw: fake_tokenizer)})(),
    )
    monkeypatch.setattr(
        "amct_pytorch.common.models.llm.common.base.init_empty_weights",
        lambda: MagicMock(__enter__=lambda s: None, __exit__=lambda *a: None),
    )
    from transformers import AutoModelForCausalLM as auto_model_for_causal_lm
    with init_empty_weights():
        empty_model = auto_model_for_causal_lm.from_config(cfg, trust_remote_code=True, torch_dtype=torch.bfloat16)
    monkeypatch.setattr(
        "amct_pytorch.common.models.llm.common.base.AutoModelForCausalLM",
        type("FakeAMFCLM", (), {
            FROM_PRETRAINED: staticmethod(lambda *a, **kw: empty_model),
            "from_config": staticmethod(lambda *a, **kw: empty_model),
        })(),
    )
    monkeypatch.setattr(
        "amct_pytorch.common.models.llm.common.base.get_weight_mappings",
        lambda path: {},
    )


class TestQwen3NextMocked:
    def test_qwen3_next_init(self, monkeypatch):
        from transformers.models.qwen3_next.modeling_qwen3_next import (
            Qwen3NextConfig,
            Qwen3NextDecoderLayer,
        )

        from amct_pytorch.common.models.llm.qwen.qwen3_next.qwen3_next import Qwen3Next
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
        from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextConfig

        from amct_pytorch.common.models.llm.qwen.qwen3_next.qwen3_next import Qwen3Next
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
        args = _make_mock_args(quant_target=[QUANT_TARGET_MLP])
        with pytest.raises(ValueError, match=f"does not support quant_target='{QUANT_TARGET_MLP}'"):
            Qwen3Next(args)

    def test_qwen3_next_export_block_deploy(self, monkeypatch):
        from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextConfig

        from amct_pytorch.common.models.llm.common.deploy_export import (
            export_block_deploy,
        )
        from amct_pytorch.common.models.llm.qwen.qwen3_next.qwen3_next import Qwen3Next
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

        def block(layer_idx):
            return nn.Linear(4, 4)

        model.block = block

        def load_selected_layer_ptq_params(layer_idx, block, strict):
            return None

        model.load_selected_layer_ptq_params = load_selected_layer_ptq_params
        tensors, routes = export_block_deploy(model, 0, [])
        assert isinstance(tensors, dict)

    def test_qwen3_next_get_layer_weight_prefix(self, monkeypatch):
        from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextConfig

        from amct_pytorch.common.models.llm.qwen.qwen3_next.qwen3_next import Qwen3Next
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
        from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextConfig

        from amct_pytorch.common.models.llm.qwen.qwen3_next.qwen3_next import Qwen3Next
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
        from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextConfig

        from amct_pytorch.common.models.llm.qwen.qwen3_next.qwen3_next import Qwen3Next
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
        from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextConfig

        from amct_pytorch.common.models.llm.qwen.qwen3_next.qwen3_next import Qwen3Next
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
        from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextConfig

        from amct_pytorch.common.models.llm.qwen.qwen3_next.qwen3_next import Qwen3Next
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
        from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextConfig

        from amct_pytorch.common.models.llm.qwen.qwen3_next.qwen3_next import Qwen3Next
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
        from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextConfig

        from amct_pytorch.common.models.llm.qwen.qwen3_next.quant_module import (
            QuantQwen3NextLinearAttn,
        )
        from amct_pytorch.common.models.llm.qwen.qwen3_next.qwen3_next import Qwen3Next
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
        from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextConfig

        from amct_pytorch.common.models.llm.qwen.qwen3_next.quant_module import (
            QuantQwen3NextAttn,
        )
        from amct_pytorch.common.models.llm.qwen.qwen3_next.qwen3_next import Qwen3Next
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
        from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextConfig

        from amct_pytorch.common.models.llm.qwen.qwen3_next.qwen3_next import Qwen3Next
        from amct_pytorch.quantization.modules.quant_linear import QuantLinear
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
        from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextConfig

        from amct_pytorch.common.models.llm.qwen.qwen3_next.qwen3_next import Qwen3Next
        from amct_pytorch.quantization.modules.quant_linear import QuantLinear
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
        from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextConfig

        from amct_pytorch.common.models.llm.qwen.qwen3_next.qwen3_next import Qwen3Next
        from amct_pytorch.quantization.modules.quant_linear import QuantLinear
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
        from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextConfig

        from amct_pytorch.common.models.llm.qwen.qwen3_next.qwen3_next import Qwen3Next
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
        assert not bindings

    def test_qwen3_next_float_model_delegates(self, monkeypatch):
        from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextConfig

        from amct_pytorch.common.models.llm.qwen.qwen3_next.qwen3_next import Qwen3Next
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
        from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextConfig

        from amct_pytorch.common.models.llm.qwen.qwen3_next.qwen3_next import Qwen3Next
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
        from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextConfig

        from amct_pytorch.common.models.llm.qwen.qwen3_next.qwen3_next import Qwen3Next
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
        from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextConfig

        from amct_pytorch.common.models.llm.qwen.qwen3_next.qwen3_next import Qwen3Next
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
        fake_block_result = mock_result

        def _mock_do_block_forward(self, layer_idx, samples, hook_name=None,
                                   use_quant_block=False, enable_quant=False):
            return fake_block_result

        monkeypatch.setattr(
            "amct_pytorch.common.models.llm.common.base.BaseModel.do_block_forward",
            _mock_do_block_forward,
        )
        result = model.do_block_forward(0, [torch.randn(2, 4, 8)])
        assert result is mock_result

    def test_qwen3_next_do_head_forward_delegates(self, monkeypatch):
        from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextConfig

        from amct_pytorch.common.models.llm.qwen.qwen3_next.qwen3_next import Qwen3Next
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
        from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextConfig

        from amct_pytorch.common.models.llm.qwen.qwen3_next.qwen3_next import Qwen3Next
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
        from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextConfig

        from amct_pytorch.common.models.llm.common.ptq_units import make_ptq_unit
        from amct_pytorch.common.models.llm.qwen.qwen3_next.qwen3_next import Qwen3Next
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
        unit = make_ptq_unit("moe", QUANT_TARGET_MLP, layer_idx=0, module=nn.Linear(4, 4))
        fake_data = torch.randn(4, 4)
        monkeypatch.setattr(
            "amct_pytorch.common.models.llm.common.base.load_ptq_inps",
            lambda data_dir, kind, layer_idx: fake_data,
        )
        result = model.load_unit_inputs("/tmp/fake", unit)
        assert result is fake_data

    def test_qwen3_next_iter_ptq_units(self, monkeypatch):
        from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextConfig

        from amct_pytorch.common.models.llm.qwen.qwen3_next.qwen3_next import Qwen3Next
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


