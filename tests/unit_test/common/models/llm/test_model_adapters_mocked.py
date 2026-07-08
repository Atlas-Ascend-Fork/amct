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


def _mock_base_deps_wrapper(monkeypatch, cfg):
    """Mock BaseModel HF dependencies with a wrapper config that has text_config."""
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
        empty_model = auto_model_for_causal_lm.from_config(
            cfg.text_config, trust_remote_code=True, torch_dtype=torch.bfloat16)
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


class TestQwen35Mocked:
    def test_qwen3_5_init(self, monkeypatch):
        from transformers.models.qwen3_5.modeling_qwen3_5 import (
            Qwen3_5Config,
            Qwen3_5DecoderLayer,
        )

        from amct_pytorch.common.models.llm.qwen.qwen3_5.qwen3_5 import Qwen3_5
        cfg = Qwen3_5Config(text_config={
            "num_hidden_layers": 2, "hidden_size": 64, "intermediate_size": 128,
            "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16,
            "rms_norm_eps": 1e-6, "vocab_size": 100, "max_position_embeddings": 512,
            "rope_theta": 10000.0, "tie_word_embeddings": False,
        })
        _mock_base_deps_wrapper(monkeypatch, cfg)
        args = _make_mock_args(quant_target=[QUANT_TARGET_MLP])
        model = Qwen3_5(args)
        assert model.num_layers == 2
        assert model.cls is Qwen3_5DecoderLayer
        assert model.quant_target == [QUANT_TARGET_MLP]

    def test_qwen3_5_parse_quant_mode_rejects_moe(self, monkeypatch):
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Config

        from amct_pytorch.common.models.llm.qwen.qwen3_5.qwen3_5 import Qwen3_5
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
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Config

        from amct_pytorch.common.models.llm.qwen.qwen3_5.qwen3_5 import Qwen3_5
        cfg = Qwen3_5Config(text_config={
            "num_hidden_layers": 2, "hidden_size": 64, "intermediate_size": 128,
            "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16,
            "rms_norm_eps": 1e-6, "vocab_size": 100, "max_position_embeddings": 512,
            "rope_theta": 10000.0, "tie_word_embeddings": False,
        })
        _mock_base_deps_wrapper(monkeypatch, cfg)
        model = Qwen3_5(_make_mock_args(quant_target=[QUANT_TARGET_MLP]))
        assert model.get_layer_weight_prefix(0) == "model.language_model.layers.0."

    def test_qwen3_5_embed_base_prefix(self, monkeypatch):
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Config

        from amct_pytorch.common.models.llm.qwen.qwen3_5.qwen3_5 import Qwen3_5
        cfg = Qwen3_5Config(text_config={
            "num_hidden_layers": 2, "hidden_size": 64, "intermediate_size": 128,
            "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16,
            "rms_norm_eps": 1e-6, "vocab_size": 100, "max_position_embeddings": 512,
            "rope_theta": 10000.0, "tie_word_embeddings": False,
        })
        _mock_base_deps_wrapper(monkeypatch, cfg)
        model = Qwen3_5(_make_mock_args(quant_target=[QUANT_TARGET_MLP]))
        assert model._embed_base_prefix() == "model.language_model."

    def test_qwen3_5_get_block_attention_mask(self, monkeypatch):
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Config

        from amct_pytorch.common.models.llm.qwen.qwen3_5.qwen3_5 import Qwen3_5
        cfg = Qwen3_5Config(text_config={
            "num_hidden_layers": 2, "hidden_size": 64, "intermediate_size": 128,
            "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16,
            "rms_norm_eps": 1e-6, "vocab_size": 100, "max_position_embeddings": 512,
            "rope_theta": 10000.0, "tie_word_embeddings": False,
        })
        _mock_base_deps_wrapper(monkeypatch, cfg)
        model = Qwen3_5(_make_mock_args(quant_target=[QUANT_TARGET_MLP]))
        assert model.get_block_attention_mask(nn.Module()) is None

    def test_qwen3_5_build_quant_block_mlp(self, monkeypatch):
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Config

        from amct_pytorch.common.models.llm.qwen.qwen3_5.qwen3_5 import Qwen3_5
        cfg = Qwen3_5Config(text_config={
            "num_hidden_layers": 2, "hidden_size": 64, "intermediate_size": 128,
            "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16,
            "rms_norm_eps": 1e-6, "vocab_size": 100, "max_position_embeddings": 512,
            "rope_theta": 10000.0, "tie_word_embeddings": False,
        })
        _mock_base_deps_wrapper(monkeypatch, cfg)
        model = Qwen3_5(_make_mock_args(quant_target=[QUANT_TARGET_MLP]))
        mock_block = nn.Linear(4, 4)
        monkeypatch.setattr(model, "block", lambda idx: mock_block)
        monkeypatch.setattr(model, "apply_quant_moe_mlp", lambda dl: setattr(dl, "_q", True))
        result = model.build_quant_block(0)
        assert hasattr(result, "_q")

    def test_qwen3_5_build_quant_block_attn_linear(self, monkeypatch):
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Config

        from amct_pytorch.common.models.llm.qwen.qwen3_5.qwen3_5 import Qwen3_5
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
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Config

        from amct_pytorch.common.models.llm.qwen.qwen3_5.quant_module import (
            QuantQwen35LinearAttn,
        )
        from amct_pytorch.common.models.llm.qwen.qwen3_5.qwen3_5 import Qwen3_5
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
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Config

        from amct_pytorch.common.models.llm.qwen.qwen3_5.quant_module import (
            QuantQwen35Attn,
        )
        from amct_pytorch.common.models.llm.qwen.qwen3_5.qwen3_5 import Qwen3_5
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
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Config

        from amct_pytorch.common.models.llm.qwen.qwen3_5.quant_module import (
            QuantQwen35MLP,
        )
        from amct_pytorch.common.models.llm.qwen.qwen3_5.qwen3_5 import Qwen3_5
        cfg = Qwen3_5Config(text_config={
            "num_hidden_layers": 2, "hidden_size": 64, "intermediate_size": 128,
            "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16,
            "rms_norm_eps": 1e-6, "vocab_size": 100, "max_position_embeddings": 512,
            "rope_theta": 10000.0, "tie_word_embeddings": False,
        })
        _mock_base_deps_wrapper(monkeypatch, cfg)
        model = Qwen3_5(_make_mock_args(quant_target=[QUANT_TARGET_MLP]))
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
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Config

        from amct_pytorch.common.models.llm.qwen.qwen3_5.qwen3_5 import Qwen3_5
        cfg = Qwen3_5Config(text_config={
            "num_hidden_layers": 2, "hidden_size": 64, "intermediate_size": 128,
            "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16,
            "rms_norm_eps": 1e-6, "vocab_size": 100, "max_position_embeddings": 512,
            "rope_theta": 10000.0, "tie_word_embeddings": False,
        })
        _mock_base_deps_wrapper(monkeypatch, cfg)
        model = Qwen3_5(_make_mock_args(quant_target=[QUANT_TARGET_MLP]))
        block = _make_block_with_attn_and_mlp()
        units = list(model.iter_ptq_units(0, block))
        assert len(units) == 1
        assert units[0].kind == QUANT_TARGET_MLP

    def test_qwen3_5_iter_deploy_bindings(self, monkeypatch):
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Config

        from amct_pytorch.common.models.llm.qwen.qwen3_5.qwen3_5 import Qwen3_5
        from amct_pytorch.quantization.modules.quant_linear import QuantLinear
        cfg = Qwen3_5Config(text_config={
            "num_hidden_layers": 2, "hidden_size": 64, "intermediate_size": 128,
            "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16,
            "rms_norm_eps": 1e-6, "vocab_size": 100, "max_position_embeddings": 512,
            "rope_theta": 10000.0, "tie_word_embeddings": False,
        })
        _mock_base_deps_wrapper(monkeypatch, cfg)
        model = Qwen3_5(_make_mock_args(quant_target=[QUANT_TARGET_MLP]))
        ql = QuantLinear.__new__(QuantLinear)
        ql._parameters = {}
        ql._modules = {}
        ql._buffers = {}
        block = nn.Module()
        block.linear = ql
        bindings = list(model.iter_deploy_bindings(0, block))
        assert len(bindings) == 1

    def test_qwen3_5_float_model_delegates(self, monkeypatch):
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Config

        from amct_pytorch.common.models.llm.qwen.qwen3_5.qwen3_5 import Qwen3_5
        cfg = Qwen3_5Config(text_config={
            "num_hidden_layers": 2, "hidden_size": 64, "intermediate_size": 128,
            "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16,
            "rms_norm_eps": 1e-6, "vocab_size": 100, "max_position_embeddings": 512,
            "rope_theta": 10000.0, "tie_word_embeddings": False,
        })
        _mock_base_deps_wrapper(monkeypatch, cfg)
        model = Qwen3_5(_make_mock_args(quant_target=[QUANT_TARGET_MLP]))
        mock_result = MagicMock()
        monkeypatch.setattr(
            "amct_pytorch.common.models.llm.common.base.BaseModel.float_model",
            lambda self: mock_result,
        )
        assert model.float_model() is mock_result

    def test_qwen3_5_load_layer_weight_delegates(self, monkeypatch):
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Config

        from amct_pytorch.common.models.llm.qwen.qwen3_5.qwen3_5 import Qwen3_5
        cfg = Qwen3_5Config(text_config={
            "num_hidden_layers": 2, "hidden_size": 64, "intermediate_size": 128,
            "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16,
            "rms_norm_eps": 1e-6, "vocab_size": 100, "max_position_embeddings": 512,
            "rope_theta": 10000.0, "tie_word_embeddings": False,
        })
        _mock_base_deps_wrapper(monkeypatch, cfg)
        model = Qwen3_5(_make_mock_args(quant_target=[QUANT_TARGET_MLP]))
        mock_result = {"some.weight": torch.randn(4, 4)}
        monkeypatch.setattr(
            "amct_pytorch.common.models.llm.common.base.BaseModel.load_layer_weight",
            lambda self, prefix: mock_result,
        )
        assert model.load_layer_weight("model.language_model.layers.0.") is mock_result

    def test_qwen3_5_load_embed_state_dict_delegates(self, monkeypatch):
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Config

        from amct_pytorch.common.models.llm.qwen.qwen3_5.qwen3_5 import Qwen3_5
        cfg = Qwen3_5Config(text_config={
            "num_hidden_layers": 2, "hidden_size": 64, "intermediate_size": 128,
            "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16,
            "rms_norm_eps": 1e-6, "vocab_size": 100, "max_position_embeddings": 512,
            "rope_theta": 10000.0, "tie_word_embeddings": False,
        })
        _mock_base_deps_wrapper(monkeypatch, cfg)
        model = Qwen3_5(_make_mock_args(quant_target=[QUANT_TARGET_MLP]))
        monkeypatch.setattr(
            "amct_pytorch.common.models.llm.common.base.BaseModel.load_embed_state_dict",
            lambda self: None,
        )
        model.load_embed_state_dict()

    def test_qwen3_5_block_delegates(self, monkeypatch):
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Config

        from amct_pytorch.common.models.llm.qwen.qwen3_5.qwen3_5 import Qwen3_5
        cfg = Qwen3_5Config(text_config={
            "num_hidden_layers": 2, "hidden_size": 64, "intermediate_size": 128,
            "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16,
            "rms_norm_eps": 1e-6, "vocab_size": 100, "max_position_embeddings": 512,
            "rope_theta": 10000.0, "tie_word_embeddings": False,
        })
        _mock_base_deps_wrapper(monkeypatch, cfg)
        model = Qwen3_5(_make_mock_args(quant_target=[QUANT_TARGET_MLP]))
        mock_result = MagicMock()
        monkeypatch.setattr(
            "amct_pytorch.common.models.llm.common.base.BaseModel.block",
            lambda self, idx: mock_result,
        )
        assert model.block(0) is mock_result

    def test_qwen3_5_do_embedding_forward_delegates(self, monkeypatch):
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Config

        from amct_pytorch.common.models.llm.qwen.qwen3_5.qwen3_5 import Qwen3_5
        cfg = Qwen3_5Config(text_config={
            "num_hidden_layers": 2, "hidden_size": 64, "intermediate_size": 128,
            "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16,
            "rms_norm_eps": 1e-6, "vocab_size": 100, "max_position_embeddings": 512,
            "rope_theta": 10000.0, "tie_word_embeddings": False,
        })
        _mock_base_deps_wrapper(monkeypatch, cfg)
        model = Qwen3_5(_make_mock_args(quant_target=[QUANT_TARGET_MLP]))
        mock_result = [torch.randn(2, 4, 8)]
        monkeypatch.setattr(
            "amct_pytorch.common.models.llm.common.base.BaseModel.do_embedding_forward",
            lambda self, samples, dtype=torch.bfloat16, hook_name=None: mock_result,
        )
        result = model.do_embedding_forward([torch.randn(2, 4)])
        assert result is mock_result

    def test_qwen3_5_do_block_forward_delegates(self, monkeypatch):
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Config

        from amct_pytorch.common.models.llm.qwen.qwen3_5.qwen3_5 import Qwen3_5
        cfg = Qwen3_5Config(text_config={
            "num_hidden_layers": 2, "hidden_size": 64, "intermediate_size": 128,
            "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16,
            "rms_norm_eps": 1e-6, "vocab_size": 100, "max_position_embeddings": 512,
            "rope_theta": 10000.0, "tie_word_embeddings": False,
        })
        _mock_base_deps_wrapper(monkeypatch, cfg)
        model = Qwen3_5(_make_mock_args(quant_target=[QUANT_TARGET_MLP]))
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

    def test_qwen3_5_do_head_forward_delegates(self, monkeypatch):
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Config

        from amct_pytorch.common.models.llm.qwen.qwen3_5.qwen3_5 import Qwen3_5
        cfg = Qwen3_5Config(text_config={
            "num_hidden_layers": 2, "hidden_size": 64, "intermediate_size": 128,
            "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16,
            "rms_norm_eps": 1e-6, "vocab_size": 100, "max_position_embeddings": 512,
            "rope_theta": 10000.0, "tie_word_embeddings": False,
        })
        _mock_base_deps_wrapper(monkeypatch, cfg)
        model = Qwen3_5(_make_mock_args(quant_target=[QUANT_TARGET_MLP]))
        mock_result = [torch.randn(2, 4, 100)]
        monkeypatch.setattr(
            "amct_pytorch.common.models.llm.common.base.BaseModel.do_head_forward",
            lambda self, inps: mock_result,
        )
        result = model.do_head_forward([torch.randn(2, 4, 8)])
        assert result is mock_result

    def test_qwen3_5_load_unit_inputs_delegates(self, monkeypatch):
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Config

        from amct_pytorch.common.models.llm.common.ptq_units import make_ptq_unit
        from amct_pytorch.common.models.llm.qwen.qwen3_5.qwen3_5 import Qwen3_5
        cfg = Qwen3_5Config(text_config={
            "num_hidden_layers": 2, "hidden_size": 64, "intermediate_size": 128,
            "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16,
            "rms_norm_eps": 1e-6, "vocab_size": 100, "max_position_embeddings": 512,
            "rope_theta": 10000.0, "tie_word_embeddings": False,
        })
        _mock_base_deps_wrapper(monkeypatch, cfg)
        model = Qwen3_5(_make_mock_args(quant_target=[QUANT_TARGET_MLP]))
        unit = make_ptq_unit(QUANT_TARGET_MLP, QUANT_TARGET_MLP, layer_idx=0, module=nn.Linear(4, 4))
        fake_data = torch.randn(4, 4)
        monkeypatch.setattr(
            "amct_pytorch.common.models.llm.common.base.load_ptq_inps",
            lambda data_dir, kind, layer_idx: fake_data,
        )
        result = model.load_unit_inputs("/tmp/fake", unit)
        assert result is fake_data


class TestQwen35MoeMocked:
    def test_qwen3_5_moe_init(self, monkeypatch):
        from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
            Qwen3_5MoeConfig,
            Qwen3_5MoeDecoderLayer,
        )

        from amct_pytorch.common.models.llm.qwen.qwen3_5.qwen3_5_moe import Qwen3_5Moe
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
        from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
            Qwen3_5MoeConfig,
        )

        from amct_pytorch.common.models.llm.qwen.qwen3_5.qwen3_5_moe import Qwen3_5Moe
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
        args = _make_mock_args(quant_target=[QUANT_TARGET_MLP], moe_mlp_param_dir="")
        with pytest.raises(ValueError, match=f"does not support quant_target='{QUANT_TARGET_MLP}'"):
            Qwen3_5Moe(args)

    def test_qwen3_5_moe_get_layer_weight_prefix(self, monkeypatch):
        from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
            Qwen3_5MoeConfig,
        )

        from amct_pytorch.common.models.llm.qwen.qwen3_5.qwen3_5_moe import Qwen3_5Moe
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
        from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
            Qwen3_5MoeConfig,
        )

        from amct_pytorch.common.models.llm.qwen.qwen3_5.qwen3_5_moe import Qwen3_5Moe
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
        from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
            Qwen3_5MoeConfig,
        )

        from amct_pytorch.common.models.llm.qwen.qwen3_5.qwen3_5_moe import Qwen3_5Moe
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
        from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
            Qwen3_5MoeConfig,
        )

        from amct_pytorch.common.models.llm.qwen.qwen3_5.qwen3_5_moe import Qwen3_5Moe
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
        from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
            Qwen3_5MoeConfig,
        )

        from amct_pytorch.common.models.llm.qwen.qwen3_5.qwen3_5_moe import Qwen3_5Moe
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
        from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
            Qwen3_5MoeConfig,
        )

        from amct_pytorch.common.models.llm.qwen.qwen3_5.qwen3_5_moe import Qwen3_5Moe
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
        from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
            Qwen3_5MoeConfig,
        )

        from amct_pytorch.common.models.llm.qwen.qwen3_5.qwen3_5_moe import Qwen3_5Moe
        from amct_pytorch.quantization.modules.quant_linear import QuantLinear
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
        assert bindings[0][0] == "model.language_model.layers.0.mlp.experts.1.gate_proj.weight"

    def test_qwen3_5_moe_iter_ptq_units(self, monkeypatch):
        from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
            Qwen3_5MoeConfig,
        )

        from amct_pytorch.common.models.llm.qwen.qwen3_5.qwen3_5_moe import Qwen3_5Moe
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
        from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
            Qwen3_5MoeConfig,
        )

        from amct_pytorch.common.models.llm.qwen.qwen3_5.qwen3_5_moe import Qwen3_5Moe
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
        from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
            Qwen3_5MoeConfig,
        )

        from amct_pytorch.common.models.llm.qwen.qwen3_5.qwen3_5_moe import Qwen3_5Moe
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
        from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
            Qwen3_5MoeConfig,
        )

        from amct_pytorch.common.models.llm.qwen.qwen3_5.qwen3_5_moe import Qwen3_5Moe
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
        from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
            Qwen3_5MoeConfig,
        )

        from amct_pytorch.common.models.llm.qwen.qwen3_5.qwen3_5_moe import Qwen3_5Moe
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
        from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
            Qwen3_5MoeConfig,
        )

        from amct_pytorch.common.models.llm.qwen.qwen3_5.qwen3_5_moe import Qwen3_5Moe
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
        from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
            Qwen3_5MoeConfig,
        )

        from amct_pytorch.common.models.llm.qwen.qwen3_5.qwen3_5_moe import Qwen3_5Moe
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

    def test_qwen3_5_moe_do_head_forward_delegates(self, monkeypatch):
        from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
            Qwen3_5MoeConfig,
        )

        from amct_pytorch.common.models.llm.qwen.qwen3_5.qwen3_5_moe import Qwen3_5Moe
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
        from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
            Qwen3_5MoeConfig,
        )

        from amct_pytorch.common.models.llm.common.ptq_units import make_ptq_unit
        from amct_pytorch.common.models.llm.qwen.qwen3_5.qwen3_5_moe import Qwen3_5Moe
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
        unit = make_ptq_unit("moe", QUANT_TARGET_MLP, layer_idx=0, module=nn.Linear(4, 4))
        fake_data = torch.randn(4, 4)
        monkeypatch.setattr(
            "amct_pytorch.common.models.llm.common.base.load_ptq_inps",
            lambda data_dir, kind, layer_idx: fake_data,
        )
        result = model.load_unit_inputs("/tmp/fake", unit)
        assert result is fake_data


class TestLongcatNextMocked:
    def test_longcat_next_init(self, monkeypatch):
        from transformers.models.longcat_flash.configuration_longcat_flash import (
            LongcatFlashConfig,
        )

        from amct_pytorch.common.models.llm.longcat.longcat_next.longcat_next import (
            LongcatNext,
        )

        def _mock_empty_weights_model(self):
            return AutoModelForCausalLM.from_config(
                self.config, trust_remote_code=True, torch_dtype=torch.bfloat16)

        monkeypatch.setattr(
            LongcatNext,
            "empty_weights_model",
            _mock_empty_weights_model,
        )
        cfg = LongcatFlashConfig(
            num_hidden_layers=2, num_layers=2, hidden_size=64, intermediate_size=128,
            num_attention_heads=4, num_key_value_heads=2, head_dim=16,
            rms_norm_eps=1e-6, vocab_size=100, max_position_embeddings=512,
            tie_word_embeddings=False,
        )
        _mock_base_deps(monkeypatch, cfg)
        args = _make_mock_args(quant_target=[QUANT_TARGET_MLP])
        model = LongcatNext(args)
        assert model.num_layers == 2

    def test_longcat_next_get_layer_weight_prefix(self, monkeypatch):
        from transformers.models.longcat_flash.configuration_longcat_flash import (
            LongcatFlashConfig,
        )

        from amct_pytorch.common.models.llm.longcat.longcat_next.longcat_next import (
            LongcatNext,
        )

        def _mock_empty_weights_model(self):
            return AutoModelForCausalLM.from_config(
                self.config, trust_remote_code=True, torch_dtype=torch.bfloat16)

        monkeypatch.setattr(
            LongcatNext,
            "empty_weights_model",
            _mock_empty_weights_model,
        )
        cfg = LongcatFlashConfig(
            num_hidden_layers=2, num_layers=2, hidden_size=64, intermediate_size=128,
            num_attention_heads=4, num_key_value_heads=2, head_dim=16,
            rms_norm_eps=1e-6, vocab_size=100, max_position_embeddings=512,
            tie_word_embeddings=False,
        )
        _mock_base_deps(monkeypatch, cfg)
        args = _make_mock_args(quant_target=[QUANT_TARGET_MLP])
        model = LongcatNext(args)
        assert model.get_layer_weight_prefix(0) == "model.layers.0."


class TestLongcatLiteMocked:
    def test_longcat_lite_init(self, monkeypatch):
        from transformers.models.longcat_flash.configuration_longcat_flash import (
            LongcatFlashConfig,
        )

        from amct_pytorch.common.models.llm.longcat.longcat_lite.longcat_lite import (
            LongcatLite,
        )
        cfg = LongcatFlashConfig(
            num_hidden_layers=2, num_layers=2, hidden_size=64, intermediate_size=128,
            num_attention_heads=4, num_key_value_heads=2, head_dim=16,
            rms_norm_eps=1e-6, vocab_size=100, max_position_embeddings=512,
            tie_word_embeddings=False,
        )
        _mock_base_deps(monkeypatch, cfg)
        args = _make_mock_args(quant_target=[QUANT_TARGET_MLP])
        model = LongcatLite(args)
        assert model.num_layers == 2

    def test_longcat_lite_iter_ptq_units_mlp(self, monkeypatch):
        from transformers.models.longcat_flash.configuration_longcat_flash import (
            LongcatFlashConfig,
        )

        from amct_pytorch.common.models.llm.longcat.longcat_lite.longcat_lite import (
            LongcatLite,
        )
        cfg = LongcatFlashConfig(
            num_hidden_layers=2, num_layers=2, hidden_size=64, intermediate_size=128,
            num_attention_heads=4, num_key_value_heads=2, head_dim=16,
            rms_norm_eps=1e-6, vocab_size=100, max_position_embeddings=512,
            tie_word_embeddings=False,
        )
        _mock_base_deps(monkeypatch, cfg)
        args = _make_mock_args(quant_target=[QUANT_TARGET_MLP])
        model = LongcatLite(args)
        block = nn.Module()
        block.mlps = nn.ModuleList([nn.Linear(4, 4)])
        units = list(model.iter_ptq_units(0, block))
        assert len(units) == 1
        assert units[0].kind == QUANT_TARGET_MLP


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
        from amct_pytorch.common.models.llm.qwen.moe_common import (
            pack_gated_expert_weights,
        )
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
        from amct_pytorch.common.models.llm.qwen.moe_common import (
            pack_gated_expert_weights,
        )
        sd = {"other.weight": torch.randn(4, 4)}
        result = pack_gated_expert_weights(dict(sd))
        assert result == sd

    def test_pack_gated_expert_weights_inconsistent(self):
        from amct_pytorch.common.models.llm.qwen.moe_common import (
            pack_gated_expert_weights,
        )
        sd = {
            "mlp.experts.0.gate_proj.weight": torch.randn(8, 8),
            "mlp.experts.0.up_proj.weight": torch.randn(8, 8),
            "mlp.experts.1.gate_proj.weight": torch.randn(8, 8),
            "mlp.experts.1.up_proj.weight": torch.randn(8, 8),
            "mlp.experts.1.down_proj.weight": torch.randn(8, 8),
        }
        with pytest.raises(KeyError, match="Inconsistent expert weights"):
            pack_gated_expert_weights(dict(sd))

