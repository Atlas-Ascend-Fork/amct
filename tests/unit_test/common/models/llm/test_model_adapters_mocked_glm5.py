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
"""Tests for GLM5_1_EX adapter with mocked HuggingFace model loading."""
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn
from accelerate import init_empty_weights
from transformers import AutoModelForCausalLM
from transformers.models.glm_moe_dsa.configuration_glm_moe_dsa import GlmMoeDsaConfig
from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import GlmMoeDsaDecoderLayer

FROM_PRETRAINED = 'from_pretrained'
QUANT_TARGET_MLP = 'mlp'
QUANT_TARGET_MOE = 'moe'
W_BITS = 'w_bits'
A_BITS = 'a_bits'


def _make_glm5_config(num_layers=2):
    return GlmMoeDsaConfig(
        num_hidden_layers=num_layers,
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=64,
        num_attention_heads=4,
        num_key_value_heads=4,
        kv_lora_rank=32,
        q_lora_rank=64,
        qk_rope_head_dim=16,
        qk_nope_head_dim=16,
        v_head_dim=16,
        rms_norm_eps=1e-6,
        vocab_size=100,
        max_position_embeddings=512,
        rope_parameters={"rope_theta": 10000.0, "rope_type": "default"},
        tie_word_embeddings=False,
        mlp_layer_types=["dense"] * num_layers,
        n_routed_experts=8,
        n_shared_experts=1,
        num_experts_per_tok=2,
        index_topk=512,
        index_n_heads=8,
        index_head_dim=32,
        indexer_types=["full"] * num_layers,
    )


def _make_mock_args(model_path="/tmp/fake_glm5", quant_target=(QUANT_TARGET_MLP,), **extra):
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


def _mock_base_deps(monkeypatch, cfg):
    """Mock BaseModel HF dependencies with a given GlmMoeDsaConfig."""
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


class TestGLM5_1Mocked:
    """Tests for GLM5_1_EX adapter with mocked HF dependencies."""

    def test_glm5_1_init(self, monkeypatch):
        """Verify GLM5_1_EX.__init__ sets num_layers, cls, and quant_target."""
        from amct_pytorch.common.models.llm.glm.glm5_1_1.glm5_1_1 import GLM5_1_EX
        cfg = _make_glm5_config(num_layers=2)
        _mock_base_deps(monkeypatch, cfg)
        args = _make_mock_args(quant_target=[QUANT_TARGET_MOE])
        model = GLM5_1_EX(args)
        assert model.num_layers == 2
        assert model.cls is GlmMoeDsaDecoderLayer
        assert model.quant_target == [QUANT_TARGET_MOE]

    def test_glm5_1_reject_mlp_quant(self, monkeypatch):
        """Verify parse_quant_mode raises ValueError for quant_target=['mlp']."""
        from amct_pytorch.common.models.llm.glm.glm5_1_1.glm5_1_1 import GLM5_1_EX
        cfg = _make_glm5_config(num_layers=2)
        _mock_base_deps(monkeypatch, cfg)
        args = _make_mock_args(quant_target=[QUANT_TARGET_MLP])
        with pytest.raises(ValueError, match="does not support quant_target='mlp'"):
            GLM5_1_EX(args)

    def test_glm5_1_accept_moe_quant(self, monkeypatch):
        """Verify quant_target=['moe'] works without error."""
        from amct_pytorch.common.models.llm.glm.glm5_1_1.glm5_1_1 import GLM5_1_EX
        cfg = _make_glm5_config(num_layers=2)
        _mock_base_deps(monkeypatch, cfg)
        args = _make_mock_args(quant_target=[QUANT_TARGET_MOE])
        model = GLM5_1_EX(args)
        assert model.quant_target == [QUANT_TARGET_MOE]

    def test_glm5_1_get_layer_weight_prefix(self, monkeypatch):
        """Verify get_layer_weight_prefix returns model.layers.{idx}."""
        from amct_pytorch.common.models.llm.glm.glm5_1_1.glm5_1_1 import GLM5_1_EX
        cfg = _make_glm5_config(num_layers=2)
        _mock_base_deps(monkeypatch, cfg)
        args = _make_mock_args(quant_target=[QUANT_TARGET_MOE])
        model = GLM5_1_EX(args)
        assert model.get_layer_weight_prefix(0) == "model.layers.0."
        assert model.get_layer_weight_prefix(1) == "model.layers.1."

    def test_glm5_1_load_layer_weight_calls_pack_experts(self, monkeypatch):
        """Verify load_layer_weight calls pack_gated_expert_weights with expert_prefix."""
        from amct_pytorch.common.models.llm.glm.glm5_1_1.glm5_1_1 import GLM5_1_EX
        cfg = _make_glm5_config(num_layers=2)
        _mock_base_deps(monkeypatch, cfg)
        args = _make_mock_args(quant_target=[QUANT_TARGET_MOE])
        model = GLM5_1_EX(args)

        fake_sd = {"some.weight": torch.randn(4, 4)}
        monkeypatch.setattr(
            "amct_pytorch.common.models.llm.common.base.BaseModel.load_layer_weight",
            lambda self, prefix: fake_sd,
        )

        captured = {}
        def _mock_pack(state_dict, expert_prefix="mlp.experts"):
            captured["expert_prefix"] = expert_prefix
            captured["state_dict"] = state_dict
            return dict(state_dict)

        monkeypatch.setattr(
            "amct_pytorch.common.models.llm.glm.glm5_1_1.glm5_1_1.pack_gated_expert_weights",
            _mock_pack,
        )
        result = model.load_layer_weight("model.layers.0.")
        assert captured["expert_prefix"] == "mlp.experts"
        assert captured["state_dict"] is fake_sd
        assert result is not fake_sd

    def test_glm5_1_register(self, monkeypatch):
        """Verify the adapter is registered in MODEL_REGISTRY with correct name/family."""
        from amct_pytorch.common.models import MODEL_REGISTRY
        from amct_pytorch.common.models.llm.glm.glm5_1_1.glm5_1_1 import GLM5_1_EX

        cfg = _make_glm5_config(num_layers=2)
        _mock_base_deps(monkeypatch, cfg)
        args = _make_mock_args(quant_target=[QUANT_TARGET_MOE])
        model = GLM5_1_EX(args)

        assert "glm_5_1_ex" in MODEL_REGISTRY
        item = MODEL_REGISTRY.get_item("glm_5_1_ex")
        assert item.name == "glm_5_1_ex"
        assert item.target is GLM5_1_EX
        assert item.metadata.get("family") == "glm"
        assert item.metadata.get("task") == "llm"

    def test_glm5_1_not_subclass_of_deepseek_v32(self):
        """Verify GLM5_1_EX is NOT a subclass of DeepseekV32."""
        from amct_pytorch.common.models.llm.deepseek.deepseek_v3_2.deepseekv3_2 import DeepseekV32
        from amct_pytorch.common.models.llm.glm.glm5_1_1.glm5_1_1 import GLM5_1_EX

        assert not issubclass(GLM5_1_EX, DeepseekV32)

    def test_glm5_1_build_quant_block_moe_packed(self, monkeypatch):
        """Verify build_quant_block replaces packed experts with QuantGatedExperts."""
        from amct_pytorch.common.models.llm.glm.glm5_1_1.glm5_1_1 import GLM5_1_EX
        cfg = _make_glm5_config(num_layers=2)
        _mock_base_deps(monkeypatch, cfg)
        args = _make_mock_args(quant_target=[QUANT_TARGET_MOE])
        model = GLM5_1_EX(args)

        dl = nn.Module()
        packed = type("PE", (), {"gate_up_proj": None, "down_proj": None})()
        moe = nn.Module()
        moe.experts = packed
        dl.mlp = moe
        monkeypatch.setattr(model, "block", lambda idx: dl)
        monkeypatch.setattr(
            "amct_pytorch.common.models.llm.glm.glm5_1_1.glm5_1_1.QuantGatedExperts",
            lambda args_in, experts: setattr(experts, "_q", True) or experts,
        )
        result = model.build_quant_block(0)
        assert result is dl
        assert hasattr(result.mlp.experts, "_q")

    def test_glm5_1_build_quant_block_moe_shared_experts(self, monkeypatch):
        """Verify build_quant_block creates QuantGatedMLP for shared experts."""
        from amct_pytorch.common.models.llm.glm.glm5_1_1.glm5_1_1 import GLM5_1_EX
        cfg = _make_glm5_config(num_layers=2)
        _mock_base_deps(monkeypatch, cfg)
        args = _make_mock_args(quant_target=[QUANT_TARGET_MOE])
        model = GLM5_1_EX(args)

        dl = nn.Module()
        packed = type("PE", (), {"gate_up_proj": None, "down_proj": None})()
        moe = nn.Module()
        moe.experts = packed
        shared = nn.Linear(4, 4)
        moe.shared_experts = shared
        dl.mlp = moe
        monkeypatch.setattr(model, "block", lambda idx: dl)
        monkeypatch.setattr(
            "amct_pytorch.common.models.llm.glm.glm5_1_1.glm5_1_1.QuantGatedExperts",
            lambda args_in, experts: experts,
        )
        captured = {}
        def _mock_quant_gated_mlp(args_in, module, group=None):
            captured["group"] = group
            captured["module"] = module
            return module
        monkeypatch.setattr(
            "amct_pytorch.common.models.llm.glm.glm5_1_1.glm5_1_1.QuantGatedMLP",
            _mock_quant_gated_mlp,
        )
        model.build_quant_block(0)
        assert captured.get("group") == "moe.shared"
        assert captured.get("module") is shared

    def test_glm5_1_float_model_delegates(self, monkeypatch):
        """Verify float_model delegates to BaseModel."""
        from amct_pytorch.common.models.llm.glm.glm5_1_1.glm5_1_1 import GLM5_1_EX
        cfg = _make_glm5_config(num_layers=2)
        _mock_base_deps(monkeypatch, cfg)
        model = GLM5_1_EX(_make_mock_args(quant_target=[QUANT_TARGET_MOE]))
        mock_result = MagicMock()
        monkeypatch.setattr(
            "amct_pytorch.common.models.llm.common.base.BaseModel.float_model",
            lambda self: mock_result,
        )
        assert model.float_model() is mock_result

    def test_glm5_1_block_delegates(self, monkeypatch):
        """Verify block delegates to BaseModel."""
        from amct_pytorch.common.models.llm.glm.glm5_1_1.glm5_1_1 import GLM5_1_EX
        cfg = _make_glm5_config(num_layers=2)
        _mock_base_deps(monkeypatch, cfg)
        model = GLM5_1_EX(_make_mock_args(quant_target=[QUANT_TARGET_MOE]))
        mock_result = MagicMock()
        monkeypatch.setattr(
            "amct_pytorch.common.models.llm.common.base.BaseModel.block",
            lambda self, idx: mock_result,
        )
        assert model.block(0) is mock_result

    def test_glm5_1_do_embedding_forward_delegates(self, monkeypatch):
        """Verify do_embedding_forward delegates to BaseModel."""
        from amct_pytorch.common.models.llm.glm.glm5_1_1.glm5_1_1 import GLM5_1_EX
        cfg = _make_glm5_config(num_layers=2)
        _mock_base_deps(monkeypatch, cfg)
        model = GLM5_1_EX(_make_mock_args(quant_target=[QUANT_TARGET_MOE]))
        mock_result = [torch.randn(2, 4, 8)]
        monkeypatch.setattr(
            "amct_pytorch.common.models.llm.common.base.BaseModel.do_embedding_forward",
            lambda self, samples, dtype=torch.bfloat16, hook_name=None: mock_result,
        )
        result = model.do_embedding_forward([torch.randn(2, 4)])
        assert result is mock_result

    def test_glm5_1_do_block_forward_delegates(self, monkeypatch):
        """Verify do_block_forward delegates to BaseModel."""
        from amct_pytorch.common.models.llm.glm.glm5_1_1.glm5_1_1 import GLM5_1_EX
        cfg = _make_glm5_config(num_layers=2)
        _mock_base_deps(monkeypatch, cfg)
        model = GLM5_1_EX(_make_mock_args(quant_target=[QUANT_TARGET_MOE]))
        mock_result = [torch.randn(2, 4, 8)]

        def _mock_do_block_forward(self, layer_idx, samples, hook_name=None,
                                   use_quant_block=False, enable_quant=False):
            return mock_result

        monkeypatch.setattr(
            "amct_pytorch.common.models.llm.common.base.BaseModel.do_block_forward",
            _mock_do_block_forward,
        )
        result = model.do_block_forward(0, [torch.randn(2, 4, 8)])
        assert result is mock_result

    def test_glm5_1_do_head_forward_delegates(self, monkeypatch):
        """Verify do_head_forward delegates to BaseModel."""
        from amct_pytorch.common.models.llm.glm.glm5_1_1.glm5_1_1 import GLM5_1_EX
        cfg = _make_glm5_config(num_layers=2)
        _mock_base_deps(monkeypatch, cfg)
        model = GLM5_1_EX(_make_mock_args(quant_target=[QUANT_TARGET_MOE]))
        mock_result = [torch.randn(2, 4, 100)]
        monkeypatch.setattr(
            "amct_pytorch.common.models.llm.common.base.BaseModel.do_head_forward",
            lambda self, inps: mock_result,
        )
        result = model.do_head_forward([torch.randn(2, 4, 8)])
        assert result is mock_result


if __name__ == "__main__":
    pytest.main([__file__])
