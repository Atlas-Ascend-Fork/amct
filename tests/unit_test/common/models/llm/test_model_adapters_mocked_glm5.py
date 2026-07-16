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
"""Tests for GLM5_2 adapter with mocked HuggingFace model loading."""
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn
from accelerate import init_empty_weights
from transformers import AutoModelForCausalLM
from transformers.models.glm_moe_dsa.configuration_glm_moe_dsa import GlmMoeDsaConfig
from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import GlmMoeDsaDecoderLayer
from amct_pytorch.common.models.llm.glm.glm5_2.quant_module import QuantGlmIndexer, QuantGlmMoeDsaAttention
from amct_pytorch.quantization.modules.quant_linear import QuantLinear
from amct_pytorch.common.models.llm.common.quant_apply import PlainLinear
from amct_pytorch.common.models.llm.glm.glm5_2.glm5_2 import GLM5_2
from amct_pytorch.common.models import MODEL_REGISTRY
from amct_pytorch.quantization.dtypes import register_dtype
from amct_pytorch.quantization.bit_policy import BitPolicy

FROM_PRETRAINED = 'from_pretrained'
QUANT_TARGET_MLP = 'mlp'
QUANT_TARGET_MOE = 'moe'
W_BITS = 'w_bits'
A_BITS = 'a_bits'
GATE_PROJ = 'gate_proj'
DOWN_PROJ = 'down_proj'
UP_PROJ = 'up_proj'


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
    # GLM5_2 imports get_weight_mappings at module level via "from X import Y".
    # Patching the source module alone does NOT affect the local reference in
    # glm5_2.py's namespace — we must patch the adapter module directly.
    monkeypatch.setattr(
        "amct_pytorch.common.models.llm.glm.glm5_2.glm5_2.get_weight_mappings",
        lambda path: {},
    )


class TestGLM5_2Mocked:
    """Tests for GLM5_2 adapter with mocked HF dependencies."""

    @staticmethod
    def test_glm5_2_init(monkeypatch):
        """Verify GLM5_2.__init__ sets num_layers, cls, and quant_target."""
        cfg = _make_glm5_config(num_layers=2)
        _mock_base_deps(monkeypatch, cfg)
        args = _make_mock_args(quant_target=[QUANT_TARGET_MOE])
        model = GLM5_2(args)
        assert model.num_layers == 2
        assert model.cls is GlmMoeDsaDecoderLayer
        assert model.quant_target == [QUANT_TARGET_MOE]

    @staticmethod
    def test_glm5_2_reject_mlp_quant(monkeypatch):
        """Verify parse_quant_mode raises ValueError for quant_target=['mlp']."""
        cfg = _make_glm5_config(num_layers=2)
        _mock_base_deps(monkeypatch, cfg)
        args = _make_mock_args(quant_target=[QUANT_TARGET_MLP])
        with pytest.raises(ValueError, match="does not support quant_target='mlp'"):
            GLM5_2(args)

    @staticmethod
    def test_glm5_2_accept_moe_quant(monkeypatch):
        """Verify quant_target=['moe'] works without error."""
        cfg = _make_glm5_config(num_layers=2)
        _mock_base_deps(monkeypatch, cfg)
        args = _make_mock_args(quant_target=[QUANT_TARGET_MOE])
        model = GLM5_2(args)
        assert model.quant_target == [QUANT_TARGET_MOE]

    @staticmethod
    def test_glm5_2_get_layer_weight_prefix(monkeypatch):
        """Verify get_layer_weight_prefix returns model.layers.{idx}."""
        cfg = _make_glm5_config(num_layers=2)
        _mock_base_deps(monkeypatch, cfg)
        args = _make_mock_args(quant_target=[QUANT_TARGET_MOE])
        model = GLM5_2(args)
        assert model.get_layer_weight_prefix(0) == "model.layers.0."
        assert model.get_layer_weight_prefix(1) == "model.layers.1."

    @staticmethod
    def test_glm5_2_load_layer_weight_calls_pack_experts(monkeypatch):
        """Verify load_layer_weight calls pack_gated_expert_weights with expert_prefix."""
        cfg = _make_glm5_config(num_layers=2)
        _mock_base_deps(monkeypatch, cfg)
        args = _make_mock_args(quant_target=[QUANT_TARGET_MOE])
        model = GLM5_2(args)

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
            "amct_pytorch.common.models.llm.glm.glm5_2.glm5_2.pack_gated_expert_weights",
            _mock_pack,
        )
        result = model.load_layer_weight("model.layers.0.")
        assert captured["expert_prefix"] == "mlp.experts"
        assert captured["state_dict"] is fake_sd
        assert result is not fake_sd

    @staticmethod
    def test_glm5_2_register(monkeypatch):
        """Verify the adapter is registered in MODEL_REGISTRY with correct name/family."""

        cfg = _make_glm5_config(num_layers=2)
        _mock_base_deps(monkeypatch, cfg)
        args = _make_mock_args(quant_target=[QUANT_TARGET_MOE])
        model = GLM5_2(args)

        assert "glm5_2" in MODEL_REGISTRY
        item = MODEL_REGISTRY.get_item("glm5_2")
        assert item.name == "glm5_2"
        assert item.target is GLM5_2
        assert item.metadata.get("family") == "glm"
        assert item.metadata.get("task") == "llm"

    @staticmethod
    def test_glm5_2_build_quant_block_moe_packed(monkeypatch):
        """Verify build_quant_block replaces packed experts with QuantGatedExperts."""
        cfg = _make_glm5_config(num_layers=2)
        _mock_base_deps(monkeypatch, cfg)
        args = _make_mock_args(quant_target=[QUANT_TARGET_MOE])
        model = GLM5_2(args)

        dl = nn.Module()
        packed = type("PE", (), {"gate_up_proj": None, "down_proj": None})()
        moe = nn.Module()
        moe.experts = packed
        dl.mlp = moe
        monkeypatch.setattr(model, "block", lambda idx: dl)
        monkeypatch.setattr(
            "amct_pytorch.common.models.llm.glm.glm5_2.glm5_2.QuantGatedExperts",
            lambda args_in, experts: setattr(experts, "_q", True) or experts,
        )
        result = model.build_quant_block(0)
        assert result is dl
        assert hasattr(result.mlp.experts, "_q")

    @staticmethod
    def test_glm5_2_build_quant_block_moe_shared_experts(monkeypatch):
        """Verify build_quant_block creates QuantGatedMLP for shared experts."""
        cfg = _make_glm5_config(num_layers=2)
        _mock_base_deps(monkeypatch, cfg)
        args = _make_mock_args(quant_target=[QUANT_TARGET_MOE])
        model = GLM5_2(args)

        dl = nn.Module()
        packed = type("PE", (), {"gate_up_proj": None, "down_proj": None})()
        moe = nn.Module()
        moe.experts = packed
        shared = nn.Linear(4, 4)
        moe.shared_experts = shared
        dl.mlp = moe
        monkeypatch.setattr(model, "block", lambda idx: dl)
        monkeypatch.setattr(
            "amct_pytorch.common.models.llm.glm.glm5_2.glm5_2.QuantGatedExperts",
            lambda args_in, experts: experts,
        )
        captured = {}
        def _mock_quant_gated_mlp(args_in, module, group=None):
            captured["group"] = group
            captured["module"] = module
            return module
        monkeypatch.setattr(
            "amct_pytorch.common.models.llm.glm.glm5_2.glm5_2.QuantGatedMLP",
            _mock_quant_gated_mlp,
        )
        model.build_quant_block(0)
        assert captured.get("group") == "moe.shared"
        assert captured.get("module") is shared

    @staticmethod
    def test_glm5_2_float_model_delegates(monkeypatch):
        """Verify float_model delegates to BaseModel."""
        cfg = _make_glm5_config(num_layers=2)
        _mock_base_deps(monkeypatch, cfg)
        model = GLM5_2(_make_mock_args(quant_target=[QUANT_TARGET_MOE]))
        mock_result = MagicMock()
        monkeypatch.setattr(
            "amct_pytorch.common.models.llm.common.base.BaseModel.float_model",
            lambda self: mock_result,
        )
        assert model.float_model() is mock_result

    @staticmethod
    def test_glm5_2_do_head_forward_delegates(monkeypatch):
        """Verify do_head_forward delegates to BaseModel."""
        cfg = _make_glm5_config(num_layers=2)
        _mock_base_deps(monkeypatch, cfg)
        model = GLM5_2(_make_mock_args(quant_target=[QUANT_TARGET_MOE]))
        mock_result = [torch.randn(2, 4, 100)]
        monkeypatch.setattr(
            "amct_pytorch.common.models.llm.common.base.BaseModel.do_head_forward",
            lambda self, inps: mock_result,
        )
        result = model.do_head_forward([torch.randn(2, 4, 8)])
        assert result is mock_result



# ---------------------------------------------------------------------------
# Helpers for quant_module tests
# ---------------------------------------------------------------------------

QUANT_TARGET_ATTN_LINEAR = 'attn-linear'


def _register_dtype_once():
    """Register dtype entries (idempotent)."""
    register_dtype()


def _make_quant_args(quant_target=("attn-linear",), quant_dtype="int"):
    """Construct args for QuantGlmMoeDsaAttention.__init__."""
    cfg = {
        "w_bits": 8,
        "a_bits": 8,
        "attn-linear": {
            "q_a_proj": {"w_bits": 16, "a_bits": 16},
            "q_b_proj": {"w_bits": 8, "a_bits": 8},
            "kv_a_proj_with_mqa": {"w_bits": 16, "a_bits": 16},
            "o_proj": {"w_bits": 8, "a_bits": 8},
            "wq_b": {"w_bits": 8, "a_bits": 8},
            "wk": {"w_bits": 16, "a_bits": 16},
        },
    }
    policy = BitPolicy(cfg)
    return SimpleNamespace(
        quant_target=list(quant_target),
        bit_policy=policy,
        algos=[],
        device="cpu",
        w_bits=8,
        a_bits=8,
        quant_dtype=quant_dtype,
    )


def _make_mock_attn_module(with_indexer=True):
    """Construct a mock GlmMoeDsaAttention with all required attributes."""
    attn = SimpleNamespace(
        config=SimpleNamespace(hidden_size=64),
        num_heads=4,
        q_lora_rank=64,
        kv_lora_rank=32,
        qk_head_dim=32,
        qk_nope_head_dim=16,
        qk_rope_head_dim=16,
        v_head_dim=16,
        scaling=1.0,
        skip_topk=False,
        num_key_value_groups=1,
        q_a_layernorm=nn.LayerNorm(64),
        kv_a_layernorm=nn.LayerNorm(32),
        q_a_proj=nn.Linear(64, 64, bias=False),
        q_b_proj=nn.Linear(64, 4 * 32, bias=False),
        kv_a_proj_with_mqa=nn.Linear(64, 32 + 16, bias=False),
        kv_b_proj=nn.Linear(32, 4 * 32, bias=False),
        o_proj=nn.Linear(4 * 16, 64, bias=False),
        indexer=None,
    )
    if with_indexer:
        indexer = SimpleNamespace(
            n_heads=4,
            head_dim=32,
            qk_rope_head_dim=16,
            index_topk=4,
            q_lora_rank=64,
            softmax_scale=32 ** -0.5,
            wq_b=nn.Linear(64, 4 * 32, bias=False),
            wk=nn.Linear(64, 32, bias=False),
            k_norm=nn.LayerNorm(32),
            weights_proj=nn.Linear(64, 4, bias=False),
        )
        attn.indexer = indexer
    return attn


def _make_real_quant_linear(in_f=4, out_f=4, name="test_ql"):
    """Create a real QuantLinear for isinstance checks in iter_deploy_bindings."""
    _register_dtype_once()
    args = SimpleNamespace(
        quant_dtype="int", algos=[], w_bits=8, w_size=(out_f, in_f),
    )
    linear = nn.Linear(in_f, out_f, bias=False)
    return QuantLinear(args, linear, w_bits=8, name=name)


# ---------------------------------------------------------------------------
# A. GLM5_2 block-level methods (A1-A10)
# ---------------------------------------------------------------------------

class TestGLM5_2BlockMethods:
    """Tests for GLM5_2 block-level methods."""

    @staticmethod
    def _make_model(monkeypatch, quant_target=None):
        if quant_target is None:
            quant_target = [QUANT_TARGET_MOE]
        cfg = _make_glm5_config(num_layers=2)
        _mock_base_deps(monkeypatch, cfg)
        args = _make_mock_args(quant_target=quant_target)
        return GLM5_2(args)

    def test_block_loads_layer(self, monkeypatch):
        """block() loads weights via load_state_dict(strict=True)."""
        model = self._make_model(monkeypatch)
        fake_layer = MagicMock()
        monkeypatch.setattr(model, "cls", lambda *a, **kw: fake_layer)
        fake_sd = {"some.weight": torch.randn(4, 4)}
        monkeypatch.setattr(model, "load_layer_weight", lambda prefix: fake_sd)
        result = model.block(0)
        assert result is fake_layer
        fake_layer.load_state_dict.assert_called_once_with(fake_sd, strict=True)
        fake_layer.eval.assert_called_once()
        fake_layer.eval.return_value.bfloat16.assert_called_once()

    def test_build_block_for_forward_captures_topk(self, monkeypatch):
        """_build_block_for_forward hooks forward to capture topk_indices."""
        model = self._make_model(monkeypatch)
        fake_topk = torch.tensor([[1, 2], [3, 4]])
        fake_hidden = torch.randn(2, 4, 8)
        fake_block = MagicMock()
        fake_block.forward = lambda *a, **kw: (fake_hidden, fake_topk)
        monkeypatch.setattr(
            "amct_pytorch.common.models.llm.common.base.BaseModel._build_block_for_forward",
            lambda self, layer_idx, use_quant_block=False: fake_block,
        )
        result = getattr(model, "_build_block_for_forward")(0)
        result.forward()
        assert getattr(model, "_dsa_topk_indices") is fake_topk

    def test_get_block_forward_kwargs_injects_topk(self, monkeypatch):
        """get_block_forward_kwargs injects prev_topk_indices when available."""
        model = self._make_model(monkeypatch)
        fake_topk = torch.tensor([[1, 2], [3, 4]])
        setattr(model, "_dsa_topk_indices", fake_topk)
        kwargs = model.get_block_forward_kwargs()
        assert "prev_topk_indices" in kwargs
        assert kwargs["prev_topk_indices"] is fake_topk

    def test_get_block_forward_kwargs_no_topk(self, monkeypatch):
        """get_block_forward_kwargs omits prev_topk_indices when None."""
        model = self._make_model(monkeypatch)
        setattr(model, "_dsa_topk_indices", None)
        kwargs = model.get_block_forward_kwargs()
        assert "prev_topk_indices" not in kwargs

    def test_build_quant_block_attn_linear(self, monkeypatch):
        """build_quant_block applies attn quantization for attn-linear target."""
        model = self._make_model(monkeypatch, quant_target=["attn-linear"])
        dl = nn.Module()
        dl.mlp = nn.Linear(4, 4)
        monkeypatch.setattr(model, "block", lambda idx: dl)
        called = {"attn": False}

        def _fake_apply(args, layer, cls):
            called["attn"] = True

        monkeypatch.setattr(
            "amct_pytorch.common.models.llm.glm.glm5_2.glm5_2.apply_quant_to_attn",
            _fake_apply,
        )
        model.build_quant_block(0)
        assert called["attn"] is True

    def test_build_quant_block_mlp_none_raises(self, monkeypatch):
        """build_quant_block raises ValueError when mlp is missing."""
        model = self._make_model(monkeypatch, quant_target=["moe"])
        dl = nn.Module()
        monkeypatch.setattr(model, "block", lambda idx: dl)
        with pytest.raises(ValueError, match="MLP not found"):
            model.build_quant_block(0)

    def test_build_quant_block_dense_mlp(self, monkeypatch):
        """build_quant_block uses QuantGatedMLP for dense MLP (no experts)."""
        model = self._make_model(monkeypatch, quant_target=["moe"])
        dl = nn.Module()
        dl.mlp = nn.Linear(4, 4)
        monkeypatch.setattr(model, "block", lambda idx: dl)
        captured = {}

        def _mock_qgmlp(args, module, group=None):
            captured["group"] = group
            captured["module"] = module
            return module

        monkeypatch.setattr(
            "amct_pytorch.common.models.llm.glm.glm5_2.glm5_2.QuantGatedMLP",
            _mock_qgmlp,
        )
        model.build_quant_block(0)
        assert captured.get("group") == "mlp"

    def test_iter_deploy_bindings_moe_path(self, monkeypatch):
        """iter_deploy_bindings translates MoE expert paths correctly."""
        model = self._make_model(monkeypatch)
        block = nn.Module()
        block.mlp = nn.Module()
        block.mlp.experts = nn.Module()
        experts_list = nn.ModuleList()
        expert0 = nn.Module()
        expert0.gate_proj = _make_real_quant_linear(name="gate_proj")
        experts_list.append(expert0)
        block.mlp.experts.expert_modules = experts_list
        bindings = list(model.iter_deploy_bindings(0, block))
        paths = [b[0] for b in bindings]
        assert "model.layers.0.mlp.experts.0.gate_proj.weight" in paths

    def test_iter_deploy_bindings_non_moe_path(self, monkeypatch):
        """iter_deploy_bindings yields non-MoE QuantLinear paths directly."""
        model = self._make_model(monkeypatch)
        block = nn.Module()
        block.self_attn = nn.Module()
        block.self_attn.q_a_proj = _make_real_quant_linear(name="q_a_proj")
        bindings = list(model.iter_deploy_bindings(0, block))
        paths = [b[0] for b in bindings]
        assert "model.layers.0.self_attn.q_a_proj.weight" in paths

    def test_trivial_delegations(self, monkeypatch):
        """Verify trivial delegation methods call super()."""
        model = self._make_model(monkeypatch)
        # load_embed_state_dict
        mock_result = MagicMock()
        monkeypatch.setattr(
            "amct_pytorch.common.models.llm.common.base.BaseModel.load_embed_state_dict",
            lambda self: mock_result,
        )
        assert model.load_embed_state_dict() is mock_result
        # do_embedding_forward
        monkeypatch.setattr(
            "amct_pytorch.common.models.llm.common.base.BaseModel.do_embedding_forward",
            lambda self, *a, **kw: mock_result,
        )
        assert model.do_embedding_forward(None) is mock_result
        # do_block_forward
        monkeypatch.setattr(
            "amct_pytorch.common.models.llm.common.base.BaseModel.do_block_forward",
            lambda self, *a, **kw: mock_result,
        )
        assert model.do_block_forward(0, None) is mock_result
        # iter_ptq_units
        yielded = [("unit0", MagicMock())]
        monkeypatch.setattr(
            "amct_pytorch.common.models.llm.common.base.BaseModel.iter_ptq_units",
            lambda self, *a, **kw: iter(yielded),
        )
        assert list(model.iter_ptq_units(0, MagicMock())) == yielded
        # load_unit_inputs
        monkeypatch.setattr(
            "amct_pytorch.common.models.llm.common.base.BaseModel.load_unit_inputs",
            lambda self, *a, **kw: mock_result,
        )
        assert model.load_unit_inputs("/tmp", MagicMock()) is mock_result


# ---------------------------------------------------------------------------
# B. QuantGlmMoeDsaAttention tests (B1-B6)
# ---------------------------------------------------------------------------

class TestQuantGlmMoeDsaAttention:
    """Tests for QuantGlmMoeDsaAttention init and forward."""

    @staticmethod
    def test_init_without_quant(monkeypatch):
        """__init__ without attn-linear creates PlainLinear for all projections."""
        _register_dtype_once()

        args = _make_quant_args(quant_target=["moe"])
        attn_module = _make_mock_attn_module(with_indexer=True)
        quant_attn = QuantGlmMoeDsaAttention(args, attn_module)

        assert isinstance(quant_attn.q_a_proj, PlainLinear)
        assert isinstance(quant_attn.q_b_proj, PlainLinear)
        assert isinstance(quant_attn.kv_a_proj_with_mqa, PlainLinear)
        assert isinstance(quant_attn.kv_b_proj, PlainLinear)
        assert isinstance(quant_attn.o_proj, PlainLinear)
        assert isinstance(quant_attn.indexer.wq_b, PlainLinear)
        assert isinstance(quant_attn.indexer.wk, PlainLinear)
        assert isinstance(quant_attn.inp_afq, nn.Identity)
        assert isinstance(quant_attn.q_b_proj_afq, nn.Identity)
        assert isinstance(quant_attn.o_proj_afq, nn.Identity)
        assert isinstance(quant_attn.q_cache_quantizer, nn.Identity)
        assert isinstance(quant_attn.k_cache_quantizer, nn.Identity)

    @staticmethod
    def _patch_structure_transforms(monkeypatch):
        """Mock build_algorithms_by_target to return None for structure target."""
        monkeypatch.setattr(
            "amct_pytorch.common.models.llm.glm.glm5_2.quant_module.build_algorithms_by_target",
            lambda args, target, *ctor_args: None if target == "structure" else nn.ModuleDict(),
        )

    @staticmethod
    def _mock_rotary_and_attention(monkeypatch, batch=2, heads=4, seq=4, v_dim=16):
        """Mock apply_rotary_pos_emb_interleave and eager_attention_forward."""
        monkeypatch.setattr(
            "amct_pytorch.common.models.llm.glm.glm5_2.quant_module.apply_rotary_pos_emb_interleave",
            lambda q, k, cos, sin: (q, k),
        )
        fake_attn_out = torch.randn(batch, heads, seq, v_dim)
        monkeypatch.setattr(
            "amct_pytorch.common.models.llm.glm.glm5_2.quant_module.eager_attention_forward",
            lambda *a, **kw: (fake_attn_out, None),
        )

    def test_init_with_quant_and_indexer(self, monkeypatch):
        """__init__ with attn-linear target: INT mode quantizes q_b/o/wq_b, not q_a/kv_a/wk."""
        _register_dtype_once()
        self._patch_structure_transforms(monkeypatch)

        args = _make_quant_args(quant_target=["attn-linear"])
        attn_module = _make_mock_attn_module(with_indexer=True)
        quant_attn = QuantGlmMoeDsaAttention(args, attn_module)

        assert isinstance(quant_attn.q_a_proj, PlainLinear)
        assert isinstance(quant_attn.q_b_proj, QuantLinear)
        assert isinstance(quant_attn.kv_a_proj_with_mqa, PlainLinear)
        assert isinstance(quant_attn.o_proj, QuantLinear)
        assert isinstance(quant_attn.kv_b_proj, PlainLinear)
        assert isinstance(quant_attn.indexer.wq_b, QuantLinear)
        assert isinstance(quant_attn.indexer.wk, PlainLinear)
        assert isinstance(quant_attn.indexer.weights_proj, nn.Linear)
        assert hasattr(quant_attn, "inp_afq")
        assert hasattr(quant_attn, "q_b_proj_afq")
        assert hasattr(quant_attn, "o_proj_afq")

    def test_init_shared_layer_no_indexer(self, monkeypatch):
        """__init__ with indexer=None (shared layer) skips indexer quantization."""
        _register_dtype_once()
        self._patch_structure_transforms(monkeypatch)

        args = _make_quant_args(quant_target=["attn-linear"])
        attn_module = _make_mock_attn_module(with_indexer=False)
        quant_attn = QuantGlmMoeDsaAttention(args, attn_module)

        assert quant_attn.indexer is None

    def test_init_attn_cache_creates_cache_quantizers(self, monkeypatch):
        """attn-cache target creates ActivationQuantizer for q/k cache."""
        _register_dtype_once()
        self._patch_structure_transforms(monkeypatch)

        args = _make_quant_args(quant_target=["attn-cache"])
        attn_module = _make_mock_attn_module(with_indexer=True)
        quant_attn = QuantGlmMoeDsaAttention(args, attn_module)

        from amct_pytorch.quantization.modules.quant_base import ActivationQuantizer
        assert isinstance(quant_attn.q_cache_quantizer, ActivationQuantizer)
        assert isinstance(quant_attn.k_cache_quantizer, ActivationQuantizer)

    def test_forward_full_layer(self, monkeypatch):
        """forward with indexer and 4D attention_mask."""
        _register_dtype_once()
        self._patch_structure_transforms(monkeypatch)
        self._mock_rotary_and_attention(monkeypatch)

        args = _make_quant_args(quant_target=["attn-linear"])
        attn_module = _make_mock_attn_module(with_indexer=True)
        quant_attn = QuantGlmMoeDsaAttention(args, attn_module)

        fake_topk = torch.zeros(2, 4, 2, dtype=torch.long)
        quant_attn.indexer.forward = MagicMock(return_value=fake_topk)

        hidden = torch.randn(2, 4, 64)
        pos_emb = (torch.randn(2, 4, 16), torch.randn(2, 4, 16))
        mask = torch.zeros(2, 1, 4, 4)

        out, _, topk = quant_attn.forward(hidden, pos_emb, attention_mask=mask)

        assert out.shape == (2, 4, 64)
        assert topk is fake_topk

    def test_forward_shared_layer(self, monkeypatch):
        """forward without indexer uses prev_topk_indices."""
        _register_dtype_once()
        self._patch_structure_transforms(monkeypatch)
        self._mock_rotary_and_attention(monkeypatch)

        args = _make_quant_args(quant_target=["attn-linear"])
        attn_module = _make_mock_attn_module(with_indexer=False)
        quant_attn = QuantGlmMoeDsaAttention(args, attn_module)

        fake_topk = torch.zeros(2, 4, 2, dtype=torch.long)
        hidden = torch.randn(2, 4, 64)
        pos_emb = (torch.randn(2, 4, 16), torch.randn(2, 4, 16))
        mask = torch.zeros(2, 1, 4, 4)

        out, _, topk = quant_attn.forward(
            hidden, pos_emb, attention_mask=mask, prev_topk_indices=fake_topk,
        )

        assert out.shape == (2, 4, 64)
        assert topk is fake_topk

    def test_forward_attention_mask_variants(self, monkeypatch):
        """forward handles non-4D mask and None mask."""
        _register_dtype_once()
        self._patch_structure_transforms(monkeypatch)
        self._mock_rotary_and_attention(monkeypatch)

        args = _make_quant_args(quant_target=["attn-linear"])
        fake_topk = torch.zeros(2, 4, 2, dtype=torch.long)
        hidden = torch.randn(2, 4, 64)
        pos_emb = (torch.randn(2, 4, 16), torch.randn(2, 4, 16))

        # Non-4D mask (2D) — shared layer
        attn_2d = QuantGlmMoeDsaAttention(args, _make_mock_attn_module(with_indexer=False))
        out_2d, _, _ = attn_2d.forward(
            hidden, pos_emb, attention_mask=torch.zeros(4, 4),
            prev_topk_indices=fake_topk,
        )
        assert out_2d.shape == (2, 4, 64)

        # None mask — shared layer
        attn_none = QuantGlmMoeDsaAttention(args, _make_mock_attn_module(with_indexer=False))
        out_none, _, _ = attn_none.forward(
            hidden, pos_emb, attention_mask=None,
            prev_topk_indices=fake_topk,
        )
        assert out_none.shape == (2, 4, 64)


class TestQuantGlmIndexer:
    """Tests for QuantGlmIndexer init and forward."""

    @staticmethod
    def _make_mock_indexer():
        return SimpleNamespace(
            n_heads=4,
            head_dim=32,
            qk_rope_head_dim=16,
            index_topk=4,
            q_lora_rank=64,
            softmax_scale=32 ** -0.5,
            wq_b=nn.Linear(64, 4 * 32, bias=False),
            wk=nn.Linear(64, 32, bias=False),
            k_norm=nn.LayerNorm(32),
            weights_proj=nn.Linear(64, 4, bias=False),
        )

    def test_init_without_quant(self, monkeypatch):
        """__init__ without use_quant creates PlainLinear for all projections."""
        _register_dtype_once()

        args = _make_quant_args(quant_target=["moe"])
        indexer = self._make_mock_indexer()
        bits = args.bit_policy["attn-linear"]
        quant_indexer = QuantGlmIndexer(args, indexer, bits, use_quant=False)

        assert isinstance(quant_indexer.wq_b, PlainLinear)
        assert isinstance(quant_indexer.wk, PlainLinear)
        assert isinstance(quant_indexer.qr_afq, nn.Identity)
        assert isinstance(quant_indexer.x_afq, nn.Identity)
        assert isinstance(quant_indexer.q_cache_quantizer, nn.Identity)
        assert isinstance(quant_indexer.k_cache_quantizer, nn.Identity)

    def test_init_with_attn_linear_int(self, monkeypatch):
        """INT mode: wq_b quantized, wk PlainLinear, activation quantizers created."""
        _register_dtype_once()
        args = _make_quant_args(quant_target=["attn-linear"], quant_dtype="int")
        indexer = self._make_mock_indexer()
        bits = args.bit_policy["attn-linear"]
        quant_indexer = QuantGlmIndexer(args, indexer, bits, use_quant=True)

        assert isinstance(quant_indexer.wq_b, QuantLinear)
        assert isinstance(quant_indexer.wk, PlainLinear)
        from amct_pytorch.quantization.modules.quant_base import ActivationQuantizer
        assert isinstance(quant_indexer.qr_afq, ActivationQuantizer)
        assert isinstance(quant_indexer.x_afq, ActivationQuantizer)

    def test_init_with_attn_linear_mxfp(self, monkeypatch):
        """MXFP mode: both wq_b and wk quantized."""
        _register_dtype_once()
        args = _make_quant_args(quant_target=["attn-linear"], quant_dtype="mxfp")
        indexer = self._make_mock_indexer()
        bits = args.bit_policy["attn-linear"]
        quant_indexer = QuantGlmIndexer(args, indexer, bits, use_quant=True)

        assert isinstance(quant_indexer.wq_b, QuantLinear)
        assert isinstance(quant_indexer.wk, QuantLinear)

    def test_init_with_attn_cache(self, monkeypatch):
        """attn-cache target creates cache quantizers."""
        _register_dtype_once()
        args = _make_quant_args(quant_target=["attn-cache"])
        indexer = self._make_mock_indexer()
        bits = args.bit_policy["attn-linear"]
        quant_indexer = QuantGlmIndexer(args, indexer, bits, use_quant=True)

        from amct_pytorch.quantization.modules.quant_base import ActivationQuantizer
        assert isinstance(quant_indexer.q_cache_quantizer, ActivationQuantizer)
        assert isinstance(quant_indexer.k_cache_quantizer, ActivationQuantizer)

    def test_forward_returns_topk_indices(self, monkeypatch):
        """forward produces topk indices of correct shape and dtype."""
        _register_dtype_once()
        args = _make_quant_args(quant_target=["attn-linear"])
        indexer = self._make_mock_indexer()
        bits = args.bit_policy["attn-linear"]
        quant_indexer = QuantGlmIndexer(args, indexer, bits, use_quant=True)

        monkeypatch.setattr(
            "amct_pytorch.common.models.llm.glm.glm5_2.quant_module.apply_rotary_pos_emb",
            lambda q, k, cos, sin, unsqueeze_dim=0: (q, k),
        )

        batch, seq, hidden = 2, 4, 64
        hidden_states = torch.randn(batch, seq, hidden)
        q_resid = torch.randn(batch, seq, hidden)
        pos_emb = (torch.randn(batch, seq, 16), torch.randn(batch, seq, 16))
        mask = torch.zeros(batch, seq, seq)

        topk = quant_indexer.forward(hidden_states, q_resid, pos_emb, mask, None)

        assert topk.shape == (batch, seq, 4)
        assert topk.dtype == torch.int32


# ---------------------------------------------------------------------------
# C. GLM5_2 new functions (lines 156-413)
# ---------------------------------------------------------------------------

def _make_glm5_config_ext(num_layers=2, num_nextn=0, first_k_dense=0,
                          indexer_types=None):
    """Extended config factory with MTP and dense-replace control."""
    kwargs = dict(
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
        indexer_types=indexer_types if indexer_types is not None
                      else ["full"] * num_layers,
        first_k_dense_replace=first_k_dense,
    )
    if num_nextn > 0:
        kwargs["num_nextn_predict_layers"] = num_nextn
    return GlmMoeDsaConfig(**kwargs)


def _make_tensor_bit_policy(w4a8=False):
    """Construct a BitPolicy with per-group entries for tensor mode tests.

    Mirrors the structure AMCT deploy would build from a bit_config YAML:
    - attn-linear: per-name bits (q_a/kv_a/wk=16 to model "not quantized in int")
    - moe.routed: w_bits=4 for w4a8, else 8
    - moe.shared / mlp: 8
    """
    moe_w = 4 if w4a8 else 8
    cfg = {
        W_BITS: 8,
        A_BITS: 8,
        "attn-linear": {
            "q_a_proj": {W_BITS: 16, A_BITS: 16},
            "q_b_proj": {W_BITS: 8, A_BITS: 8},
            "kv_a_proj_with_mqa": {W_BITS: 16, A_BITS: 16},
            "o_proj": {W_BITS: 8, A_BITS: 8},
            "wq_b": {W_BITS: 8, A_BITS: 8},
            "wk": {W_BITS: 16, A_BITS: 16},
        },
        "mlp": {
            GATE_PROJ: {W_BITS: 8, A_BITS: 8},
            DOWN_PROJ: {W_BITS: 8, A_BITS: 8},
            UP_PROJ: {W_BITS: 8, A_BITS: 8},
        },
        "moe": {
            "routed": {
                GATE_PROJ: {W_BITS: moe_w, A_BITS: 8},
                DOWN_PROJ: {W_BITS: moe_w, A_BITS: 8},
                UP_PROJ: {W_BITS: moe_w, A_BITS: 8},
            },
            "shared": {
                GATE_PROJ: {W_BITS: 8, A_BITS: 8},
                DOWN_PROJ: {W_BITS: 8, A_BITS: 8},
                UP_PROJ: {W_BITS: 8, A_BITS: 8},
            },
        },
    }
    return BitPolicy(cfg)


class TestGLM5_2NewFunctions:
    """Tests for GLM5_2 functions at lines 156-413."""

    @staticmethod
    def _make_model(monkeypatch, num_layers=2, num_nextn=0,
                    first_k_dense=0, indexer_types=None,
                    quant_target=None, quant_dtype="int8"):
        if quant_target is None:
            quant_target = [QUANT_TARGET_MOE]
        cfg = _make_glm5_config_ext(
            num_layers=num_layers, num_nextn=num_nextn,
            first_k_dense=first_k_dense, indexer_types=indexer_types,
        )
        _mock_base_deps(monkeypatch, cfg)
        args = _make_mock_args(
            quant_target=quant_target, quant_dtype=quant_dtype,
        )
        args.bit_policy = _make_tensor_bit_policy(w4a8=False)
        return GLM5_2(args)

    # -- get_scale_name --

    @pytest.mark.cpu
    def test_get_scale_name_replaces_weight_with_scale(self, monkeypatch):
        """get_scale_name returns (.scale, <name>.scale) for a layered weight."""
        model = self._make_model(monkeypatch, quant_dtype="int8")
        scale_prefix, scale_inv_name = model.get_scale_name(
            "model.layers.0.mlp.up_proj.weight"
        )
        assert scale_prefix == ".scale"
        assert scale_inv_name == "model.layers.0.mlp.up_proj.scale"

    @pytest.mark.cpu
    def test_get_scale_name_with_simple_weight_name(self, monkeypatch):
        """get_scale_name handles simple weight names (no layer path)."""
        model = self._make_model(monkeypatch, quant_dtype="int8")
        scale_prefix, scale_inv_name = model.get_scale_name("lm_head.weight")
        assert scale_prefix == ".scale"
        assert scale_inv_name == "lm_head.scale"

    # -- cache_scheme --

    @pytest.mark.cpu
    def test_cache_scheme_int_skips_kv_keeps_li(self, monkeypatch):
        """cache_scheme for non-mxfp: kv_cache_scheme is None, li_cache_scheme is int."""
        model = self._make_model(monkeypatch, quant_dtype="int8")
        result = model.cache_scheme()
        assert result["kv_cache_scheme"] is None
        assert result["li_cache_scheme"] == {"type": "int", "num_bits": 8}

    @pytest.mark.cpu
    def test_cache_scheme_mxfp(self, monkeypatch):
        """cache_scheme for mxfp: kv/li cache + activation_scheme + quant_method."""
        model = self._make_model(monkeypatch, quant_dtype="mxfp")
        result = model.cache_scheme()
        kv = result["kv_cache_scheme"]
        assert kv["num_bits"] == 8
        assert kv["type"] == "float"
        assert kv["strategy"] == "group"
        assert kv["group_size"] == 128
        assert kv["dynamic"] == "true"
        assert kv["symmetric"] == "true"
        assert result["li_cache_scheme"] == {"type": "float", "num_bits": 8}
        assert result["activation_scheme"] == "dynamic"
        # w8a8 (non-w4a8) mxfp uses quant_method=mxfp8
        assert result["quant_method"] == "mxfp8"

    @pytest.mark.cpu
    def test_cache_scheme_mxfp_w4a8(self, monkeypatch):
        """w4a8-mxfp: quant_method should NOT be set (only w8a8-mxfp sets mxfp8)."""
        model = self._make_model(monkeypatch, quant_dtype="mxfp")
        model.args.bit_policy = BitPolicy({
            "w_bits": 8, "a_bits": 8,
            "moe": {"routed": {"w_bits": 4, "a_bits": 8}},
        })
        result = model.cache_scheme()
        assert result["kv_cache_scheme"]["type"] == "float"
        assert result["li_cache_scheme"]["type"] == "float"
        assert result["activation_scheme"] == "dynamic"
        assert "quant_method" not in result

    # -- generate_tensorwise_quant_layers --

    @pytest.mark.cpu
    def test_tensorwise_quant_int_dense_and_moe_mix(self, monkeypatch):
        """INT mode: dense layers (i<first_k_dense) use mlp; MoE layers use experts+shared."""
        model = self._make_model(
            monkeypatch, num_layers=4, first_k_dense=2, num_nextn=0,
            quant_dtype="int8",
        )
        layers = model.generate_tensorwise_quant_layers()
        # Dense layers 0,1: mlp.gate_proj/down_proj/up_proj
        assert "model.layers.0.mlp.gate_proj" in layers
        assert "model.layers.0.mlp.down_proj" in layers
        assert "model.layers.0.mlp.up_proj" in layers
        assert "model.layers.1.mlp.gate_proj" in layers
        # MoE layers 2,3: experts + shared_experts
        assert "model.layers.2.mlp.experts.0.gate_proj" in layers
        assert "model.layers.2.mlp.experts.7.down_proj" in layers
        assert "model.layers.2.mlp.shared_experts.up_proj" in layers
        assert "model.layers.3.mlp.experts.0.gate_proj" in layers
        # Dense layers should NOT have experts/shared_experts
        assert not any("model.layers.0.mlp.experts" in k for k in layers)
        assert not any("model.layers.0.mlp.shared_experts" in k for k in layers)
        # MoE layers should NOT have plain mlp.gate_proj
        assert "model.layers.2.mlp.gate_proj" not in layers
        # INT mode: no q_a_proj/kv_a_proj_with_mqa/indexer.wk
        assert not any("q_a_proj" in k for k in layers)
        assert not any("kv_a_proj_with_mqa" in k for k in layers)
        assert not any("indexer.wk" in k for k in layers)
        # All layers have o_proj/q_b_proj/indexer.wq_b
        for i in range(4):
            assert f"model.layers.{i}.self_attn.o_proj" in layers
            assert f"model.layers.{i}.self_attn.q_b_proj" in layers
            assert f"model.layers.{i}.self_attn.indexer.wq_b" in layers
        # kv_b_proj never in quant_layers
        assert not any("kv_b_proj" in k for k in layers)
        # mlp.gate (router) never in quant_layers
        assert not any(k.endswith(".mlp.gate") for k in layers)

    @pytest.mark.cpu
    def test_tensorwise_quant_mxfp_extra_layers(self, monkeypatch):
        """MXFP mode: q_a_proj/kv_a_proj_with_mqa/indexer.wk are quantized."""
        model = self._make_model(
            monkeypatch, num_layers=2, first_k_dense=0, num_nextn=0,
            quant_dtype="mxfp",
        )
        layers = model.generate_tensorwise_quant_layers()
        # mxfp-only layers present
        assert "model.layers.0.self_attn.q_a_proj" in layers
        assert "model.layers.0.self_attn.kv_a_proj_with_mqa" in layers
        assert "model.layers.0.self_attn.indexer.wk" in layers
        assert "model.layers.1.self_attn.q_a_proj" in layers
        # always-quantized still present
        assert "model.layers.0.self_attn.o_proj" in layers
        assert "model.layers.0.self_attn.q_b_proj" in layers
        assert "model.layers.0.self_attn.indexer.wq_b" in layers

    @pytest.mark.cpu
    def test_tensorwise_quant_covers_mtp_layer(self, monkeypatch):
        """Tensor mode covers MTP layer (num_layers = num_hidden + num_nextn)."""
        model = self._make_model(
            monkeypatch, num_layers=2, num_nextn=1, first_k_dense=0,
            quant_dtype="int8",
        )
        layers = model.generate_tensorwise_quant_layers()
        # Layer 2 is MTP — must be covered
        assert "model.layers.2.self_attn.o_proj" in layers
        assert "model.layers.2.mlp.experts.0.gate_proj" in layers
        assert "model.layers.2.mlp.shared_experts.gate_proj" in layers

    @pytest.mark.cpu
    def test_tensorwise_quant_w4a8_routed_bit(self, monkeypatch):
        """w4a8: moe.routed experts get bit=4, shared/attn stay 8."""
        model = self._make_model(
            monkeypatch, num_layers=2, first_k_dense=0, num_nextn=0,
            quant_dtype="int8",
        )
        model.args.bit_policy = _make_tensor_bit_policy(w4a8=True)
        layers = model.generate_tensorwise_quant_layers()
        # Routed experts = 4
        assert layers["model.layers.0.mlp.experts.0.gate_proj"] == 4
        assert layers["model.layers.0.mlp.experts.7.up_proj"] == 4
        # Shared experts = 8
        assert layers["model.layers.0.mlp.shared_experts.gate_proj"] == 8
        # Attn bits = 8
        assert layers["model.layers.0.self_attn.o_proj"] == 8
        assert layers["model.layers.0.self_attn.q_b_proj"] == 8

    @pytest.mark.cpu
    def test_tensorwise_quant_default_bit_fallback(self, monkeypatch):
        """Without per-group bit_policy, all bits fall back to top-level w_bits=8."""
        model = self._make_model(
            monkeypatch, num_layers=2, first_k_dense=0, num_nextn=0,
            quant_dtype="int8",
        )
        # Minimal bit_policy: only top-level, no per-group entries
        model.args.bit_policy = BitPolicy({"w_bits": 8, "a_bits": 8})
        layers = model.generate_tensorwise_quant_layers()
        assert layers["model.layers.0.mlp.experts.0.gate_proj"] == 8
        assert layers["model.layers.0.mlp.shared_experts.gate_proj"] == 8
        assert layers["model.layers.0.self_attn.o_proj"] == 8

    @pytest.mark.cpu
    def test_tensorwise_indexer_types_filtering(self, monkeypatch):
        """'shared' layers have no indexer weights; only 'full' and MTP layers do."""
        model = self._make_model(
            monkeypatch, num_layers=3, first_k_dense=0, num_nextn=0,
            indexer_types=["full", "shared", "full"], quant_dtype="int8",
        )
        layers = model.generate_tensorwise_quant_layers()
        ignore = model.generate_tensorwise_ignore_layers()
        # Layer 0 (full): has indexer.wq_b
        assert "model.layers.0.self_attn.indexer.wq_b" in layers
        assert "model.layers.0.self_attn.indexer.weights_proj" in ignore
        # Layer 1 (shared): NO indexer weights
        assert "model.layers.1.self_attn.indexer.wq_b" not in layers
        assert not any("model.layers.1.self_attn.indexer" in n for n in ignore)
        # Layer 2 (full): has indexer.wq_b
        assert "model.layers.2.self_attn.indexer.wq_b" in layers
        assert "model.layers.2.self_attn.indexer.weights_proj" in ignore

    # -- bits_scheme --

    @pytest.mark.cpu
    def test_bits_scheme_w8a8_single_group(self, monkeypatch):
        """w8a8: bits_scheme returns only Linear group, no MoEGMM."""
        model = self._make_model(
            monkeypatch, num_layers=2, first_k_dense=0, num_nextn=0,
            quant_dtype="int8",
        )
        groups = model.bits_scheme()
        assert len(groups) == 1
        assert groups[0]["targets"] == ["Linear"]
        assert groups[0]["w_bits"] == 8
        assert groups[0]["a_bits"] == 8

    @pytest.mark.cpu
    def test_bits_scheme_w4a8_dual_group(self, monkeypatch):
        """w4a8: bits_scheme returns Linear group + MoEGMM group with w_bits=4."""
        model = self._make_model(
            monkeypatch, num_layers=2, first_k_dense=0, num_nextn=0,
            quant_dtype="int8",
        )
        # Build a bit_policy where moe.routed group default has w_bits=4
        model.args.bit_policy = BitPolicy({
            W_BITS: 8, A_BITS: 8,
            "moe": {"routed": {W_BITS: 4, A_BITS: 8}},
        })
        groups = model.bits_scheme()
        assert len(groups) == 2
        assert groups[0]["targets"] == ["Linear"]
        assert groups[1]["targets"] == ["MoEGMM"]
        assert groups[1][W_BITS] == 4
        assert groups[1][A_BITS] == 8

    # -- generate_tensorwise_ignore_layers --

    @pytest.mark.cpu
    def test_tensorwise_ignore_basic(self, monkeypatch):
        """ignore list aligns with infer repo generate_ignore_item (int mode)."""
        model = self._make_model(
            monkeypatch, num_layers=3, num_nextn=0, quant_dtype="int8",
        )
        ignore = model.generate_tensorwise_ignore_layers()
        # kv_b_proj always ignored
        assert "model.layers.0.self_attn.kv_b_proj" in ignore
        assert "model.layers.1.self_attn.kv_b_proj" in ignore
        assert "model.layers.2.self_attn.kv_b_proj" in ignore
        assert "lm_head" in ignore
        # INT mode: q_a_proj, kv_a_proj_with_mqa, indexer.wk are in ignore
        assert "model.layers.0.self_attn.q_a_proj" in ignore
        assert "model.layers.0.self_attn.kv_a_proj_with_mqa" in ignore
        assert "model.layers.0.self_attn.indexer.wk" in ignore
        assert "model.layers.0.self_attn.indexer.weights_proj" in ignore
        # No embed_tokens, no eh_proj (non-MTP)
        assert not any("embed_tokens" in n for n in ignore)
        assert not any("eh_proj" in n for n in ignore)

    @pytest.mark.cpu
    def test_tensorwise_ignore_covers_mtp(self, monkeypatch):
        """MTP layer adds eh_proj + shared_head.head to ignore list."""
        model = self._make_model(
            monkeypatch, num_layers=2, num_nextn=1, quant_dtype="int8",
        )
        ignore = model.generate_tensorwise_ignore_layers()
        # num_layers = 2 + 1 = 3
        assert "model.layers.0.self_attn.kv_b_proj" in ignore
        assert "model.layers.1.self_attn.kv_b_proj" in ignore
        assert "model.layers.2.self_attn.kv_b_proj" in ignore
        assert "lm_head" in ignore
        # MTP layer (layer 2) has eh_proj and shared_head.head
        assert "model.layers.2.eh_proj" in ignore
        assert "model.layers.2.shared_head.head" in ignore
        # Non-MTP layers do NOT have eh_proj
        assert "model.layers.0.eh_proj" not in ignore
        assert "model.layers.1.eh_proj" not in ignore


if __name__ == "__main__":
    pytest.main([__file__])
