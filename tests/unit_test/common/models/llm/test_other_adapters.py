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
"""Lightweight adapter tests for longcat / qwen3.6 families."""

import json
import contextlib
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from amct_pytorch.common.models import MODEL_REGISTRY
from amct_pytorch.common.models.llm.longcat.longcat_lite.longcat_lite import LongcatLite
from amct_pytorch.common.models.llm.longcat.longcat_next.longcat_next import LongcatNext
from amct_pytorch.common.models.llm.kimi.kimi_k3.kimi_k3 import KimiK3
from amct_pytorch.common.models.llm.qwen.qwen3_6.qwen3_6_moe import Qwen3_6Moe
from amct_pytorch.quantization.dtypes import register_dtype

register_dtype()


def _stub(adapter_cls, **attrs):
    obj = adapter_cls.__new__(adapter_cls)
    obj.args = SimpleNamespace(quant_target=attrs.get("quant_target", []))
    obj.quant_target = attrs.get("quant_target", [])
    for k, v in attrs.items():
        setattr(obj, k, v)
    return obj


# ---- LongcatLite / LongcatNext ------------------------------------------


def test_longcat_lite_layer_weight_prefix():
    obj = _stub(LongcatLite, quant_target=["moe"])
    assert obj.get_layer_weight_prefix(0) == "model.layers.0."


def test_longcat_next_layer_weight_prefix_delegates_to_super():
    # LongcatNext inherits LongcatLite's prefix scheme via super().
    obj = _stub(LongcatNext, quant_target=["moe"])
    assert obj.get_layer_weight_prefix(2) == "model.layers.2."


def test_longcat_lite_set_safe_attn_impl_writes_eager():
    cfg = SimpleNamespace()
    obj = _stub(LongcatLite, config=cfg)
    obj._set_safe_attn_impl()
    assert cfg._attn_implementation == "eager"


# ---- Qwen3.6Moe is essentially a stub class ------------------------------


def test_qwen3_6_moe_registers_in_model_registry():
    assert MODEL_REGISTRY.get("qwen3_6_moe") is Qwen3_6Moe


def test_kimi_k3_registers_in_model_registry():
    assert MODEL_REGISTRY.get("kimi_k3") is KimiK3


def test_kimi_k3_eval_initializes_base_text_config(monkeypatch):
    config = SimpleNamespace(text_config=SimpleNamespace(num_hidden_layers=3))
    tokenizer = object()
    registered = {}
    monkeypatch.setattr(
        KimiK3,
        "_register_kimi_config",
        staticmethod(lambda: registered.setdefault("config", True)),
    )
    monkeypatch.setattr(
        KimiK3,
        "_load_text_modeling",
        staticmethod(lambda: (
            registered.setdefault("modeling", True),
            SimpleNamespace(KimiDecoderLayer=object),
        )[1]),
    )
    monkeypatch.setattr(KimiK3, "empty_weights_model", lambda self: object())
    monkeypatch.setattr(
        "amct_pytorch.common.models.llm.kimi.kimi_k3.kimi_k3.KimiK3Config.from_pretrained",
        lambda model_path: config,
    )
    monkeypatch.setattr(
        "amct_pytorch.common.models.llm.kimi.kimi_k3.kimi_k3.AutoTokenizer.from_pretrained",
        lambda model_path, trust_remote_code: tokenizer,
    )

    args = SimpleNamespace(granularity="block", model="/fake", quant_target=[], quant_dtype="bf16")
    model = KimiK3(args)

    assert registered == {"config": True, "modeling": True}
    assert model.kimi_config is config
    assert model.config is config.text_config
    assert model.text_config is config.text_config
    assert model.tokenizer is tokenizer
    assert model.num_layers == 3


def test_kimi_k3_float_model_loads_text_backbone(monkeypatch):
    text_config = SimpleNamespace()
    model = KimiK3.__new__(KimiK3)
    model.model_path = "/fake"
    model.text_config = text_config
    state_dict = {"model.embed_tokens.weight": torch.ones(1, 1)}
    loaded = {}

    class FakeTextModel:
        def load_state_dict(self, state, strict):
            loaded["state"] = state
            loaded["strict"] = strict

        def bfloat16(self):
            loaded["bfloat16"] = True
            return self

    fake_text_model = FakeTextModel()

    class FakeTextModelCls:
        def __init__(self, config):
            loaded["config"] = config

        def __new__(cls, config):
            loaded["config"] = config
            return fake_text_model

    def fake_modeling():
        return SimpleNamespace(KimiLinearForCausalLM=FakeTextModelCls)

    monkeypatch.setattr(KimiK3, "_load_text_modeling", staticmethod(fake_modeling))
    monkeypatch.setattr(model, "_load_text_state_dict", lambda: state_dict)

    assert model.float_model() is fake_text_model
    assert loaded == {
        "config": text_config,
        "state": state_dict,
        "strict": True,
        "bfloat16": True,
    }


def test_kimi_k3_empty_weights_model_uses_repo_text_modeling(monkeypatch):
    model = KimiK3.__new__(KimiK3)
    model.text_config = SimpleNamespace()
    loaded = {}

    class FakeTextModel:
        def __init__(self, config):
            loaded["config"] = config

    monkeypatch.setattr(KimiK3, "_load_text_modeling", staticmethod(
        lambda: SimpleNamespace(KimiLinearForCausalLM=FakeTextModel)
    ))
    monkeypatch.setattr(
        "amct_pytorch.common.models.llm.kimi.kimi_k3.kimi_k3.init_empty_weights",
        lambda: contextlib.nullcontext(),
    )

    result = model.empty_weights_model()

    assert isinstance(result, FakeTextModel)
    assert loaded["config"] is model.text_config


def test_kimi_k3_load_layer_weight_skips_packed_mxfp4_without_dequant(monkeypatch, tmp_path):
    model = KimiK3.__new__(KimiK3)
    model.model_path = str(tmp_path)
    model._weight_map = {
        "language_model.model.layers.0.block_sparse_moe.experts.0.w1.weight_packed": "shard.safetensors",
        "language_model.model.layers.0.block_sparse_moe.experts.0.w1.weight_scale": "shard.safetensors",
        "language_model.model.layers.0.block_sparse_moe.experts.0.w2.weight": "shard.safetensors",
        "language_model.model.layers.0.input_layernorm.weight": "shard.safetensors",
    }
    packed = torch.ones(2, 2, dtype=torch.uint8)
    scale = torch.ones(2, 1, dtype=torch.uint8)
    converted = torch.ones(4, 2, dtype=torch.bfloat16)
    norm = torch.ones(2, dtype=torch.bfloat16)
    tensors = {
        "language_model.model.layers.0.block_sparse_moe.experts.0.w1.weight_packed": packed,
        "language_model.model.layers.0.block_sparse_moe.experts.0.w1.weight_scale": scale,
        "language_model.model.layers.0.block_sparse_moe.experts.0.w2.weight": converted,
        "language_model.model.layers.0.input_layernorm.weight": norm,
    }

    class FakeSafeOpen:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def get_tensor(self, name):
            return tensors[name]

    monkeypatch.setattr(
        "amct_pytorch.common.models.llm.kimi.kimi_k3.kimi_k3.safe_open",
        lambda *args, **kwargs: FakeSafeOpen(),
    )
    state = model.load_layer_weight("language_model.model.layers.0.")

    assert "block_sparse_moe.experts.0.w1.weight_packed" not in state
    assert "block_sparse_moe.experts.0.w1.weight_scale" not in state
    assert state["block_sparse_moe.experts.0.w2.weight"] is converted
    assert state["input_layernorm.weight"] is norm


def test_kimi_k3_get_embed_load_specs_uses_language_model_prefixes():
    model = KimiK3.__new__(KimiK3)
    text_model = SimpleNamespace(
        model=SimpleNamespace(
            embed_tokens="embed",
            norm="norm",
            output_attn_res_norm="out_norm",
            output_attn_res_proj="out_proj",
        ),
        lm_head="head",
    )
    model.model = text_model

    assert model.get_embed_load_specs() == [
        ("embed", "language_model.model.embed_tokens."),
        ("norm", "language_model.model.norm."),
        ("head", "language_model.lm_head."),
        ("out_norm", "language_model.model.output_attn_res_norm."),
        ("out_proj", "language_model.model.output_attn_res_proj."),
    ]


class _FakeToModule:
    def to(self, device):
        self.device = device
        return self

    def eval(self):
        return self


def test_kimi_k3_embedding_forward_initializes_attn_residuals():
    model = KimiK3.__new__(KimiK3)
    model.args = SimpleNamespace(device="cpu")
    model.input_ids = None
    model.load_embed_state_dict = lambda: None

    class FakeEmbed(_FakeToModule):
        def __call__(self, inputs):
            return inputs.to(torch.bfloat16).unsqueeze(-1).repeat(1, 1, 2)

    model.model = SimpleNamespace(
        model=SimpleNamespace(
            embed_tokens=FakeEmbed(),
            use_attn_residuals=True,
        )
    )
    samples = [torch.tensor([[1, 2, 3]]), torch.tensor([[4, 5]])]

    outs = model.do_embedding_forward(samples)

    assert torch.equal(outs[0], torch.tensor([[[1, 1], [2, 2], [3, 3]]], dtype=torch.bfloat16))
    assert torch.equal(outs[1], torch.tensor([[[4, 4], [5, 5]]], dtype=torch.bfloat16))
    assert model.input_ids == samples
    assert [tuple(residual.shape) for residual in model._block_residuals] == [(3, 0, 2), (2, 0, 2)]


def test_kimi_k3_block_forward_threads_attn_residuals():
    model = KimiK3.__new__(KimiK3)
    model.args = SimpleNamespace(device="cpu")
    model.model = SimpleNamespace(model=SimpleNamespace(use_attn_residuals=True))
    model._block_residuals = [torch.zeros(2, 0, 2), torch.zeros(1, 0, 2)]
    model.config = SimpleNamespace()
    model.quant_target = []
    calls = []

    class FakeBlock(_FakeToModule):
        is_linear_attn = True

        def __call__(self, sample, **kwargs):
            calls.append(kwargs)
            residual = kwargs["block_residual"]
            new_column = sample.reshape(-1, sample.shape[-1]).unsqueeze(1)
            return sample + 1, torch.cat([residual, new_column], dim=1)

    model._build_block_for_forward = lambda layer_idx, use_quant_block=False: FakeBlock()
    model._dispatch_block = lambda block: block

    samples = [torch.ones(1, 2, 2), torch.ones(1, 1, 2) * 3]
    outs = model.do_block_forward(0, samples)

    assert torch.equal(outs[0], samples[0] + 1)
    assert torch.equal(outs[1], samples[1] + 1)
    assert len(calls) == 2
    assert all(call["block_residual"].shape[-2] == 0 for call in calls)
    assert [tuple(residual.shape) for residual in model._block_residuals] == [(2, 1, 2), (1, 1, 2)]
    assert "cache_position" in calls[0]


def test_kimi_k3_head_forward_applies_output_attn_residual():
    model = KimiK3.__new__(KimiK3)
    model.args = SimpleNamespace(device="cpu")
    model._block_residuals = [torch.ones(2, 1, 2) * 3]
    applied = {}

    class FakeNorm(_FakeToModule):
        def __call__(self, hidden):
            return hidden + 1

    class FakeHead(_FakeToModule):
        def __call__(self, hidden):
            return torch.cat([hidden, hidden.sum(dim=-1, keepdim=True)], dim=-1)

    class FakeTextModel:
        use_attn_residuals = True
        norm = FakeNorm()
        output_attn_res_norm = _FakeToModule()
        output_attn_res_proj = _FakeToModule()

        def _apply_output_attn_res(self, hidden, residual):
            applied["hidden"] = hidden.clone()
            applied["residual"] = residual.clone()
            return hidden + residual.mean()

    model.model = SimpleNamespace(model=FakeTextModel(), lm_head=FakeHead())
    inp = torch.ones(1, 2, 2)

    preds = model.do_head_forward([inp])

    assert torch.equal(applied["hidden"], inp)
    assert torch.equal(applied["residual"], model._block_residuals[0])
    assert preds[0].shape == (1, 1, 3)
    assert torch.equal(preds[0], torch.tensor([[[5., 5., 10.]]]))


def test_kimi_k3_tensor_deploy_builds_moe_quant_layers_from_index(tmp_path):
    index = {
        "weight_map": {
            "language_model.model.layers.19.block_sparse_moe.experts.895.w3.weight_packed": "shard.safetensors",
            "language_model.model.layers.19.block_sparse_moe.experts.895.w3.weight_scale": "shard.safetensors",
            "language_model.model.layers.19.self_attn.q_proj.weight": "shard.safetensors",
        }
    }
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))
    args = SimpleNamespace(granularity="tensor", model=str(tmp_path), quant_target=["moe"])

    model = KimiK3(args)

    assert model.generate_tensorwise_quant_layers() == {
        "language_model.model.layers.19.block_sparse_moe.experts.895.w3": 4,
    }
    assert model.generate_tensorwise_ignore_layers() == []


def test_kimi_k3_tensor_deploy_respects_non_moe_quant_target(tmp_path):
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {
        "language_model.model.layers.0.block_sparse_moe.experts.0.w1.weight_packed": "shard.safetensors",
    }}))
    args = SimpleNamespace(granularity="tensor", model=str(tmp_path), quant_target=["attn-linear"])

    model = KimiK3(args)

    assert model.generate_tensorwise_quant_layers() == {}


def test_kimi_k3_scale_and_output_names_for_packed_weight(tmp_path):
    args = SimpleNamespace(granularity="tensor", model=str(tmp_path), quant_target=["moe"])
    model = KimiK3(args)
    weight_name = "language_model.model.layers.0.block_sparse_moe.experts.0.w1.weight_packed"

    assert model.get_scale_name(weight_name) == (
        ".weight_scale",
        "language_model.model.layers.0.block_sparse_moe.experts.0.w1.weight_scale",
    )
    assert model.tensorwise_output_weight_name(weight_name) == (
        "language_model.model.layers.0.block_sparse_moe.experts.0.w1.weight"
    )


# ---- registry coverage check --------------------------------------------


@pytest.mark.parametrize(
    "key,expected_cls",
    [
        ("longcat_lite", LongcatLite),
    ],
)
def test_other_adapters_register_under_expected_keys(key, expected_cls):
    assert MODEL_REGISTRY.get(key) is expected_cls

