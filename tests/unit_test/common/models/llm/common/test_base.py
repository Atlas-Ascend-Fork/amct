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

import json
import os
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn
from safetensors.torch import save_file

from amct_pytorch.common.models.llm.common.base import BaseModel
from amct_pytorch.common.models.llm.common.ptq_units import make_ptq_unit
from amct_pytorch.quantization.modules.quant_linear import QuantLinear
from tests.unit_test.conftest import SAFETENSORS_TMP_DIR

QUANT_TARGET_ATTN_CACHE = 'attn-cache'
QUANT_TARGET_ATTN_LINEAR = 'attn-linear'
QUANT_TARGET_MLP = 'mlp'

CALLED = 'called'
FROM_PRETRAINED = 'from_pretrained'
POSITION_IDS = 'position_ids'

FLAG = 'flag'


class _StubModel(BaseModel):
    """Subclass that bypasses BaseModel HF model loading via mocking."""

    def __init__(self, args=None, quant_target=()):
        from unittest.mock import patch

        if args is None:
            args = SimpleNamespace()
        if not hasattr(args, 'model'):
            args.model = "/fake/path"
        if not hasattr(args, 'quant_target'):
            args.quant_target = list(quant_target)
        if not hasattr(args, 'quant_dtype'):
            args.quant_dtype = 'int8'

        with patch(
            "amct_pytorch.common.models.llm.common.base.AutoConfig.from_pretrained",
            return_value=SimpleNamespace(tie_word_embeddings=False),
        ), patch(
            "amct_pytorch.common.models.llm.common.base.AutoTokenizer.from_pretrained",
            return_value=SimpleNamespace(),
        ):
            super().__init__(args)


# ---- get_embed_load_specs ---------------------------


def _build_fake_hf_model():
    model = nn.Module()
    model.model = nn.Module()
    model.model.embed_tokens = nn.Embedding(8, 4)
    model.model.norm = nn.LayerNorm(4)
    model.lm_head = nn.Linear(4, 8, bias=False)
    return model


def test_get_embed_load_specs_uses_lm_head_prefix_when_untied():
    stub = _StubModel()
    stub.config = SimpleNamespace(tie_word_embeddings=False)
    stub.model = _build_fake_hf_model()
    specs = stub.get_embed_load_specs()
    prefixes = [p for _, p in specs]
    assert prefixes == ["model.embed_tokens.", "model.norm.", "lm_head."]


def test_get_embed_load_specs_uses_embed_prefix_when_tied():
    stub = _StubModel()
    stub.config = SimpleNamespace(tie_word_embeddings=True)
    stub.model = _build_fake_hf_model()
    specs = stub.get_embed_load_specs()
    prefixes = [p for _, p in specs]
    assert prefixes == ["model.embed_tokens.", "model.norm.", "model.embed_tokens."]


def test_get_embed_load_specs_falls_back_to_untied_when_attr_missing():
    stub = _StubModel()
    stub.config = SimpleNamespace()  # no tie_word_embeddings
    stub.model = _build_fake_hf_model()
    specs = stub.get_embed_load_specs()
    assert specs[2][1] == "lm_head."


# ---- get_block_forward_kwargs --------------------------------------------


def test_get_block_forward_kwargs_empty_when_nothing_set():
    stub = _StubModel(args=SimpleNamespace(device="cpu"))
    assert not stub.get_block_forward_kwargs()


def test_get_block_forward_kwargs_moves_tensors_to_args_device():
    stub = _StubModel(args=SimpleNamespace(device="cpu"))
    stub.position_ids = torch.tensor([[0, 1, 2]])
    stub.attention_mask = torch.ones(1, 3)
    stub.position_embeddings = (torch.ones(1, 3, 4), torch.zeros(1, 3, 4))

    kwargs = stub.get_block_forward_kwargs()
    assert set(kwargs) == {POSITION_IDS, "attention_mask", "position_embeddings"}
    assert kwargs[POSITION_IDS].device.type == "cpu"
    assert torch.equal(kwargs[POSITION_IDS], stub.position_ids)
    pe0, pe1 = kwargs["position_embeddings"]
    assert torch.equal(pe0, stub.position_embeddings[0])
    assert torch.equal(pe1, stub.position_embeddings[1])


# ---- iter_ptq_units ------------------------------------------------------


def _make_block_with_attn_and_mlp():
    block = nn.Module()
    block.self_attn = nn.Linear(4, 4)
    block.mlp = nn.Linear(4, 4)
    return block


def test_iter_ptq_units_yields_attn_when_target_is_attn_linear():
    stub = _StubModel(quant_target=[QUANT_TARGET_ATTN_LINEAR])
    block = _make_block_with_attn_and_mlp()
    units = list(stub.iter_ptq_units(layer_idx=2, block=block))
    assert len(units) == 1
    u = units[0]
    assert u.kind == "attn"
    assert u.name == "self_attn"
    assert u.layer_idx == 2
    assert u.module is block.self_attn


def test_iter_ptq_units_picks_linear_attn_when_block_layer_type_is_linear_attention():
    stub = _StubModel(quant_target=[QUANT_TARGET_ATTN_CACHE])
    block = nn.Module()
    block.linear_attn = nn.Linear(4, 4)
    block.layer_type = "linear_attention"
    units = list(stub.iter_ptq_units(layer_idx=0, block=block))
    assert units[0].name == "linear_attn"
    assert units[0].module is block.linear_attn


def test_iter_ptq_units_yields_mlp_when_target_is_mlp():
    stub = _StubModel(quant_target=[QUANT_TARGET_MLP])
    block = _make_block_with_attn_and_mlp()
    units = list(stub.iter_ptq_units(layer_idx=1, block=block))
    assert len(units) == 1
    assert units[0].kind == QUANT_TARGET_MLP and units[0].name == QUANT_TARGET_MLP
    assert units[0].module is block.mlp


def test_iter_ptq_units_yields_each_expert_when_target_is_moe():
    stub = _StubModel(quant_target=["moe"])
    block = nn.Module()
    block.mlp = nn.Module()
    block.mlp.experts = nn.ModuleList([nn.Linear(4, 4), nn.Linear(4, 4), nn.Linear(4, 4)])
    units = list(stub.iter_ptq_units(layer_idx=0, block=block))
    assert [u.name for u in units] == ["expert_0", "expert_1", "expert_2"]
    assert all(u.kind == "moe" for u in units)
    assert [u.metadata["expert_idx"] for u in units] == [0, 1, 2]


def test_iter_ptq_units_raises_when_target_is_unsupported_and_block_lacks_mlp():
    stub = _StubModel(quant_target=["nonsense"])
    block = nn.Module()  # no mlp attribute
    with pytest.raises(ValueError, match="Unsupported quant target"):
        list(stub.iter_ptq_units(layer_idx=0, block=block))


# ---- save_block_hook_inputs ----------------------------------------------


def test_save_block_hook_inputs_raises_when_hook_name_is_none():
    stub = _StubModel(quant_target=[QUANT_TARGET_MLP], args=SimpleNamespace(data_dir="/tmp"))
    with pytest.raises(ValueError, match="hook_name cannot be None"):
        stub.save_block_hook_inputs(act_stat={}, hook_name=None, layer_idx=0)


def test_save_block_hook_inputs_chooses_attn_save_target_when_attn_in_quant_target(monkeypatch):
    captured = {}

    def fake_save(act_stat, hook_name, save_target, layer_idx, data_dir):
        captured.update(
            act_stat=act_stat,
            hook_name=hook_name,
            save_target=save_target,
            layer_idx=layer_idx,
            data_dir=data_dir,
        )

    monkeypatch.setattr(
        "amct_pytorch.common.models.llm.common.base.save_ptq_inps", fake_save
    )
    stub = _StubModel(quant_target=[QUANT_TARGET_ATTN_LINEAR], args=SimpleNamespace(data_dir="/d"))
    stub.save_block_hook_inputs({"k": 1}, hook_name="hook", layer_idx=4)
    assert captured["save_target"] == "attn"
    assert captured["layer_idx"] == 4
    assert captured["data_dir"] == "/d"
    assert captured["hook_name"] == "hook"


def test_save_block_hook_inputs_uses_quant_target_for_non_attn(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        "amct_pytorch.common.models.llm.common.base.save_ptq_inps",
        lambda *a, **k: captured.update(args=a, kwargs=k),
    )
    stub = _StubModel(quant_target=[QUANT_TARGET_MLP], args=SimpleNamespace(data_dir="/d"))
    stub.save_block_hook_inputs({}, hook_name="hook", layer_idx=2)
    assert captured.get("args")[2] == QUANT_TARGET_MLP  # noqa: E1111


# ---- load_selected_layer_ptq_params --------------------------------------


def test_load_selected_layer_ptq_params_warns_on_missing_dir(monkeypatch):
    stub = _StubModel(
        quant_target=[QUANT_TARGET_MLP],
        args=SimpleNamespace(
            attn_linear_param_dir="",
            attn_cache_param_dir="",
            moe_mlp_param_dir="",  # empty -> fallback
        ),
    )
    # Should never reach the store when the param_dir is empty.
    stub.ptq_param_store = SimpleNamespace(
        load_layer=lambda *a, **k: pytest.fail("should not be called"),
    )
    results = stub.load_selected_layer_ptq_params(0, block=nn.Module())  # pylint: disable=assignment-from-no-return
    assert results == {QUANT_TARGET_MLP: {"loaded": [], "missing": []}}
    # Original quant_target preserved after the temporary swap.
    assert stub.quant_target == [QUANT_TARGET_MLP]  # noqa: E1111


def test_load_selected_layer_ptq_params_routes_to_store_per_target():
    stub = _StubModel(
        quant_target=[QUANT_TARGET_ATTN_LINEAR, QUANT_TARGET_MLP],
        args=SimpleNamespace(
            attn_linear_param_dir="/a",
            attn_cache_param_dir="",
            moe_mlp_param_dir="/m",
        ),
    )
    seen = []

    def fake_load_layer(layer_idx, block, param_dir, strict):
        seen.append((layer_idx, param_dir, list(stub.quant_target)))
        return {"loaded": ["x"], "missing": []}

    stub.ptq_param_store = SimpleNamespace(load_layer=fake_load_layer)
    results = stub.load_selected_layer_ptq_params(7, block=nn.Module())  # pylint: disable=assignment-from-no-return
    assert set(results) == {QUANT_TARGET_ATTN_LINEAR, QUANT_TARGET_MLP}
    # quant_target was temporarily set to a single-target list per call.
    assert seen[0][2] == [QUANT_TARGET_ATTN_LINEAR]
    assert seen[1][2] == [QUANT_TARGET_MLP]
    # The original quant_target is restored.
    assert stub.quant_target == [QUANT_TARGET_ATTN_LINEAR, QUANT_TARGET_MLP]


def test_build_block_for_forward_quant_branch(monkeypatch):
    stub = _StubModel()
    block = nn.Module()

    def fake_build_quant_block(layer_idx):
        return block

    def fake_load(layer_idx, block, strict):
        pass

    stub.build_quant_block = fake_build_quant_block
    stub.load_selected_layer_ptq_params = fake_load
    result = stub._build_block_for_forward(3, use_quant_block=True)
    assert result is block


def test_build_block_for_forward_plain_branch(monkeypatch):
    stub = _StubModel()
    block = nn.Module()

    def fake_block(layer_idx):
        return block

    stub.block = fake_block
    result = stub._build_block_for_forward(5, use_quant_block=False)
    assert result is block


def test_iter_deploy_bindings_returns_empty_on_no_quant_linear():
    stub = _StubModel()

    def get_layer_weight_prefix(layer_idx):
        return "p."

    stub.get_layer_weight_prefix = get_layer_weight_prefix
    block = nn.Module()
    block.fc = nn.Linear(4, 4)
    bindings = list(stub.iter_deploy_bindings(0, block))
    assert not bindings


def test_get_block_forward_kwargs_omits_none_values():
    stub = _StubModel(args=SimpleNamespace(device="cpu"))
    stub.position_ids = torch.tensor([[0, 1, 2]])
    assert "attention_mask" not in stub.get_block_forward_kwargs()
    assert POSITION_IDS in stub.get_block_forward_kwargs()


def test_iter_ptq_units_moe_uses_expert_modules_fallback():
    stub = _StubModel(quant_target=["moe"])
    block = nn.Module()
    block.mlp = nn.Module()
    block.mlp.experts = type("E", (), {"expert_modules": [nn.Linear(4, 4), nn.Linear(4, 4)]})()
    units = list(stub.iter_ptq_units(layer_idx=0, block=block))
    assert [u.name for u in units] == ["expert_0", "expert_1"]


def test_iter_ptq_units_moe_uses_iter_ptq_expert_modules():
    stub = _StubModel(quant_target=["moe"])
    block = nn.Module()
    block.mlp = nn.Module()

    class _IterExperts:
        def __iter__(self):
            yield nn.Linear(4, 4)
            yield nn.Linear(4, 4)

        def iter_ptq_expert_modules(self):
            return self

    block.mlp.experts = _IterExperts()
    units = list(stub.iter_ptq_units(layer_idx=0, block=block))
    assert [u.name for u in units] == ["expert_0", "expert_1"]  # noqa: E1111


def test_load_selected_layer_ptq_params_attn_cache_target():
    stub = _StubModel(
        quant_target=[QUANT_TARGET_ATTN_CACHE],
        args=SimpleNamespace(
            attn_linear_param_dir="",
            attn_cache_param_dir="/cache_params",
            moe_mlp_param_dir="",
        ),
    )
    seen = []

    def fake_load(layer_idx, block, param_dir, strict):
        seen.append((layer_idx, param_dir))
        return {"loaded": ["u1"], "missing": []}

    stub.ptq_param_store = SimpleNamespace(load_layer=fake_load)
    results = stub.load_selected_layer_ptq_params(5, block=nn.Module())  # pylint: disable=assignment-from-no-return
    assert QUANT_TARGET_ATTN_CACHE in results
    assert seen[0][1] == "/cache_params"
    assert stub.quant_target == [QUANT_TARGET_ATTN_CACHE]  # noqa: E1111


def test_load_selected_layer_ptq_params_moe_target():
    stub = _StubModel(
        quant_target=["moe"],
        args=SimpleNamespace(
            attn_linear_param_dir="",
            attn_cache_param_dir="",
            moe_mlp_param_dir="/moe_params",
        ),
    )
    seen = []

    def fake_load(layer_idx, block, param_dir, strict):
        seen.append((layer_idx, param_dir))
        return {"loaded": ["e0"], "missing": []}

    stub.ptq_param_store = SimpleNamespace(load_layer=fake_load)
    results = stub.load_selected_layer_ptq_params(3, block=nn.Module())  # pylint: disable=assignment-from-no-return
    assert "moe" in results
    assert seen[0][1] == "/moe_params"


def test_load_selected_layer_ptq_params_partial_warns():
    stub = _StubModel(
        quant_target=[QUANT_TARGET_MLP],
        args=SimpleNamespace(
            attn_linear_param_dir="",
            attn_cache_param_dir="",
            moe_mlp_param_dir="/mlp_params",
        ),
    )

    def fake_load(layer_idx, block, param_dir, strict):
        return {"loaded": ["x"], "missing": ["y"]}

    stub.ptq_param_store = SimpleNamespace(load_layer=fake_load)
    results = stub.load_selected_layer_ptq_params(1, block=nn.Module())  # pylint: disable=assignment-from-no-return
    assert results[QUANT_TARGET_MLP]["loaded"] == ["x"]
    assert results[QUANT_TARGET_MLP]["missing"] == ["y"]


def test_load_selected_layer_ptq_params_full_missing_warns():
    stub = _StubModel(
        quant_target=[QUANT_TARGET_MLP],
        args=SimpleNamespace(
            attn_linear_param_dir="",
            attn_cache_param_dir="",
            moe_mlp_param_dir="/mlp_params",
        ),
    )

    def fake_load(layer_idx, block, param_dir, strict):
        return {"loaded": [], "missing": ["z"]}

    stub.ptq_param_store = SimpleNamespace(load_layer=fake_load)
    results = stub.load_selected_layer_ptq_params(2, block=nn.Module())  # pylint: disable=assignment-from-no-return
    assert results[QUANT_TARGET_MLP]["loaded"] == []
    assert results[QUANT_TARGET_MLP]["missing"] == ["z"]


def test_iter_deploy_bindings_yields_quant_linear_weights(monkeypatch):
    stub = _StubModel()

    def get_layer_weight_prefix(layer_idx):
        return "model.layers.0."

    stub.get_layer_weight_prefix = get_layer_weight_prefix

    block = nn.Module()
    block.q_linear = QuantLinear.__new__(QuantLinear)
    block.q_linear._parameters = {}
    block.q_linear._buffers = {}
    block.q_linear._modules = {}

    bindings = list(stub.iter_deploy_bindings(0, block))
    assert len(bindings) == 1
    assert bindings[0][0] == "model.layers.0.q_linear.weight"


def test_iter_ptq_units_raises_for_moe_without_mlp():
    stub = _StubModel(quant_target=["moe"])
    block = nn.Module()
    with pytest.raises(ValueError, match="Unsupported quant target"):
        list(stub.iter_ptq_units(layer_idx=0, block=block))


# ---- build_quant_block ----------------------------------------------------


def test_build_quant_block_delegates_to_block():
    stub = _StubModel(quant_target=[QUANT_TARGET_MLP])
    captured = {}

    def _fake_block(idx):
        captured["idx"] = idx
        return nn.Linear(4, 4)

    stub.block = _fake_block
    result = stub.build_quant_block(3)
    assert captured["idx"] == 3
    assert isinstance(result, nn.Linear)


# ---- get_block_forward_kwargs (individual) / _build_block_for_forward ----


def test_get_block_forward_kwargs_returns_empty_when_no_state():
    stub = _StubModel(quant_target=[QUANT_TARGET_MLP])
    stub.args = SimpleNamespace(device="cpu")
    stub.position_ids = None
    stub.position_embeddings = None
    stub.attention_mask = None
    assert not stub.get_block_forward_kwargs()


def test_get_block_forward_kwargs_includes_position_ids():
    stub = _StubModel(quant_target=[QUANT_TARGET_MLP])
    stub.args = SimpleNamespace(device="cpu")
    pid = torch.tensor([1, 2, 3])
    stub.position_ids = pid
    stub.position_embeddings = None
    stub.attention_mask = None
    kwargs = stub.get_block_forward_kwargs()
    assert torch.equal(kwargs[POSITION_IDS], pid)


def test_get_block_forward_kwargs_includes_attention_mask():
    stub = _StubModel(quant_target=[QUANT_TARGET_MLP])
    stub.args = SimpleNamespace(device="cpu")
    mask = torch.tensor([1, 1, 0])
    stub.position_ids = None
    stub.position_embeddings = None
    stub.attention_mask = mask
    kwargs = stub.get_block_forward_kwargs()
    assert torch.equal(kwargs["attention_mask"], mask)


def test_get_block_forward_kwargs_includes_position_embeddings():
    stub = _StubModel(quant_target=[QUANT_TARGET_MLP])
    stub.args = SimpleNamespace(device="cpu")
    emb = (torch.tensor([1.0]), torch.tensor([2.0]))
    stub.position_ids = None
    stub.position_embeddings = emb
    stub.attention_mask = None
    kwargs = stub.get_block_forward_kwargs()
    assert len(kwargs["position_embeddings"]) == 2
    assert torch.equal(kwargs["position_embeddings"][0], emb[0])


def test_build_block_for_forward_calls_block_when_no_quant():
    stub = _StubModel(quant_target=[QUANT_TARGET_MLP])
    captured = {}

    def _fake_block(idx):
        captured["idx"] = idx
        return nn.Linear(4, 4)
    stub.block = _fake_block
    result = stub._build_block_for_forward(3, use_quant_block=False)
    assert captured["idx"] == 3
    assert isinstance(result, nn.Linear)


def test_build_block_for_forward_calls_build_quant_block_when_quant():
    stub = _StubModel(quant_target=[QUANT_TARGET_MLP])
    captured = {}

    def _fake_quant_block(idx):  # noqa: E1111
        captured["quant_idx"] = idx
        mod = nn.Linear(4, 4)
        return mod
    stub.build_quant_block = _fake_quant_block

    def load_selected_layer_ptq_params(idx, block, strict):
        return {}

    stub.load_selected_layer_ptq_params = load_selected_layer_ptq_params  # noqa: E1111
    result = stub._build_block_for_forward(3, use_quant_block=True)
    assert captured["quant_idx"] == 3


# ---- load_selected_layer_ptq_params (no-param-dir fallbacks) -------------


def test_load_selected_layer_ptq_params_attn_linear_with_no_param_dir_logs_warning():  # pylint: disable=assignment-from-no-return
    stub = _StubModel(quant_target=[QUANT_TARGET_ATTN_LINEAR])
    stub.args = SimpleNamespace(attn_linear_param_dir="")
    block = nn.Module()
    results = stub.load_selected_layer_ptq_params(0, block, strict=False)  # pylint: disable=assignment-from-no-return
    assert QUANT_TARGET_ATTN_LINEAR in results
    assert results[QUANT_TARGET_ATTN_LINEAR]["loaded"] == []


def test_load_selected_layer_ptq_params_mlp_with_no_param_dir():
    stub = _StubModel(quant_target=[QUANT_TARGET_MLP])
    stub.args = SimpleNamespace(moe_mlp_param_dir="")
    block = nn.Module()
    results = stub.load_selected_layer_ptq_params(0, block, strict=False)  # pylint: disable=assignment-from-no-return
    assert QUANT_TARGET_MLP in results


def test_load_selected_layer_ptq_params_moe_with_no_param_dir():
    stub = _StubModel(quant_target=["moe"])
    stub.args = SimpleNamespace(moe_mlp_param_dir="")
    block = nn.Module()
    results = stub.load_selected_layer_ptq_params(0, block, strict=False)  # pylint: disable=assignment-from-no-return
    assert "moe" in results


def test_load_selected_layer_ptq_params_attn_cache_with_no_param_dir():
    stub = _StubModel(quant_target=[QUANT_TARGET_ATTN_CACHE])
    stub.args = SimpleNamespace(attn_cache_param_dir="")
    block = nn.Module()
    results = stub.load_selected_layer_ptq_params(0, block, strict=False)  # pylint: disable=assignment-from-no-return
    assert QUANT_TARGET_ATTN_CACHE in results
    assert results[QUANT_TARGET_ATTN_CACHE]["loaded"] == []


def test_load_selected_layer_ptq_params_restores_original_quant_target():
    stub = _StubModel(quant_target=[QUANT_TARGET_MLP])
    stub.args = SimpleNamespace(moe_mlp_param_dir="")
    block = nn.Module()
    original = stub.quant_target
    stub.load_selected_layer_ptq_params(0, block, strict=False)
    assert stub.quant_target == original


# ---- iter_ptq_units additional branches -----------------------------------


def test_iter_ptq_units_attn_linear_yields_attn():
    stub = _StubModel(quant_target=[QUANT_TARGET_ATTN_LINEAR])
    block = nn.Module()
    block.self_attn = nn.Linear(4, 4)
    units = list(stub.iter_ptq_units(layer_idx=0, block=block))
    assert len(units) == 1
    assert units[0].kind == "attn"


def test_iter_ptq_units_attn_cache_yields_attn():
    stub = _StubModel(quant_target=[QUANT_TARGET_ATTN_CACHE])
    block = nn.Module()
    block.self_attn = nn.Linear(4, 4)
    units = list(stub.iter_ptq_units(layer_idx=0, block=block))
    assert len(units) == 1
    assert units[0].kind == "attn"


def test_iter_ptq_units_moe_yields_experts():
    stub = _StubModel(quant_target=["moe"])
    block = nn.Module()
    expert0 = nn.Linear(4, 4)
    expert1 = nn.Linear(4, 4)
    block.mlp = nn.Module()
    block.mlp.experts = [expert0, expert1]
    units = list(stub.iter_ptq_units(layer_idx=0, block=block))
    assert len(units) == 2
    assert all(u.kind == "moe" for u in units)
    assert units[0].metadata == {"expert_idx": 0}
    assert units[1].metadata == {"expert_idx": 1}


def test_iter_ptq_units_raises_when_no_mlp_and_not_attn():
    stub = _StubModel(quant_target=["unknown"])
    block = nn.Module()
    with pytest.raises(ValueError, match="Unsupported quant target"):
        list(stub.iter_ptq_units(layer_idx=0, block=block))


# ---- iter_deploy_bindings -------------------------------------------------


def test_iter_deploy_bindings_yields_quant_linear_weight_paths():
    stub = _StubModel(quant_target=[QUANT_TARGET_MLP])

    def get_layer_weight_prefix(idx):
        return "model.layers.0."

    stub.get_layer_weight_prefix = get_layer_weight_prefix
    block = nn.Module()
    ql = QuantLinear.__new__(QuantLinear)
    ql._parameters = {}
    ql._buffers = {}
    ql._modules = {}
    block.q_proj = ql
    bindings = list(stub.iter_deploy_bindings(0, block))
    assert len(bindings) == 1
    assert bindings[0][0] == "model.layers.0.q_proj.weight"
    assert bindings[0][1] is ql


def test_iter_deploy_bindings_skips_non_quant_linear():
    stub = _StubModel(quant_target=[QUANT_TARGET_MLP])

    def get_layer_weight_prefix(idx):
        return "model.layers.0."

    stub.get_layer_weight_prefix = get_layer_weight_prefix
    block = nn.Module()
    block.linear = nn.Linear(4, 4)
    bindings = list(stub.iter_deploy_bindings(0, block))
    assert len(bindings) == 0


# ---- load_unit_inputs -----------------------------------------------------


def test_load_unit_inputs_delegates_to_load_ptq_inps(tmp_path, monkeypatch):
    stub = _StubModel(quant_target=[QUANT_TARGET_MLP])
    unit = make_ptq_unit(QUANT_TARGET_MLP, QUANT_TARGET_MLP, layer_idx=2, module=nn.Linear(4, 4))
    fake_data = torch.randn(4, 4)
    monkeypatch.setattr(
        "amct_pytorch.common.models.llm.common.base.load_ptq_inps",
        lambda data_dir, kind, layer_idx: fake_data,
    )
    result = stub.load_unit_inputs("/tmp/fake", unit)
    assert result is fake_data


# ---- safetensors-based weight loading tests ------------------------------


def _make_tiny_safetensors_model_dir(num_layers=1):
    """Create a fake model directory with real safetensors files and index.json."""
    from accelerate import init_empty_weights
    from transformers.models.qwen3.modeling_qwen3 import Qwen3Config, Qwen3DecoderLayer

    model_dir = SAFETENSORS_TMP_DIR / "fake_model"
    model_dir.mkdir(parents=True, exist_ok=True)

    hidden_size = 8
    intermediate_size = 16
    num_attention_heads = 2
    num_key_value_heads = 2
    head_dim = 4
    vocab_size = 100

    config = Qwen3Config(
        num_hidden_layers=num_layers,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        head_dim=head_dim,
        rms_norm_eps=1e-6,
        vocab_size=vocab_size,
        max_position_embeddings=512,
        rope_theta=10000.0,
        tie_word_embeddings=False,
    )

    weight_map = {}

    # Discover parameter names and shapes via init_empty_weights
    with init_empty_weights():
        layer_decoder = Qwen3DecoderLayer(config, 0)
    layer_tensors = {}
    for name, param in layer_decoder.named_parameters():
        full_name = f"model.layers.0.{name}"
        layer_tensors[full_name] = torch.zeros(param.shape, dtype=torch.bfloat16)
        weight_map[full_name] = "model-00001-of-00002.safetensors"

    file_path = model_dir / "model-00001-of-00002.safetensors"
    save_file(layer_tensors, str(file_path))

    # Create embed / head tensors
    embed_head_tensors = {
        "model.embed_tokens.weight": torch.zeros(vocab_size, hidden_size, dtype=torch.bfloat16),
        "model.norm.weight": torch.zeros(hidden_size, dtype=torch.bfloat16),
        "lm_head.weight": torch.zeros(vocab_size, hidden_size, dtype=torch.bfloat16),
    }
    for name in embed_head_tensors:
        weight_map[name] = "model-00002-of-00002.safetensors"
    embed_file = model_dir / "model-00002-of-00002.safetensors"
    save_file(embed_head_tensors, str(embed_file))

    # Write index.json
    index = {
        "metadata": {"total_size": 0},
        "weight_map": weight_map,
    }
    with open(model_dir / "model.safetensors.index.json", "w") as f:
        json.dump(index, f)

    return model_dir, config, weight_map


def _mock_hf_for_safetensors_test(monkeypatch, model_dir, config):
    """Mock HF dependencies to point at the fake safetensors model dir."""
    from accelerate import init_empty_weights
    from transformers import AutoModelForCausalLM

    fake_tokenizer = MagicMock()

    # Build a real empty model to return from mocked from_config/from_pretrained
    with init_empty_weights():
        empty_model = AutoModelForCausalLM.from_config(
            config, trust_remote_code=True, torch_dtype=torch.bfloat16
        )

    monkeypatch.setattr(
        "amct_pytorch.common.models.llm.common.base.AutoConfig",
        type("FakeAC", (), {FROM_PRETRAINED: staticmethod(lambda *a, **kw: config)})(),
    )
    monkeypatch.setattr(
        "amct_pytorch.common.models.llm.common.base.AutoTokenizer",
        type("FakeAT", (), {FROM_PRETRAINED: staticmethod(lambda *a, **kw: fake_tokenizer)})(),
    )
    monkeypatch.setattr(
        "amct_pytorch.common.models.llm.common.base.init_empty_weights",
        lambda: MagicMock(__enter__=lambda s: None, __exit__=lambda *a: None),
    )
    monkeypatch.setattr(
        "amct_pytorch.common.models.llm.common.base.AutoModelForCausalLM",
        type("FakeAMFCLM", (), {
            FROM_PRETRAINED: staticmethod(lambda *a, **kw: empty_model),
            "from_config": staticmethod(lambda *a, **kw: empty_model),
        })(),
    )


def test_load_layer_weight_reads_safetensors_from_disk(monkeypatch):
    from amct_pytorch.common.models.llm.qwen.qwen3.qwen3 import Qwen3

    model_dir, config, weight_map = _make_tiny_safetensors_model_dir()
    _mock_hf_for_safetensors_test(monkeypatch, model_dir, config)
    monkeypatch.setattr(
        "amct_pytorch.common.models.llm.common.base.get_weight_mappings",
        lambda path: weight_map,
    )

    args = SimpleNamespace(
        model=str(model_dir),
        quant_target=[QUANT_TARGET_MLP],
        device="cpu",
        data_dir="/tmp/fake",
        output_dir="/tmp/fake",
        quant_dtype="int8",
        w_bits=8,
        a_bits=8,
        algos=[],
        cali_bsz=2,
        moe_mlp_param_dir="",
    )

    model = Qwen3(args)
    state_dict = model.load_layer_weight("model.layers.0.")
    assert len(state_dict) > 0
    # self_attn.q_proj.weight should be present after prefix stripping
    assert "self_attn.q_proj.weight" in state_dict
    assert state_dict["self_attn.q_proj.weight"].shape == (8, 8)


def test_block_creates_layer_with_weights(monkeypatch):
    from amct_pytorch.common.models.llm.qwen.qwen3.qwen3 import Qwen3

    model_dir, config, weight_map = _make_tiny_safetensors_model_dir()
    _mock_hf_for_safetensors_test(monkeypatch, model_dir, config)
    monkeypatch.setattr(
        "amct_pytorch.common.models.llm.common.base.get_weight_mappings",
        lambda path: weight_map,
    )

    args = SimpleNamespace(
        model=str(model_dir),
        quant_target=[QUANT_TARGET_MLP],
        device="cpu",
        data_dir="/tmp/fake",
        output_dir="/tmp/fake",
        quant_dtype="int8",
        w_bits=8,
        a_bits=8,
        algos=[],
        cali_bsz=2,
        moe_mlp_param_dir="",
    )

    model = Qwen3(args)
    layer = model.block(0)
    assert isinstance(layer, nn.Module)
    # Should have loaded real weights from safetensors
    assert layer.self_attn.q_proj.weight.shape == (8, 8)


def test_load_embed_state_dict_loads_embed_weights(monkeypatch):
    from amct_pytorch.common.models.llm.qwen.qwen3.qwen3 import Qwen3

    model_dir, config, weight_map = _make_tiny_safetensors_model_dir()
    _mock_hf_for_safetensors_test(monkeypatch, model_dir, config)
    monkeypatch.setattr(
        "amct_pytorch.common.models.llm.common.base.get_weight_mappings",
        lambda path: weight_map,
    )

    args = SimpleNamespace(
        model=str(model_dir),
        quant_target=[QUANT_TARGET_MLP],
        device="cpu",
        data_dir="/tmp/fake",
        output_dir="/tmp/fake",
        quant_dtype="int8",
        w_bits=8,
        a_bits=8,
        algos=[],
        cali_bsz=2,
        moe_mlp_param_dir="",
    )

    model = Qwen3(args)

    def float_model():
        return model.empty_weights_model()

    model.float_model = float_model
    model.load_embed_state_dict()


# ---- do_embedding_forward -----------------------------------------------


def test_do_embedding_forward_runs_full_model_forward(tmp_path, monkeypatch):
    """Test do_embedding_forward runs the full model forward pass with safetensors weights."""
    from amct_pytorch.common.models.llm.qwen.qwen3.qwen3 import Qwen3

    model_dir, config, weight_map = _make_tiny_safetensors_model_dir(num_layers=1)
    _mock_hf_for_safetensors_test(monkeypatch, model_dir, config)
    monkeypatch.setattr(
        "amct_pytorch.common.models.llm.common.base.get_weight_mappings",
        lambda path: weight_map,
    )
    monkeypatch.setattr(
        "amct_pytorch.common.models.llm.common.base.tqdm",
        lambda iterable, **kwargs: iterable,
    )

    args = SimpleNamespace(
        model=str(model_dir),
        quant_target=[QUANT_TARGET_MLP],
        device="cpu",
        data_dir=str(tmp_path),
        output_dir=str(tmp_path),
        quant_dtype="int8",
        w_bits=8,
        a_bits=8,
        algos=[],
        cali_bsz=2,
        moe_mlp_param_dir="",
    )

    model = Qwen3(args)

    def float_model():
        return model.empty_weights_model()

    model.float_model = float_model

    batch, seq_len = 2, 4
    samples = [torch.randint(0, config.vocab_size, (batch, seq_len))]

    outs = model.do_embedding_forward(samples)

    assert isinstance(outs, list)


def test_do_embedding_forward_saves_position_info(tmp_path, monkeypatch):
    """Test do_embedding_forward captures position_ids/embeddings from Catcher."""
    from amct_pytorch.common.models.llm.qwen.qwen3.qwen3 import Qwen3

    model_dir, config, weight_map = _make_tiny_safetensors_model_dir(num_layers=1)
    _mock_hf_for_safetensors_test(monkeypatch, model_dir, config)
    monkeypatch.setattr(
        "amct_pytorch.common.models.llm.common.base.get_weight_mappings",
        lambda path: weight_map,
    )
    monkeypatch.setattr(
        "amct_pytorch.common.models.llm.common.base.tqdm",
        lambda iterable, **kwargs: iterable,
    )

    args = SimpleNamespace(
        model=str(model_dir),
        quant_target=[QUANT_TARGET_MLP],
        device="cpu",
        data_dir=str(tmp_path),
        output_dir=str(tmp_path),
        quant_dtype="int8",
        w_bits=8,
        a_bits=8,
        algos=[],
        cali_bsz=2,
        moe_mlp_param_dir="",
    )

    model = Qwen3(args)

    def float_model():
        return model.empty_weights_model()

    model.float_model = float_model

    batch, seq_len = 2, 4
    samples = [torch.randint(0, config.vocab_size, (batch, seq_len))]

    outs = model.do_embedding_forward(samples)

    assert model.position_ids is not None
    assert model.position_embeddings is not None


# ---- do_block_forward / do_head_forward --------------------------------


def _compute_position_embeddings(config, batch, seq_len, device="cpu", dtype=torch.bfloat16):
    from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding
    rope = Qwen3RotaryEmbedding(config)
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch, -1)
    hidden_states = torch.zeros(batch, seq_len, config.hidden_size, device=device, dtype=dtype)
    cos, sin = rope(hidden_states, position_ids)
    return (cos, sin)


def test_do_block_forward_runs_layer_forward_pass(monkeypatch):
    from amct_pytorch.common.models.llm.qwen.qwen3.qwen3 import Qwen3

    model_dir, config, weight_map = _make_tiny_safetensors_model_dir()
    _mock_hf_for_safetensors_test(monkeypatch, model_dir, config)
    monkeypatch.setattr(
        "amct_pytorch.common.models.llm.common.base.get_weight_mappings",
        lambda path: weight_map,
    )

    args = SimpleNamespace(
        model=str(model_dir),
        quant_target=[QUANT_TARGET_MLP],
        device="cpu",
        data_dir="/tmp/fake",
        output_dir="/tmp/fake",
        quant_dtype="int8",
        w_bits=8,
        a_bits=8,
        algos=[],
        cali_bsz=2,
        moe_mlp_param_dir="",
    )

    model = Qwen3(args)
    batch, seq_len, hidden_size = 2, 4, 8
    model.position_embeddings = _compute_position_embeddings(config, batch, seq_len)
    samples = [torch.randn(batch, seq_len, hidden_size, dtype=torch.bfloat16)]

    outs = model.do_block_forward(0, samples, hook_name=None, use_quant_block=False)

    assert len(outs) == 1
    assert outs[0].shape == (batch, seq_len, hidden_size)


def test_do_block_forward_with_hook_enables_act_stat_collection(monkeypatch):
    from amct_pytorch.common.models.llm.qwen.qwen3.qwen3 import Qwen3

    model_dir, config, weight_map = _make_tiny_safetensors_model_dir()
    _mock_hf_for_safetensors_test(monkeypatch, model_dir, config)
    monkeypatch.setattr(
        "amct_pytorch.common.models.llm.common.base.get_weight_mappings",
        lambda path: weight_map,
    )
    monkeypatch.setattr(
        "amct_pytorch.common.models.llm.common.base.register_forward_hooks",
        lambda block, hook_name, hooks, act_stat: None,
    )
    monkeypatch.setattr(
        "amct_pytorch.common.models.llm.common.base.save_ptq_inps",
        lambda act_stat, hook_name, save_target, layer_idx, data_dir: None,
    )

    args = SimpleNamespace(
        model=str(model_dir),
        quant_target=[QUANT_TARGET_MLP],
        device="cpu",
        data_dir="/tmp/fake",
        output_dir="/tmp/fake",
        quant_dtype="int8",
        w_bits=8,
        a_bits=8,
        algos=[],
        cali_bsz=2,
        moe_mlp_param_dir="",
    )

    model = Qwen3(args)
    batch, seq_len, hidden_size = 2, 4, 8
    model.position_embeddings = _compute_position_embeddings(config, batch, seq_len)
    samples = [torch.randn(batch, seq_len, hidden_size, dtype=torch.bfloat16)]

    outs = model.do_block_forward(0, samples, hook_name="test_hook")

    assert len(outs) == 1
    assert outs[0].shape == (batch, seq_len, hidden_size)


def test_do_embedding_forward_saves_kwargs_when_hook_name_set(tmp_path, monkeypatch):
    """do_embedding_forward calls save_ptq_kwargs when hook_name is not None (line 170)."""
    from amct_pytorch.common.models.llm.qwen.qwen3.qwen3 import Qwen3

    model_dir, config, weight_map = _make_tiny_safetensors_model_dir(num_layers=1)
    _mock_hf_for_safetensors_test(monkeypatch, model_dir, config)
    monkeypatch.setattr(
        "amct_pytorch.common.models.llm.common.base.get_weight_mappings",
        lambda path: weight_map,
    )
    monkeypatch.setattr(
        "amct_pytorch.common.models.llm.common.base.tqdm",
        lambda iterable, **kwargs: iterable,
    )
    monkeypatch.setattr(
        "amct_pytorch.common.models.llm.common.base.save_ptq_inps",
        lambda *a, **kw: None,
    )
    save_captured = {}

    def _fake_save_kwargs(pos_ids, pos_embs, attn_mask, data_dir):
        save_captured[CALLED] = True
        save_captured["data_dir"] = data_dir

    monkeypatch.setattr(
        "amct_pytorch.common.models.llm.common.base.save_ptq_kwargs",
        _fake_save_kwargs,
    )

    args = SimpleNamespace(
        model=str(model_dir),
        quant_target=[QUANT_TARGET_MLP],
        device="cpu",
        data_dir=str(tmp_path),
        output_dir=str(tmp_path),
        quant_dtype="int8",
        w_bits=8,
        a_bits=8,
        algos=[],
        cali_bsz=2,
        moe_mlp_param_dir="",
    )

    model = Qwen3(args)

    def float_model():
        return model.empty_weights_model()

    model.float_model = float_model

    batch, seq_len = 2, 4
    samples = [torch.randint(0, config.vocab_size, (batch, seq_len))]

    model.do_embedding_forward(samples, hook_name="test_hook")

    assert save_captured.get(CALLED) is True
    assert save_captured.get("data_dir") == str(tmp_path)


def test_do_block_forward_with_quant_block_sets_quant_state(monkeypatch):
    """do_block_forward calls set_model_weight/act_quant_state when use_quant_block=True (lines 194-196)."""
    from amct_pytorch.common.models.llm.qwen.qwen3.qwen3 import Qwen3

    model_dir, config, weight_map = _make_tiny_safetensors_model_dir()
    _mock_hf_for_safetensors_test(monkeypatch, model_dir, config)
    monkeypatch.setattr(
        "amct_pytorch.common.models.llm.common.base.get_weight_mappings",
        lambda path: weight_map,
    )

    weight_state = {}
    act_state = {}
    monkeypatch.setattr(
        "amct_pytorch.common.models.llm.common.quant_apply.set_model_weight_quant_state",
        lambda mod, flag: weight_state.update({CALLED: True, FLAG: flag}),
    )
    monkeypatch.setattr(
        "amct_pytorch.common.models.llm.common.quant_apply.set_model_act_quant_state",
        lambda mod, flag: act_state.update({CALLED: True, FLAG: flag}),
    )
    monkeypatch.setattr(
        "amct_pytorch.common.models.llm.common.base.save_ptq_inps",
        lambda *a, **kw: None,
    )

    args = SimpleNamespace(
        model=str(model_dir),
        quant_target=[QUANT_TARGET_MLP],
        device="cpu",
        data_dir="/tmp/fake",
        output_dir="/tmp/fake",
        quant_dtype="int8",
        w_bits=8,
        a_bits=8,
        algos=[],
        cali_bsz=2,
        moe_mlp_param_dir="",
    )

    model = Qwen3(args)

    class _FakeBlock(nn.Module):
        def forward(self, x, **kwargs):
            return x

    fake_block = _FakeBlock()
    monkeypatch.setattr(model, "_build_block_for_forward", lambda idx, use_quant_block: fake_block)

    batch, seq_len, hidden_size = 2, 4, 8
    samples = [torch.randn(batch, seq_len, hidden_size)]

    model.do_block_forward(0, samples, use_quant_block=True, hook_name="some_hook")

    assert weight_state.get(CALLED) is True
    assert not weight_state.get(FLAG)
    assert act_state.get(CALLED) is True
    assert not act_state.get(FLAG)


def test_do_block_forward_quant_eval_mode(monkeypatch):
    """do_block_forward sets QuantLinear._eval_mode when use_quant_block=True and hook_name is None (lines 200-204)."""
    from amct_pytorch.common.models.llm.qwen.qwen3.qwen3 import Qwen3

    model_dir, config, weight_map = _make_tiny_safetensors_model_dir()
    _mock_hf_for_safetensors_test(monkeypatch, model_dir, config)
    monkeypatch.setattr(
        "amct_pytorch.common.models.llm.common.base.get_weight_mappings",
        lambda path: weight_map,
    )
    monkeypatch.setattr(
        "amct_pytorch.common.models.llm.common.quant_apply.set_model_weight_quant_state",
        lambda mod, flag: None,
    )
    monkeypatch.setattr(
        "amct_pytorch.common.models.llm.common.quant_apply.set_model_act_quant_state",
        lambda mod, flag: None,
    )

    args = SimpleNamespace(
        model=str(model_dir),
        quant_target=[QUANT_TARGET_MLP],
        device="cpu",
        data_dir="/tmp/fake",
        output_dir="/tmp/fake",
        quant_dtype="int8",
        w_bits=8,
        a_bits=8,
        algos=[],
        cali_bsz=2,
        moe_mlp_param_dir="",
    )

    model = Qwen3(args)

    ql = QuantLinear.__new__(QuantLinear)
    ql.eval_mode = False
    ql.cached_eval_weight = "stale"
    ql._parameters = {}
    ql._modules = {}
    ql._buffers = {}

    class _FakeBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_linear = ql

        def forward(self, x, **kwargs):
            return x

    fake_block = _FakeBlock()
    monkeypatch.setattr(model, "_build_block_for_forward", lambda idx, use_quant_block: fake_block)

    batch, seq_len, hidden_size = 2, 4, 8
    samples = [torch.randn(batch, seq_len, hidden_size)]

    model.do_block_forward(0, samples, use_quant_block=True, hook_name=None)

    assert ql.eval_mode is True
    assert ql.cached_eval_weight is None


def test_do_block_forward_with_hook_removal(monkeypatch):
    """do_block_forward calls hook.remove() when hook_name is not None (line 218)."""
    from amct_pytorch.common.models.llm.qwen.qwen3.qwen3 import Qwen3

    model_dir, config, weight_map = _make_tiny_safetensors_model_dir()
    _mock_hf_for_safetensors_test(monkeypatch, model_dir, config)
    monkeypatch.setattr(
        "amct_pytorch.common.models.llm.common.base.get_weight_mappings",
        lambda path: weight_map,
    )
    monkeypatch.setattr(
        "amct_pytorch.common.models.llm.common.quant_apply.set_model_weight_quant_state",
        lambda mod, flag: None,
    )
    monkeypatch.setattr(
        "amct_pytorch.common.models.llm.common.quant_apply.set_model_act_quant_state",
        lambda mod, flag: None,
    )

    mock_hook = MagicMock()
    hooks_list = []

    def _fake_register_hooks(block, target_name, hooks, act_stat):
        hooks.append(mock_hook)

    monkeypatch.setattr(
        "amct_pytorch.common.models.llm.common.base.register_forward_hooks",
        _fake_register_hooks,
    )
    monkeypatch.setattr(
        "amct_pytorch.common.models.llm.common.base.save_ptq_inps",
        lambda *a, **kw: None,
    )

    args = SimpleNamespace(
        model=str(model_dir),
        quant_target=[QUANT_TARGET_MLP],
        device="cpu",
        data_dir="/tmp/fake",
        output_dir="/tmp/fake",
        quant_dtype="int8",
        w_bits=8,
        a_bits=8,
        algos=[],
        cali_bsz=2,
        moe_mlp_param_dir="",
    )

    model = Qwen3(args)

    class _FakeBlock(nn.Module):
        def forward(self, x, **kwargs):
            return x

    fake_block = _FakeBlock()
    monkeypatch.setattr(model, "_build_block_for_forward", lambda idx, use_quant_block: fake_block)

    batch, seq_len, hidden_size = 2, 4, 8
    samples = [torch.randn(batch, seq_len, hidden_size)]

    model.do_block_forward(0, samples, use_quant_block=True, hook_name="test_hook")

    mock_hook.remove.assert_called_once()


def test_do_head_forward_runs_norm_and_lm_head(monkeypatch):
    from amct_pytorch.common.models.llm.qwen.qwen3.qwen3 import Qwen3

    model_dir, config, weight_map = _make_tiny_safetensors_model_dir()
    _mock_hf_for_safetensors_test(monkeypatch, model_dir, config)
    monkeypatch.setattr(
        "amct_pytorch.common.models.llm.common.base.get_weight_mappings",
        lambda path: weight_map,
    )

    args = SimpleNamespace(
        model=str(model_dir),
        quant_target=[QUANT_TARGET_MLP],
        device="cpu",
        data_dir="/tmp/fake",
        output_dir="/tmp/fake",
        quant_dtype="int8",
        w_bits=8,
        a_bits=8,
        algos=[],
        cali_bsz=2,
        moe_mlp_param_dir="",
    )

    model = Qwen3(args)

    def float_model():
        return model.empty_weights_model()

    model.float_model = float_model
    model.load_embed_state_dict()

    batch, seq_len, hidden_size = 2, 4, 8
    model.position_embeddings = _compute_position_embeddings(config, batch, seq_len)
    samples = [torch.randn(batch, seq_len, hidden_size, dtype=torch.bfloat16)]
    inter_outs = model.do_block_forward(0, samples, hook_name=None)

    preds = model.do_head_forward(inter_outs)

    assert len(preds) == 1
    assert preds[0].shape == (batch, seq_len - 1, config.vocab_size)


def test_init_cls_does_nothing():
    model = _StubModel()
    result = model.init_cls()
    assert result is None


def test_get_layer_weight_prefix_returns_none_in_base():
    model = _StubModel()
    prefix = model.get_layer_weight_prefix(0)
    assert prefix is None


def test_load_unit_inputs_when_not_cached_returns_default():
    model = _StubModel()
    unit = make_ptq_unit("mlp", "mlp", 0, nn.Linear(4, 8))
    result = model.load_unit_inputs("/nonexistent", unit)
    assert result is not None


# ---- block_size (new in diff) --------------------------------------------


def test_block_size_returns_32():
    w = torch.randn(4, 4)
    assert BaseModel.block_size(w) == 32


def test_generate_tensorwise_quant_layers_raises_not_implemented():
    stub = _StubModel()
    with pytest.raises(NotImplementedError):
        stub.generate_tensorwise_quant_layers()


def test_generate_tensorwise_ignore_layers_raises_not_implemented():
    stub = _StubModel()
    with pytest.raises(NotImplementedError):
        stub.generate_tensorwise_ignore_layers()

