# Copyright (c) 2026 Huawei Technologies Co., Ltd.
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

import json
from types import SimpleNamespace

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from amct_pytorch.common.models.llm.common.deployment.deploy_selection import (
    DeploySelection,
    QuantLayerRule,
    QuantSpec,
)
from amct_pytorch.common.models.llm.kimi.kimi_k3.kimi_k3 import KimiK3


Q_A = "language_model.model.layers.0.self_attn.q_a_proj"
Q_B = "language_model.model.layers.0.self_attn.q_b_proj"
EXPERT = "language_model.model.layers.1.block_sparse_moe.experts.0.w1"


def _args(model, output, **overrides):
    values = {
        "model": str(model),
        "output_dir": str(output),
        "granularity": "tensor",
        "deploy_platform": "ascend950",
        "quant_dtype": None,
        "device": "cpu",
        "algos": [],
        "attn_linear_param_dir": "",
        "attn_cache_param_dir": "",
        "moe_mlp_param_dir": "",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _write_model(tmp_path, *, config=None, tensors=None):
    model_dir = tmp_path / "model"
    output_dir = tmp_path / "output"
    model_dir.mkdir()
    model_config = config or {
        "model_type": "kimi_k3",
        "text_config": {"num_experts": 1},
    }
    (model_dir / "config.json").write_text(json.dumps(model_config), encoding="utf-8")
    if tensors is None:
        tensors = {"placeholder.weight": torch.ones(1, 1, dtype=torch.bfloat16)}
    save_file(tensors, str(model_dir / "model.safetensors"))
    return model_dir, output_dir


def _bare_model(tmp_path, tensors):
    model_dir, output_dir = _write_model(tmp_path, tensors=tensors)
    output_dir.mkdir()
    model = KimiK3.__new__(KimiK3)
    model.args = _args(model_dir, output_dir)
    model.model_path = model_dir
    model.output_dir = output_dir
    model.config = {"model_type": "kimi_k3", "text_config": {"num_experts": 1}}
    model.safetensors_files = frozenset({(model_dir / "model.safetensors").resolve()})
    model.processed_modules = set()
    model._planned_specs = {}
    model.output_tensor_bytes = 0
    return model


def test_init_and_deploy_args(tmp_path):
    model_dir, output_dir = _write_model(
        tmp_path, tensors={"other.weight": torch.ones(1, 1, dtype=torch.bfloat16)}
    )
    args = _args(model_dir, output_dir)
    model = KimiK3(args)

    assert model.config["model_type"] == "kimi_k3"
    assert args.quant_dtype == "mxfp"
    assert model.safetensors_files


@pytest.mark.parametrize(
    "overrides, error",
    [
        ({"granularity": "block"}, NotImplementedError),
        ({"deploy_platform": "A3"}, ValueError),
        ({"quant_dtype": "int"}, ValueError),
        ({"algos": ["lwc"]}, ValueError),
    ],
)
def test_validate_deploy_args_rejects_unsupported_options(tmp_path, overrides, error):
    args = _args(tmp_path / "model", tmp_path / "output", **overrides)
    with pytest.raises(error):
        KimiK3.validate_deploy_args(args)


def test_validate_deploy_source_and_capabilities(tmp_path):
    model_dir, output_dir = _write_model(tmp_path)
    args = _args(model_dir, output_dir)
    assert KimiK3.validate_deploy_source(args)
    assert KimiK3.get_deploy_capabilities() == {"ascend950": frozenset({"mxfp"})}

    (model_dir / "quant_model_description.json").write_text("{}", encoding="utf-8")
    assert not KimiK3.validate_deploy_source(args)

    missing = tmp_path / "missing"
    assert not KimiK3.validate_deploy_source(_args(missing, output_dir))


def test_module_helpers_and_candidates():
    assert KimiK3._canonical_module(Q_A + ".weight") == Q_A
    assert KimiK3._canonical_module(EXPERT + ".weight_packed") == EXPERT
    assert KimiK3._canonical_module(EXPERT + ".weight_scale") == EXPERT
    assert KimiK3._canonical_module("bias") is None
    assert KimiK3._canonical_weight_modules(
        {Q_A + ".weight", EXPERT + ".weight_packed"}
    ) == {
        Q_A,
        EXPERT,
    }
    assert KimiK3._is_expert(EXPERT)
    assert not KimiK3._is_expert(Q_A)
    assert KimiK3._is_supported_module(EXPERT)
    assert KimiK3._is_supported_module(Q_A)
    assert KimiK3._is_supported_module("language_model.model.layers.3.self_attn.o_proj")
    assert KimiK3._is_supported_module(
        "language_model.model.layers.92.self_attn.o_proj"
    )
    assert not KimiK3._is_supported_module(
        "language_model.model.layers.4.self_attn.o_proj"
    )
    assert not KimiK3._is_supported_module("other.weight")
    adapter = KimiK3.__new__(KimiK3)
    assert adapter.get_deploy_candidates({Q_A, "other"}) == {Q_A}


def test_validate_selection_rejects_unsupported_and_partial_experts(tmp_path):
    model_dir, output_dir = _write_model(tmp_path)
    model = KimiK3(_args(model_dir, output_dir))
    with pytest.raises(ValueError, match="Unsupported Kimi-K3"):
        model.validate_deploy_selection({"language_model.model.layers.0.mlp.down_proj"})
    with pytest.raises(ValueError, match="all native MXFP4"):
        model.validate_deploy_selection({EXPERT})
    model.validate_deploy_selection({Q_A})
    assert model._num_experts() == 1


def test_validate_selection_accepts_complete_native_experts(tmp_path):
    model_dir, output_dir = _write_model(tmp_path)
    model = KimiK3(_args(model_dir, output_dir))
    selected = {
        f"language_model.model.layers.{layer}.block_sparse_moe.experts.0.w{projection}"
        for layer in range(1, 93)
        for projection in (1, 2, 3)
    }

    model.validate_deploy_selection(selected)


def test_prepare_plan_tracks_ignored_and_unselected_modules(tmp_path):
    tensors = {
        Q_A + ".weight": torch.ones(2, 32, dtype=torch.bfloat16),
        Q_B + ".weight": torch.ones(2, 32, dtype=torch.bfloat16),
    }
    model_dir, output_dir = _write_model(tmp_path, tensors=tensors)
    selection = DeploySelection(
        (QuantLayerRule(Q_A, QuantSpec("mxfp", 8, 8)),),
        (Q_B,),
    )
    model = KimiK3(_args(model_dir, output_dir, deploy_selection=selection))
    model.args.deploy_selection = selection

    plan = model.prepare_tensor_deploy_plan(
        {name: "model.safetensors" for name in tensors}
    )

    assert plan.selected_modules == {Q_A}
    assert plan.ignored_modules == {Q_B}
    assert plan.unselected_weight_modules == {Q_B}
    assert plan.selected_weight_keys == {Q_A + ".weight"}


def test_prepare_plan_rejects_selected_module_missing_from_inventory(
    tmp_path, monkeypatch
):
    model_dir, output_dir = _write_model(tmp_path)
    selection = DeploySelection((QuantLayerRule(Q_A, QuantSpec("mxfp", 8, 8)),))
    model = KimiK3(_args(model_dir, output_dir, deploy_selection=selection))
    model.args.deploy_selection = selection
    monkeypatch.setattr(
        "amct_pytorch.common.models.llm.kimi.kimi_k3.kimi_k3.match_deploy_modules",
        lambda names, current: ({Q_A}, {Q_A: 1}),
    )

    with pytest.raises(ValueError, match="absent from checkpoint"):
        model.prepare_tensor_deploy_plan({"placeholder.weight": "model.safetensors"})


def test_prepare_plan_tracks_mtp_and_unmatched_ignore(tmp_path):
    mtp = "mtp.layers.0.self_attn.q_a_proj"
    tensors = {
        Q_A + ".weight": torch.ones(2, 32, dtype=torch.bfloat16),
        mtp + ".weight": torch.ones(2, 32, dtype=torch.bfloat16),
    }
    model_dir, output_dir = _write_model(tmp_path, tensors=tensors)
    selection = DeploySelection(
        (QuantLayerRule(Q_A, QuantSpec("mxfp", 8, 8)),),
        (mtp, "missing.*"),
    )
    model = KimiK3(_args(model_dir, output_dir, deploy_selection=selection))
    model.args.deploy_selection = selection

    plan = model.prepare_tensor_deploy_plan(
        {name: "model.safetensors" for name in tensors}
    )

    assert plan.mtp.modules == frozenset({mtp})
    assert plan.mtp.ignored == frozenset({mtp})
    assert plan.unmatched_ignore_patterns == {"missing.*"}


def test_prepare_plan_rejects_unselected_native_expert(tmp_path):
    tensors = {
        Q_A + ".weight": torch.ones(2, 32, dtype=torch.bfloat16),
        EXPERT + ".weight_packed": torch.ones(2, 16, dtype=torch.uint8),
        EXPERT + ".weight_scale": torch.ones(2, 1, dtype=torch.uint8),
    }
    model_dir, output_dir = _write_model(tmp_path, tensors=tensors)
    selection = DeploySelection((QuantLayerRule(Q_A, QuantSpec("mxfp", 8, 8)),))
    model = KimiK3(_args(model_dir, output_dir, deploy_selection=selection))
    model.args.deploy_selection = selection

    with pytest.raises(ValueError, match="native Kimi-K3 routed expert"):
        model.prepare_tensor_deploy_plan(
            {name: "model.safetensors" for name in tensors}
        )


def test_convert_tensorwise_shard_quantizes_bf16_and_renames_expert(tmp_path):
    tensors = {
        Q_A + ".weight": torch.ones(2, 32, dtype=torch.bfloat16),
        EXPERT + ".weight_packed": torch.ones(2, 16, dtype=torch.uint8),
        EXPERT + ".weight_scale": torch.ones(2, 1, dtype=torch.uint8),
        "other.weight": torch.ones(1, 1, dtype=torch.bfloat16),
    }
    model = _bare_model(tmp_path, tensors)
    specs = {
        Q_A: QuantSpec("mxfp", 8, 8),
        EXPERT: QuantSpec("mxfp", 4, 8),
    }
    model._planned_specs = specs
    routing = model.convert_tensorwise_shard(
        "model.safetensors",
        model.model_path,
        {name: "model.safetensors" for name in tensors},
        {Q_A: 8, EXPERT: 4},
        {},
    )

    assert set(routing) == {
        Q_A + ".weight",
        Q_A + ".weight_scale",
        EXPERT + ".weight",
        EXPERT + ".weight_scale",
        "other.weight",
    }
    assert model.processed_modules == {Q_A, EXPERT}
    with safe_open(
        str(model.output_dir / "model.safetensors"), framework="pt"
    ) as shard:
        assert shard.get_slice(Q_A + ".weight").get_dtype() == "F8_E4M3"
        assert shard.get_slice(EXPERT + ".weight").get_dtype() == "U8"
        assert EXPERT + ".weight_packed" not in shard.keys()


def test_convert_tensorwise_shard_uses_planned_specs_for_bit_map(tmp_path):
    tensors = {
        Q_A + ".weight": torch.ones(2, 32, dtype=torch.bfloat16),
    }
    model = _bare_model(tmp_path, tensors)
    model._planned_specs = {Q_A: QuantSpec("mxfp", 8, 8)}
    routing = model.convert_tensorwise_shard(
        "model.safetensors",
        model.model_path,
        {name: "model.safetensors" for name in tensors},
        {Q_A: 8},
        {},
    )

    assert set(routing) == {Q_A + ".weight", Q_A + ".weight_scale"}


@pytest.mark.parametrize(
    "suffix, specs, error",
    [
        (".weight_packed", {}, "missing from deployment plan"),
        (".weight_packed", {EXPERT: QuantSpec("mxfp", 8, 8)}, "must use mxfp W4A8"),
        (".weight_scale", {}, "scale is missing from deployment plan"),
        (".weight", {Q_A: QuantSpec("mxfp", 4, 8)}, "must use mxfp W8A8"),
    ],
)
def test_convert_tensorwise_shard_rejects_invalid_plan(tmp_path, suffix, specs, error):
    module = EXPERT if suffix != ".weight" else Q_A
    dtype = torch.uint8 if suffix != ".weight" else torch.bfloat16
    shape = (2, 16) if suffix == ".weight_packed" else (2, 32)
    tensors = {module + suffix: torch.ones(shape, dtype=dtype)}
    if suffix == ".weight_scale":
        tensors[module + ".weight_scale"] = torch.ones(2, 1, dtype=torch.uint8)
    model = _bare_model(tmp_path, tensors)

    with pytest.raises(ValueError, match=error):
        model.convert_tensorwise_shard(
            "model.safetensors",
            model.model_path,
            {name: "model.safetensors" for name in tensors},
            specs,
            {},
        )


def test_validate_result_refresh_config_and_description(tmp_path):
    model_dir, output_dir = _write_model(tmp_path)
    output_dir.mkdir()
    model = KimiK3.__new__(KimiK3)
    model.args = _args(model_dir, output_dir)
    model.config = {
        "model_type": "kimi_k3",
        "quantization_config": {"quant_method": "old"},
        "text_config": {"quantization_config": {"quant_method": "old"}},
    }
    model.output_dir = output_dir
    model._planned_specs = {
        Q_A: QuantSpec("mxfp", 8, 8),
        EXPERT: QuantSpec("mxfp", 4, 8),
    }
    model.processed_modules = {Q_A, EXPERT}
    plan = SimpleNamespace(selected_modules=frozenset({Q_A, EXPERT}))
    assert model.validate_tensorwise_result(plan) == {Q_A, EXPERT}

    model.refresh_config([], tensor_plan=plan)
    config = json.loads((output_dir / "config.json").read_text(encoding="utf-8"))
    assert "quantization_config" not in config
    assert "quantization_config" not in config["text_config"]

    output_names = {
        Q_A + ".weight",
        Q_A + ".weight_scale",
        EXPERT + ".weight",
        EXPERT + ".weight_scale",
        "other.weight",
    }
    description = model.make_description(output_names, {Q_A, EXPERT})
    assert description["group_size"] == 32
    assert description[Q_A + ".weight"] == "W8A8_MXFP8"
    assert description[EXPERT + ".weight"] == "W4A8_MXFP"
    assert description["other.weight"] == "FLOAT"

    with pytest.raises(ValueError, match="missing"):
        model.make_description({Q_A + ".weight"}, {Q_A})

    model.processed_modules = set()
    with pytest.raises(ValueError, match="Actual Kimi-K3"):
        model.validate_tensorwise_result(plan)
