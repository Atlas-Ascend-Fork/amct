# Copyright (c) 2026 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file, save_file

from amct_pytorch.common.models.llm.common.deploy_selection import DeploySelection
from amct_pytorch.common.models.llm.deepseek.deepseek_v4_1 import deepseekv4_1
from amct_pytorch.common.models.llm.deepseek.deepseek_v4_1.deepseekv4_1 import (
    BLOCK,
    DeepseekV41,
)
from amct_pytorch.quantization.bit_policy import BitPolicy

SHARD = "model.safetensors"
MODULE = "model.layers.0.mlp.down_proj"
MODULE_B = "model.layers.1.mlp.down_proj"


def _args(**overrides):
    values = {
        "model_name": "deepseek_v4_1",
        "model": "/fake/model",
        "output_dir": "/fake/output",
        "deploy_platform": "A3",
        "quant_dtype": "int",
        "granularity": "tensor",
        "deploy_format": "ascend",
        "bit_config": None,
        "quant_layers_config": None,
        "deploy_selection": DeploySelection((MODULE,)),
        "algos": [],
        "attn_linear_param_dir": None,
        "attn_cache_param_dir": None,
        "moe_mlp_param_dir": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _bare_model(tmp_path, *, platform="A3", selection=None):
    model_dir = tmp_path / "model"
    output_dir = tmp_path / "output"
    model_dir.mkdir()
    output_dir.mkdir()
    model = DeepseekV41.__new__(DeepseekV41)
    model.args = _args(
        model=str(model_dir),
        output_dir=str(output_dir),
        deploy_platform=platform,
        quant_dtype="int" if platform == "A3" else "mxfp",
        deploy_format="ascend" if platform == "A3" else "legacy",
        deploy_selection=selection or DeploySelection((MODULE,)),
    )
    model.model_path = model_dir
    model.output_dir = output_dir
    model.safetensors_files = frozenset()
    model.processed_modules = set()
    model.config = {
        "text_config": {
            "kv_source_layer_ids": [1],
            "index_source_layers": [2],
        }
    }
    return model


def _write_shard(model, tensors, filename=SHARD):
    path = model.model_path / filename
    save_file(tensors, str(path))
    model.safetensors_files = frozenset({path.resolve()})
    return path


@pytest.mark.parametrize(
    "platform,dtype",
    [("A3", "int"), ("ascend950", "mxfp")],
)
def test_validate_platform_accepts_supported_pairs(platform, dtype):
    assert (
        DeepseekV41._validate_platform(
            SimpleNamespace(deploy_platform=platform, quant_dtype=dtype)
        )
        == platform
    )


@pytest.mark.parametrize(
    "platform,dtype,message",
    [
        ("A2", "int", "A3 or ascend950"),
        ("A3", "mxfp", "requires --quant_dtype int"),
        ("ascend950", "int", "requires --quant_dtype mxfp"),
    ],
)
def test_validate_platform_rejects_unsupported_pairs(platform, dtype, message):
    with pytest.raises(ValueError, match=message):
        DeepseekV41._validate_platform(
            SimpleNamespace(deploy_platform=platform, quant_dtype=dtype)
        )


def test_validate_ascend_source_accepts_clean_official_fp8_config(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {"quantization_config": {"quant_method": "fp8"}, "is_rot_used": False}
        ),
        encoding="utf-8",
    )

    assert DeepseekV41.validate_ascend_source(_args(model=str(tmp_path))) is True

    (tmp_path / "rot.safetensors").touch()
    assert DeepseekV41.validate_ascend_source(_args(model=str(tmp_path))) is False


def test_validate_ascend_source_rejects_wrong_mode_and_invalid_config(tmp_path):
    assert (
        DeepseekV41.validate_ascend_source(
            _args(model=str(tmp_path), deploy_platform="ascend950", quant_dtype="mxfp")
        )
        is False
    )
    (tmp_path / "config.json").write_text("{", encoding="utf-8")
    assert DeepseekV41.validate_ascend_source(_args(model=str(tmp_path))) is False


@pytest.mark.parametrize(
    "overrides,message",
    [
        ({"granularity": "block"}, "tensor deployment only"),
        ({"deploy_format": "legacy"}, "A3 requires"),
        (
            {
                "deploy_platform": "ascend950",
                "quant_dtype": "mxfp",
                "deploy_format": "ascend",
            },
            "ascend950 requires",
        ),
        (
            {
                "deploy_platform": "ascend950",
                "quant_dtype": "mxfp",
                "deploy_format": "legacy",
                "bit_config": "bits.yaml",
            },
            "bit_config",
        ),
        ({"algos": ["lwc"]}, "algos"),
        ({"attn_linear_param_dir": "params"}, "attn_linear_param_dir"),
        ({"attn_cache_param_dir": "params"}, "attn_cache_param_dir"),
        ({"moe_mlp_param_dir": "params"}, "moe_mlp_param_dir"),
        ({"deploy_selection": None}, "quant_layers_config"),
    ],
)
def test_validate_deploy_args_rejects_incompatible_options(overrides, message):
    with pytest.raises((ValueError, NotImplementedError), match=message):
        DeepseekV41.validate_deploy_args(_args(**overrides))


def test_validate_deploy_args_loads_selection_and_sets_empty_bit_policy(tmp_path):
    selection_path = tmp_path / "layers.json"
    selection_path.write_text(
        json.dumps({"quant_layers": [MODULE], "ignore_layers": ["model.norm"]}),
        encoding="utf-8",
    )
    args = _args(quant_layers_config=str(selection_path), deploy_selection=None)

    DeepseekV41.validate_deploy_args(args)

    assert args.deploy_selection == DeploySelection((MODULE,), ("model.norm",))
    assert isinstance(args.bit_policy, BitPolicy)


def test_init_loads_config_and_checkpoint_inventory(tmp_path):
    model_dir = tmp_path / "model"
    output_dir = tmp_path / "output"
    model_dir.mkdir()
    (model_dir / "config.json").write_text('{"model_type": "deepseek_v4_1"}')
    save_file({f"{MODULE}.weight": torch.ones(2, 2)}, str(model_dir / SHARD))

    model = DeepseekV41(_args(model=str(model_dir), output_dir=str(output_dir)))

    assert model.config == {"model_type": "deepseek_v4_1"}
    assert model.safetensors_files == frozenset({(model_dir / SHARD).resolve()})
    assert model.processed_modules == set()


@pytest.mark.parametrize("output_kind", ["overlap", "nonempty"])
def test_init_rejects_unsafe_output_directories(tmp_path, output_kind):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}")
    save_file({"x": torch.ones(1)}, str(model_dir / SHARD))
    if output_kind == "overlap":
        output_dir = model_dir / "output"
        message = "must not overlap"
    else:
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        (output_dir / "existing.txt").write_text("data")
        message = "new or empty"

    with pytest.raises(ValueError, match=message):
        DeepseekV41(_args(model=str(model_dir), output_dir=str(output_dir)))


@pytest.mark.parametrize(
    "weight_dtype,weight_shape,scale_dtype,scale_shape,expected",
    [
        ("F8_E4M3", (32, 64), "F8_E8M0", (1, 2), True),
        ("F16", (32, 64), "F8_E8M0", (1, 2), False),
        ("F8_E4M3", (31, 64), "F8_E8M0", (1, 2), False),
        ("F8_E4M3", (32, 64), "F8_E8M0", (1, 1), False),
    ],
)
def test_is_block_fp8_requires_matching_32x32_layout(
    weight_dtype, weight_shape, scale_dtype, scale_shape, expected
):
    assert (
        DeepseekV41._is_block_fp8(weight_dtype, weight_shape, scale_dtype, scale_shape)
        is expected
    )


def test_prepare_a3_plan_resolves_selected_ignored_and_mtp_modules(tmp_path):
    ignored = "model.layers.0.mlp.gate_proj"
    mtp = "mtp.layers.0.eh_proj"
    selection = DeploySelection((MODULE,), (ignored, "missing.*"))
    model = _bare_model(tmp_path, selection=selection)
    tensors = {
        f"{MODULE}.weight": torch.ones(2, 2),
        f"{ignored}.weight": torch.ones(2, 2),
        f"{mtp}.weight": torch.ones(2, 2),
    }
    _write_shard(model, tensors)
    weight_map = dict.fromkeys(tensors, SHARD)

    plan = model.prepare_tensor_deploy_plan(weight_map)

    assert plan.selected_modules == {MODULE}
    assert plan.selected_weight_keys == {f"{MODULE}.weight"}
    assert plan.module_bits == {MODULE: (8, 8)}
    assert plan.ignored_modules == {ignored}
    assert plan.unselected_weight_modules == {ignored, mtp}
    assert plan.mtp.modules == {mtp}
    assert plan.unmatched_ignore_patterns == {"missing.*"}


def test_prepare_a3_plan_rejects_non_matrix_selection(tmp_path):
    model = _bare_model(tmp_path)
    _write_shard(model, {f"{MODULE}.weight": torch.ones(2)})

    with pytest.raises(ValueError, match="nonempty 2-D"):
        model.prepare_tensor_deploy_plan({f"{MODULE}.weight": SHARD})


def test_prepare_950_plan_rejects_unselected_fp8_candidate(tmp_path, monkeypatch):
    selected = "model.layers.0.mlp.gate_proj"
    model = _bare_model(
        tmp_path,
        platform="ascend950",
        selection=DeploySelection((selected,)),
    )
    scale = torch.ones((1, 1), dtype=torch.uint8).view(torch.float8_e8m0fnu)
    tensors = {
        f"{MODULE}.weight": torch.ones((32, 32), dtype=torch.float8_e4m3fn),
        f"{MODULE}.scale": scale,
        f"{selected}.weight": torch.ones(2, 2),
    }
    _write_shard(model, tensors)
    monkeypatch.setattr(model, "_make_legacy_quant_config", dict)

    with pytest.raises(ValueError, match="not selected"):
        model.prepare_tensor_deploy_plan(dict.fromkeys(tensors, SHARD))


def test_convert_a3_shard_dequantizes_then_exports_selected_weight(
    tmp_path, monkeypatch
):
    model = _bare_model(tmp_path)
    scale_shard = "scale.safetensors"
    weight_name = f"{MODULE}.weight"
    scale_name = f"{MODULE}.scale"
    other_scale = "metadata.scale"
    _write_shard(
        model,
        {
            weight_name: torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            "model.norm.weight": torch.ones(2),
            other_scale: torch.ones(1),
        },
    )
    scale_path = model.model_path / scale_shard
    save_file({scale_name: torch.ones(1, 1)}, str(scale_path))
    model.safetensors_files = frozenset(
        {model.model_path.joinpath(SHARD).resolve(), scale_path.resolve()}
    )
    calls = []

    def fake_convert(weight, *args, **kwargs):
        calls.append((args, kwargs))
        return weight.to(torch.bfloat16)

    monkeypatch.setattr(deepseekv4_1, "convert_state_dict", fake_convert)
    weight_map = {
        weight_name: SHARD,
        scale_name: scale_shard,
        "model.norm.weight": SHARD,
        other_scale: SHARD,
    }

    routing = model.convert_tensorwise_shard(
        SHARD, model.model_path, weight_map, {MODULE: 8}, {}
    )
    output = load_file(str(model.output_dir / SHARD))

    assert set(output) == {
        weight_name,
        f"{MODULE}.weight_scale",
        f"{MODULE}.weight_offset",
        "model.norm.weight",
        other_scale,
    }
    assert output[weight_name].dtype == torch.int8
    assert output["model.norm.weight"].dtype == torch.float32
    assert routing == dict.fromkeys(output, SHARD)
    assert model.processed_modules == {MODULE}
    assert model.output_tensor_bytes == sum(
        tensor.numel() * tensor.element_size() for tensor in output.values()
    )
    assert len(calls) == 1


def test_convert_a3_shard_rejects_integer_weight_without_scale(tmp_path):
    model = _bare_model(tmp_path)
    weight_name = f"{MODULE}.weight"
    _write_shard(model, {weight_name: torch.ones(2, 2, dtype=torch.int8)})

    with pytest.raises(ValueError, match="Missing source scale"):
        model.convert_tensorwise_shard(
            SHARD,
            model.model_path,
            {weight_name: SHARD},
            {MODULE: 8},
            {},
        )


def test_convert_a3_shard_rejects_scale_missing_from_mapped_shard(tmp_path):
    model = _bare_model(tmp_path)
    weight_name = f"{MODULE}.weight"
    scale_name = f"{MODULE}.scale"
    _write_shard(model, {weight_name: torch.ones(2, 2)})
    scale_path = model.model_path / "scale.safetensors"
    save_file({"other": torch.ones(1)}, str(scale_path))
    model.safetensors_files = frozenset(
        {model.model_path.joinpath(SHARD).resolve(), scale_path.resolve()}
    )

    with pytest.raises(ValueError, match="Missing source scale"):
        model.convert_tensorwise_shard(
            SHARD,
            model.model_path,
            {weight_name: SHARD, scale_name: "scale.safetensors"},
            {MODULE: 8},
            {},
        )


def test_convert_a3_shard_cache_does_not_accumulate_across_shards(tmp_path):
    """Regression for issue #246: peak host memory stays at one shard's set.

    Both scales live in a single shared shard, so the second source shard must
    still be able to re-read them after the first shard's cache is dropped.
    """
    model = _bare_model(tmp_path, selection=DeploySelection((MODULE, MODULE_B)))
    shards = ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors")
    scale_shard = "scales.safetensors"
    source = (((torch.arange(BLOCK * BLOCK, dtype=torch.float32) % 7) - 3) / 4).reshape(
        BLOCK, BLOCK
    )
    scales = {}
    weight_map = {}
    paths = []
    for shard, module in zip(shards, (MODULE, MODULE_B)):
        weight_name = f"{module}.weight"
        scale_name = f"{module}.scale"
        save_file(
            {weight_name: source.to(torch.float8_e4m3fn)},
            str(model.model_path / shard),
        )
        scales[scale_name] = torch.full((1, 1), 127, dtype=torch.uint8).view(
            torch.float8_e8m0fnu
        )
        weight_map[weight_name] = shard
        weight_map[scale_name] = scale_shard
        paths.append(model.model_path / shard)
    save_file(scales, str(model.model_path / scale_shard))
    paths.append(model.model_path / scale_shard)
    model.safetensors_files = frozenset(path.resolve() for path in paths)

    loaded_files = {}
    quant_layers = dict.fromkeys((MODULE, MODULE_B), 8)
    for shard, module in zip(shards, (MODULE, MODULE_B)):
        routing = model.convert_tensorwise_shard(
            shard, model.model_path, weight_map, quant_layers, loaded_files
        )
        # The dict shared by the whole export must never retain processed
        # source shards or the shards read only for their scales.
        assert not loaded_files, sorted(loaded_files)

        output = load_file(str(model.output_dir / shard))
        assert set(routing) == set(output)
        assert output[f"{module}.weight"].dtype == torch.int8
        assert output[f"{module}.weight_scale"].dtype == torch.bfloat16
        assert output[f"{module}.weight_offset"].dtype == torch.bfloat16
        assert output[f"{module}.weight_scale"].shape == (BLOCK, 1)

    assert model.processed_modules == {MODULE, MODULE_B}


def test_convert_a3_shard_reads_scales_without_loading_their_shards(
    tmp_path, monkeypatch
):
    """Scales are read tensor by tensor and their shard is never materialised."""
    model = _bare_model(tmp_path, selection=DeploySelection((MODULE, MODULE_B)))
    scale_shard = "scales.safetensors"
    weight_map = {}
    paths = []
    for module in (MODULE, MODULE_B):
        save_file(
            {f"{module}.weight": torch.ones(BLOCK, BLOCK, dtype=torch.float8_e4m3fn)},
            str(model.model_path / f"{module}.safetensors"),
        )
        weight_map[f"{module}.weight"] = f"{module}.safetensors"
        weight_map[f"{module}.scale"] = scale_shard
        paths.append(model.model_path / f"{module}.safetensors")
    save_file(
        {
            f"{module}.scale": torch.full((1, 1), 127, dtype=torch.uint8).view(
                torch.float8_e8m0fnu
            )
            for module in (MODULE, MODULE_B)
        },
        str(model.model_path / scale_shard),
    )
    paths.append(model.model_path / scale_shard)
    model.safetensors_files = frozenset(path.resolve() for path in paths)

    loaded = []
    real_load_file = deepseekv4_1.load_file

    def counting_load_file(path, **kwargs):
        loaded.append(Path(path).name)
        return real_load_file(path, **kwargs)

    monkeypatch.setattr(deepseekv4_1, "load_file", counting_load_file)
    quant_layers = dict.fromkeys((MODULE, MODULE_B), 8)
    for module in (MODULE, MODULE_B):
        model.convert_tensorwise_shard(
            f"{module}.safetensors",
            model.model_path,
            weight_map,
            quant_layers,
            {},
        )

    # Only the two source shards are materialised; the shared scale shard is
    # read through safe_open handles, one scale tensor at a time.
    assert loaded == [f"{MODULE}.safetensors", f"{MODULE_B}.safetensors"]
    assert model.processed_modules == {MODULE, MODULE_B}


def _write_scale_layout(model, modules, scale_shard_of):
    """Write one source shard plus per-module scale shards; return weight_map."""
    source = (
        (((torch.arange(BLOCK * BLOCK, dtype=torch.float32) % 7) - 3) / 4)
        .reshape(BLOCK, BLOCK)
        .to(torch.float8_e4m3fn)
    )
    save_file(
        # Distinct storages: safetensors rejects shared tensors on save.
        {f"{module}.weight": source.clone() for module in modules},
        str(model.model_path / SHARD),
    )
    buckets = {}
    for module in modules:
        shard = scale_shard_of[module]
        buckets.setdefault(shard, {})[f"{module}.scale"] = torch.full(
            (1, 1), 127, dtype=torch.uint8
        ).view(torch.float8_e8m0fnu)
    paths = [model.model_path / SHARD]
    for name, tensors in sorted(buckets.items()):
        save_file(tensors, str(model.model_path / name))
        paths.append(model.model_path / name)
    model.safetensors_files = frozenset(path.resolve() for path in paths)
    weight_map = {}
    for module in modules:
        weight_map[f"{module}.weight"] = SHARD
        weight_map[f"{module}.scale"] = scale_shard_of[module]
    return weight_map


def test_convert_a3_shard_evicts_scale_handles_beyond_the_bound(tmp_path, monkeypatch):
    """The MAX_OPEN_SCALE_FILES eviction path keeps the handle cache in bound."""
    module_c = "model.layers.2.mlp.down_proj"
    modules = (MODULE, MODULE_B, module_c)
    model = _bare_model(tmp_path, selection=DeploySelection(modules))
    # The third scale lives in the first scale shard again, so it must still be
    # readable after its handle was evicted by the bound.
    weight_map = _write_scale_layout(
        model,
        modules,
        {
            MODULE: "scales-a.safetensors",
            MODULE_B: "scales-b.safetensors",
            module_c: "scales-a.safetensors",
        },
    )

    monkeypatch.setattr(deepseekv4_1, "MAX_OPEN_SCALE_FILES", 1)
    real_read = deepseekv4_1.DeepseekV41._read_source_scale
    peak = {"handles": 0}

    def tracking_read(self, model_dir, scale_file, scale_name, open_files):
        scale = real_read(self, model_dir, scale_file, scale_name, open_files)
        peak["handles"] = max(peak["handles"], len(open_files))
        return scale

    monkeypatch.setattr(deepseekv4_1.DeepseekV41, "_read_source_scale", tracking_read)

    routing = model.convert_tensorwise_shard(
        SHARD, model.model_path, weight_map, dict.fromkeys(modules, 8), {}
    )

    output = load_file(str(model.output_dir / SHARD))
    assert set(routing) == set(output)
    for module in modules:
        assert output[f"{module}.weight"].dtype == torch.int8
        assert output[f"{module}.weight_scale"].dtype == torch.bfloat16
        assert output[f"{module}.weight_scale"].shape == (BLOCK, 1)
    assert model.processed_modules == set(modules)
    assert peak["handles"] <= 1, peak


def test_read_source_scale_caches_shard_keys_per_handle(tmp_path, monkeypatch):
    """``handle.keys()`` is built once per scale shard, not once per weight."""
    model = _bare_model(tmp_path, selection=DeploySelection((MODULE, MODULE_B)))
    weight_map = _write_scale_layout(
        model,
        (MODULE, MODULE_B),
        {MODULE: "scales-a.safetensors", MODULE_B: "scales-a.safetensors"},
    )

    real_safe_open = deepseekv4_1.safe_open
    calls = {"keys": 0, "opens": 0}

    class CountingHandle:
        def __init__(self, handle):
            self._handle = handle

        def keys(self):
            calls["keys"] += 1
            return self._handle.keys()

        def get_tensor(self, name):
            return self._handle.get_tensor(name)

    def counting_safe_open(*args, **kwargs):
        calls["opens"] += 1
        return CountingHandle(real_safe_open(*args, **kwargs))

    monkeypatch.setattr(deepseekv4_1, "safe_open", counting_safe_open)

    model.convert_tensorwise_shard(
        SHARD,
        model.model_path,
        weight_map,
        dict.fromkeys((MODULE, MODULE_B), 8),
        {},
    )

    # Both weights read their scale from the same shard: one open, one keys().
    assert calls == {"keys": 1, "opens": 1}, calls
    assert model.processed_modules == {MODULE, MODULE_B}


def test_prepare_a3_scale_expands_e8m0_rows_for_block_fp8():
    weight = torch.ones((32, 64), dtype=torch.float8_e4m3fn)
    scale = torch.ones((1, 2), dtype=torch.uint8).view(torch.float8_e8m0fnu)

    result = DeepseekV41._prepare_a3_scale(weight, scale)

    assert result.dtype == torch.uint8
    assert result.shape == (32, 2)
    assert torch.equal(result, torch.ones(32, 2, dtype=torch.uint8))


def test_convert_950_shard_expands_only_selected_fp8_scale(tmp_path):
    model = _bare_model(tmp_path, platform="ascend950")
    scale = torch.ones((1, 1), dtype=torch.uint8).view(torch.float8_e8m0fnu)
    tensors = {
        f"{MODULE}.weight": torch.ones((32, 32), dtype=torch.float8_e4m3fn),
        f"{MODULE}.scale": scale,
        "model.norm.weight": torch.ones(32),
    }
    _write_shard(model, tensors)

    routing = model.convert_tensorwise_shard(
        SHARD,
        model.model_path,
        dict.fromkeys(tensors, SHARD),
        {MODULE: 8},
        {},
    )
    output = load_file(str(model.output_dir / SHARD))

    assert output[f"{MODULE}.scale"].shape == (32, 1)
    assert output[f"{MODULE}.scale"].dtype == torch.float8_e8m0fnu
    assert output["model.norm.weight"].shape == (32,)
    assert routing == dict.fromkeys(output, SHARD)
    assert model.processed_modules == {MODULE}


def test_validate_tensorwise_result_requires_exact_processed_set(tmp_path):
    model = _bare_model(tmp_path)
    plan = SimpleNamespace(selected_modules=frozenset({MODULE}))
    model.processed_modules = {MODULE}
    assert model.validate_tensorwise_result(plan) == {MODULE}

    model.processed_modules.clear()
    with pytest.raises(ValueError, match="differ"):
        model.validate_tensorwise_result(plan)


@pytest.mark.parametrize(
    "fields,message",
    [
        ({"kv_source_layers": [1], "kv_source_layer_ids": [2]}, "Conflicting"),
        ({"kv_source_layers": None}, "list of layer indices"),
        ({"kv_source_layers": [True]}, "list of layer indices"),
        ({"kv_source_layers": [-1]}, "list of layer indices"),
    ],
)
def test_source_layers_rejects_conflicting_or_invalid_values(fields, message):
    with pytest.raises(ValueError, match=message):
        DeepseekV41._source_layers(fields, "kv_source_layers", "kv_source_layer_ids")


def test_make_legacy_quant_config_uses_source_layers_and_memoryless_observer(
    tmp_path,
):
    model = _bare_model(tmp_path, platform="ascend950")

    config = model._make_legacy_quant_config()

    assert config["format"] == "float-quantized"
    assert config["weight_block_size"] == [1, 32]
    assert config["ignore"] == [
        "layers.2.attn.indexer.weights_proj",
        "layers.2.attn.indexer.compressor.wgate",
        "layers.2.attn.indexer.compressor.wkv",
        "layers.1.attn.compressor.wgate",
        "layers.1.attn.compressor.wkv",
        "head",
    ]
    assert {
        (group["targets"][0], group["weights"]["num_bits"])
        for group in config["config_groups"].values()
    } == {("Linear", 8), ("MoEGMM", 4)}
    assert all(
        group["input_activations"]["observer"] == "memoryless"
        for group in config["config_groups"].values()
    )


@pytest.mark.parametrize("platform", ["A3", "ascend950"])
def test_refresh_config_writes_platform_contract_without_mutating_source(
    tmp_path, platform
):
    model = _bare_model(tmp_path, platform=platform)
    model.config["quantization_config"] = {"quant_method": "fp8"}
    original = json.loads(json.dumps(model.config))

    model.refresh_config([], tensor_plan=None)

    written = json.loads((model.output_dir / "config.json").read_text())
    assert model.config == original
    if platform == "A3":
        assert written["quantization_config"] == {
            "quant_method": "ascend",
            "model_quant_type": "W8A8_DYNAMIC",
        }
    else:
        assert written["quantization_config"]["format"] == "float-quantized"


def test_make_ascend_description_dispatches_by_platform(tmp_path):
    model = _bare_model(tmp_path)
    names = {
        f"{MODULE}.weight",
        f"{MODULE}.weight_scale",
        f"{MODULE}.weight_offset",
    }

    description = model.make_ascend_description(names, frozenset({MODULE}))

    assert description[f"{MODULE}.weight"] == "W8A8_DYNAMIC"

    model.args.deploy_platform = "ascend950"
    model.args.quant_dtype = "mxfp"
    with pytest.raises(NotImplementedError, match="not implemented"):
        model.make_ascend_description(names, frozenset({MODULE}))
