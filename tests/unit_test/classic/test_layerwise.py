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

import importlib
import json
import os
from types import SimpleNamespace
from dataclasses import dataclass

import pytest
import torch
from datasets import Dataset
from pathlib import Path
from safetensors.torch import save_file
from torch import nn

from amct_pytorch.classic.layerwise import (
    LayerwiseQuantizationPlan,
    build_args_from_config,
    load_calibration_data,
    load_model_config,
    _needs_extract,
    prepare_layerwise_quantization,
    resolve_model_name,
)
from amct_pytorch.classic.quant_result import (
    compute_model_fingerprint,
    load_quant_manifest,
    reload_quant_params,
)
from amct_pytorch.common.utils.registry_factory import Registry


def _minmax_config(**updates):
    config = {
        "batch_num": 4,
        "quant_cfg": {
            "weights": {
                "type": "int8",
                "symmetric": True,
                "strategy": "channel",
            }
        },
        "algorithm": {"minmax"},
        "skip_layers": {"lm_head"},
    }
    config.update(updates)
    return config


def _model_dir(tmp_path, model_type="qwen3", **model_config):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    payload = {"model_type": model_type, **model_config}
    (model_dir / "config.json").write_text(json.dumps(payload), encoding="utf-8")
    return model_dir, payload


def _result_dir(tmp_path):
    result_dir = tmp_path / "result"
    result_dir.mkdir()
    return result_dir


def _registry(*entries):
    registry = Registry("test-model")
    for name, model_type, adapter in entries:
        registry.register(
            adapter, name=name, model_type=model_type, task="llm", family="test"
        )
    return registry


class _FakeAdapter:
    calls = []

    def __init__(self, args):
        self.args = args
        type(self).calls.append(args)


@dataclass
class _MockQwenArgs:
    model: str
    model_name: str
    quant_result_dir: str
    quant_target: list
    quant_dtype: str
    w_bits: int
    a_bits: int
    bit_policy: object
    algos: list
    batch_num: int
    cali_bsz: int
    nsamples: int
    seq_len: int
    device: str
    work_dir: str
    data_dir: str
    output_dir: str
    skip_layers: set
    dataset: object
    trust_remote_code: bool
    weight_strategy: str
    weight_symmetric: bool
    activation_strategy: object
    activation_symmetric: object
    activation_dynamic: bool
    group_size: object
    granularity: str = "block"
    solver: str = "calibrate"
    start_block_idx: int = 0
    end_block_idx: int = 2**31 - 1
    attn_linear_param_dir: str = ""
    attn_cache_param_dir: str = ""
    moe_mlp_param_dir: str = ""


class _MockMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(4, 8)
        self.up_proj = nn.Linear(4, 8)
        self.down_proj = nn.Linear(8, 4)

    def forward(self, x):
        return self.down_proj(
            torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x)
        )


class _WrappedMockMLP(_MockMLP):
    """Shape-compatible with Qwen adapters whose build_quant_block wraps Linear."""

    def __init__(self):
        super().__init__()
        gate_linear = self.gate_proj
        up_linear = self.up_proj
        down_linear = self.down_proj
        self.gate_proj = nn.Module()
        self.gate_proj.linear = gate_linear
        self.up_proj = nn.Module()
        self.up_proj.linear = up_linear
        self.down_proj = nn.Module()
        self.down_proj.linear = down_linear


class _MockQwenAdapter:
    ffn_norm_name = "post_attention_layernorm"

    def __init__(self, args):
        self.args = args
        self.config = SimpleNamespace(
            model_type="qwen3", to_dict=lambda: {"model_type": "qwen3"}
        )
        self.tokenizer = SimpleNamespace(encode=lambda text: list(text.encode("utf-8")))
        self.num_layers = 2
        self.extract_calls = []

    def get_layer_weight_prefix(self, layer_idx):
        return "model.layers.{}.".format(layer_idx)

    def build_quant_block(self, layer_idx):
        return SimpleNamespace(mlp=_MockMLP())

    def do_embedding_forward(self, samples, hook_name=None):
        self.extract_calls.append(("embedding", len(samples), hook_name))
        return [torch.randn(1, 3, 4, dtype=torch.float32) for _ in samples]

    def do_block_forward(self, layer_idx, samples, hook_name=None):
        self.extract_calls.append(("block", layer_idx, hook_name))
        payload = torch.randn(1, len(samples), 3, 4)
        torch.save(
            payload,
            os.path.join(self.args.data_dir, "block_{}_mlp_in.pkl".format(layer_idx)),
        )
        return [item for item in payload[0]]

    def load_unit_inputs(self, data_dir, unit):
        from amct_pytorch.common.datasets.ptq_io import load_ptq_inps

        return load_ptq_inps(data_dir, unit.kind, unit.layer_idx)


def _mock_args(tmp_path, static_activation=False):
    result_dir = tmp_path / "result"
    work_dir = tmp_path / "work"
    return _MockQwenArgs(
        model=str(tmp_path / "model"),
        model_name="qwen3",
        quant_result_dir=str(result_dir),
        quant_target=["mlp"],
        quant_dtype="int",
        w_bits=8,
        a_bits=8 if static_activation else 16,
        bit_policy=SimpleNamespace(w_bits=8, a_bits=8 if static_activation else 16),
        algos=["minmax"],
        batch_num=2,
        cali_bsz=1,
        nsamples=2,
        seq_len=3,
        device="cpu",
        work_dir=str(work_dir),
        data_dir=str(work_dir / "ptq_data" / "mlp"),
        output_dir=str(work_dir),
        skip_layers=set(),
        dataset=str(tmp_path / "dataset") if static_activation else None,
        trust_remote_code=False,
        weight_strategy="channel",
        weight_symmetric=True,
        activation_strategy="tensor" if static_activation else None,
        activation_symmetric=True if static_activation else None,
        activation_dynamic=False,
        group_size=None,
    )


def _write_dataset(tmp_path, texts):
    Dataset.from_dict({"text": texts}).save_to_disk(str(tmp_path / "dataset"))


def _write_checkpoint(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps({"model_type": "qwen3"}), encoding="utf-8"
    )
    tensors = {
        "model.embed_tokens.weight": torch.randn(2, 4),
        "model.layers.0.mlp.gate_proj.weight": torch.randn(8, 4),
        "model.layers.1.mlp.up_proj.weight": torch.randn(8, 4),
    }
    save_file(tensors, str(model_dir / "model.safetensors"))
    return tensors


def _fingerprint_config():
    return SimpleNamespace(model_type="qwen3", to_dict=lambda: {"model_type": "qwen3"})


def _write_matching_checkpoint(tmp_path, model):
    """Write only the model state to keep the mock checkpoint small but exact."""
    model_dir = tmp_path / "matching-model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps({"model_type": "qwen3"}), encoding="utf-8"
    )
    state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    save_file(state, str(model_dir / "model.safetensors"))
    return str(model_dir)


class TestPipelineDispatch:
    @pytest.fixture
    def quantize_module(self):
        return importlib.import_module("amct_pytorch.classic.quantize")

    def test_default_pipeline_uses_original_full_path(
        self, monkeypatch, quantize_module
    ):
        model = nn.Sequential(nn.Linear(2, 2))
        calls = []
        monkeypatch.setattr(
            quantize_module,
            "_quantize_full",
            lambda *args: calls.append(args) or "full-result",
        )
        monkeypatch.setattr(
            quantize_module,
            "_quantize_layerwise",
            lambda *args: pytest.fail("layerwise path must not run"),
        )

        result = quantize_module.quantize(model, config={})

        assert result == "full-result"
        assert calls == [(model, {}, None)]

    def test_explicit_full_uses_original_full_path(self, monkeypatch, quantize_module):
        model = nn.Sequential(nn.Linear(2, 2))
        monkeypatch.setattr(
            quantize_module, "_quantize_full", lambda *args: "explicit-full"
        )
        assert (
            quantize_module.quantize(model, config={"pipeline": "full"})
            == "explicit-full"
        )

    def test_config_pipeline_selects_layerwise(self, monkeypatch, quantize_module):
        monkeypatch.setattr(
            quantize_module,
            "_quantize_layerwise",
            lambda *args: ("layerwise", args),
        )
        result = quantize_module.quantize(
            "/local/model",
            config={"pipeline": "layerwise"},
            quant_result_dir="/result",
        )
        assert result == (
            "layerwise",
            ("/local/model", {"pipeline": "layerwise"}, "/result"),
        )

    def test_string_path_infers_layerwise_when_pipeline_is_omitted(
        self, monkeypatch, quantize_module
    ):
        monkeypatch.setattr(quantize_module, "_quantize_layerwise", lambda *args: args)
        assert quantize_module.quantize(
            "/local/model",
            quant_result_dir="/result",
        ) == ("/local/model", None, "/result")

    def test_layerwise_rejects_module_before_dispatch(
        self, monkeypatch, quantize_module
    ):
        monkeypatch.setattr(
            quantize_module,
            "_quantize_layerwise",
            lambda *args: pytest.fail("dispatch must not run"),
        )
        with pytest.raises(TypeError, match="local directory path str"):
            quantize_module.quantize(nn.Linear(2, 2), config={"pipeline": "layerwise"})

    def test_full_rejects_string_path(self, monkeypatch, quantize_module):
        monkeypatch.setattr(
            quantize_module,
            "_quantize_full",
            lambda *args: pytest.fail("full implementation must not run"),
        )
        with pytest.raises(TypeError, match="not implemented"):
            quantize_module.quantize("/local/model", config={"pipeline": "full"})

    @pytest.mark.parametrize("pipeline", ["", "blockwise", "FULL"])
    def test_rejects_unknown_pipeline(self, quantize_module, pipeline):
        with pytest.raises(ValueError, match="full.*layerwise"):
            quantize_module.quantize(nn.Linear(2, 2), config={"pipeline": pipeline})

    def test_rejects_pipeline_keyword(self, quantize_module):
        with pytest.raises(TypeError, match="unexpected keyword argument"):
            quantize_module.quantize(
                nn.Linear(2, 2),
                pipeline="full",
            )

    def test_rejects_non_string_config_pipeline(self, quantize_module):
        with pytest.raises(TypeError, match="config.pipeline"):
            quantize_module.quantize(nn.Linear(2, 2), config={"pipeline": 1})

    def test_public_layerwise_does_not_silently_claim_completion(
        self, monkeypatch, quantize_module
    ):
        # Week 6-8 replaced the temporary Week 5 failure with the real run.
        # The same test now guards that the private preparation boundary is
        # still the only public hand-off.
        plan = SimpleNamespace(
            args=SimpleNamespace(model_name="fake", quant_target=["mlp"]),
            adapter=None,
            model_config={},
        )
        layerwise_module = importlib.import_module("amct_pytorch.classic.layerwise")
        monkeypatch.setattr(
            layerwise_module,
            "prepare_layerwise_quantization",
            lambda *args: plan,
        )
        with pytest.raises(AttributeError, match="quant_result_dir"):
            quantize_module.quantize(
                "/local/model",
                config={"pipeline": "layerwise"},
                quant_result_dir="/result",
            )


class TestModelResolution:
    def test_loads_local_config_without_transformers(self, tmp_path):
        model_dir, payload = _model_dir(tmp_path)
        normalized, loaded = load_model_config(str(model_dir))
        assert normalized == str(model_dir)
        assert loaded == payload

    def test_resolves_unique_registry_metadata(self, tmp_path):
        model_dir, _ = _model_dir(tmp_path, "qwen3")
        registry = _registry(("adapter-qwen3", "qwen3", _FakeAdapter))
        assert resolve_model_name(str(model_dir), registry) == "adapter-qwen3"

    def test_unknown_model_type_lists_available_adapters(self, tmp_path):
        model_dir, _ = _model_dir(tmp_path, "unknown")
        registry = _registry(("adapter-qwen3", "qwen3", _FakeAdapter))
        with pytest.raises(ValueError, match="adapter-qwen3.*qwen3"):
            resolve_model_name(str(model_dir), registry)

    def test_duplicate_model_type_metadata_is_rejected(self, tmp_path):
        model_dir, _ = _model_dir(tmp_path, "qwen3")
        registry = _registry(
            ("first", "qwen3", _FakeAdapter),
            ("second", "qwen3", type("OtherAdapter", (), {})),
        )
        with pytest.raises(ValueError, match="multiple.*first.*second"):
            resolve_model_name(str(model_dir), registry)

    @pytest.mark.parametrize("path_kind", ["missing", "file", "no_config"])
    def test_rejects_invalid_local_model_path(self, tmp_path, path_kind):
        path = tmp_path / path_kind
        if path_kind == "file":
            path.write_text("not a model", encoding="utf-8")
        elif path_kind == "no_config":
            path.mkdir()
        with pytest.raises(ValueError):
            load_model_config(str(path))

    @pytest.mark.parametrize(
        "payload,match",
        [
            ("{", "valid UTF-8 JSON"),
            ("[]", "JSON object"),
            ("{}", "model_type"),
            ('{"model_type": ""}', "model_type"),
            ('{"model_type": "a", "model_type": "b"}', "duplicate JSON key"),
        ],
    )
    def test_rejects_invalid_model_config(self, tmp_path, payload, match):
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        (model_dir / "config.json").write_text(payload, encoding="utf-8")
        with pytest.raises(ValueError, match=match):
            load_model_config(str(model_dir))


class TestBuildArgsFromConfig:
    def _build(self, tmp_path, config=None, model_config=None):
        return build_args_from_config(
            _minmax_config() if config is None else config,
            str(tmp_path / "model"),
            "qwen3",
            {"model_type": "qwen3"} if model_config is None else model_config,
            str(tmp_path / "result"),
        )

    def test_maps_weight_only_minmax_without_cli_defaults(self, tmp_path):
        args = self._build(tmp_path)
        assert args.model_name == "qwen3"
        assert args.quant_target == ["mlp"]
        assert args.quant_dtype == "int"
        assert (args.w_bits, args.a_bits) == (8, 16)
        assert args.bit_policy.linear_bits() == (8, 16)
        assert args.algos == ["minmax"]
        assert (args.batch_num, args.cali_bsz, args.nsamples) == (4, 1, 4)
        assert args.skip_layers == {"lm_head"}
        assert args.data_dir.endswith(".work/ptq_data/mlp")

    def test_none_config_uses_layerwise_sampling_defaults(self, tmp_path):
        args = build_args_from_config(
            None,
            str(tmp_path / "model"),
            "qwen3",
            {"model_type": "qwen3"},
            str(tmp_path / "result"),
        )
        assert (args.batch_num, args.cali_bsz, args.nsamples) == (128, 1, 128)
        assert (args.w_bits, args.a_bits) == (8, 16)

    def test_maps_weight_only_without_calibration_algorithm(self, tmp_path):
        args = self._build(tmp_path, _minmax_config(algorithm=set()))
        assert args.algos == []
        assert (args.w_bits, args.a_bits) == (8, 16)

    def test_maps_static_activation_minmax(self, tmp_path):
        config = _minmax_config()
        config["quant_cfg"]["inputs"] = {
            "type": "int8",
            "symmetric": True,
            "strategy": "tensor",
        }
        args = self._build(tmp_path, config)
        assert (args.w_bits, args.a_bits) == (8, 8)
        assert args.activation_dynamic is False
        assert args.activation_strategy == "tensor"
        assert args.bit_policy.a_dynamic is False

    def test_maps_int4_group_weight_config(self, tmp_path):
        config = _minmax_config()
        config["quant_cfg"]["weights"] = {
            "type": "int4",
            "symmetric": True,
            "strategy": "group",
            "group_size": 64,
        }
        args = self._build(tmp_path, config)
        assert args.w_bits == 4
        assert args.group_size == 64
        assert args.weight_strategy == "group"

    def test_rejects_moe_outside_layerwise_p0(self, tmp_path):
        model_config = {"model_type": "qwen3_moe", "num_experts": 8}
        with pytest.raises(ValueError, match="dense FFN"):
            self._build(
                tmp_path,
                model_config=model_config,
            )

    @pytest.mark.parametrize(
        "shape_fields,expected",
        [
            ({}, (128, 1, 128)),
            ({"batch_size": 2}, (64, 2, 128)),
            ({"batch_num": 3, "batch_size": 2}, (3, 2, 6)),
            (
                {"batch_size": 2, "num_calibration_samples": 10},
                (5, 2, 10),
            ),
            (
                {"batch_num": 5, "num_calibration_samples": 10},
                (5, 2, 10),
            ),
            (
                {
                    "batch_num": 5,
                    "batch_size": 2,
                    "num_calibration_samples": 10,
                },
                (5, 2, 10),
            ),
        ],
    )
    def test_calibration_sample_relationship(self, tmp_path, shape_fields, expected):
        config = _minmax_config()
        config.pop("batch_num")
        config.update(shape_fields)
        args = self._build(tmp_path, config)
        assert (args.batch_num, args.cali_bsz, args.nsamples) == expected

    @pytest.mark.parametrize(
        "shape_fields",
        [
            {"batch_num": True},
            {"batch_size": 0},
            {"num_calibration_samples": -1},
            {
                "batch_num": 3,
                "batch_size": 2,
                "num_calibration_samples": 5,
            },
            {"batch_size": 3, "num_calibration_samples": 10},
        ],
    )
    def test_rejects_invalid_calibration_shape(self, tmp_path, shape_fields):
        config = _minmax_config()
        config.pop("batch_num")
        config.update(shape_fields)
        with pytest.raises(ValueError):
            self._build(tmp_path, config)

    def test_normalizes_internal_paths_and_runtime_fields(self, tmp_path):
        work_dir = tmp_path / "work"
        dataset = tmp_path / "dataset"
        config = _minmax_config(
            work_dir=str(work_dir),
            dataset=str(dataset),
            device="npu:1",
            max_seq_length=2048,
            trust_remote_code=True,
        )
        args = self._build(tmp_path, config)
        assert args.work_dir == str(work_dir)
        assert args.dataset == str(dataset)
        assert args.device == "npu:1"
        assert args.seq_len == 2048
        assert args.trust_remote_code is True

    @pytest.mark.parametrize("work_dir_kind", ["same", "child", "parent"])
    def test_rejects_work_dir_overlapping_result_dir(self, tmp_path, work_dir_kind):
        result_dir = tmp_path / "result"
        work_dirs = {
            "same": result_dir,
            "child": result_dir / "work",
            "parent": tmp_path,
        }
        config = _minmax_config(work_dir=str(work_dirs[work_dir_kind]))
        with pytest.raises(ValueError, match="separate.*non-overlapping"):
            build_args_from_config(
                config,
                str(tmp_path / "model"),
                "qwen3",
                {"model_type": "qwen3"},
                str(result_dir),
            )

    @pytest.mark.parametrize(
        "config,match",
        [
            (
                _minmax_config(algorithm={"gptq"}),
                "only no algorithm or algorithm 'minmax'",
            ),
            (
                _minmax_config(
                    quant_cfg={
                        "weights": {
                            "type": "float8_e4m3fn",
                            "symmetric": True,
                            "strategy": "tensor",
                        }
                    }
                ),
                "weights.type",
            ),
            (
                _minmax_config(
                    quant_cfg={
                        "weights": {
                            "type": "int8",
                            "symmetric": True,
                            "strategy": "channel",
                        },
                        "inputs": {
                            "type": "int4",
                            "symmetric": True,
                            "strategy": "tensor",
                        },
                    }
                ),
                "inputs.type",
            ),
            (
                _minmax_config(
                    quant_cfg={
                        "weights": {
                            "type": "int8",
                            "symmetric": True,
                            "strategy": "channel",
                        },
                        "inputs": {
                            "type": "int8",
                            "symmetric": True,
                            "strategy": "token",
                            "dynamic": True,
                        },
                    }
                ),
                "static MinMax",
            ),
            (
                _minmax_config(
                    quant_cfg={
                        "kvcache": {
                            "type": "hifloat8",
                            "symmetric": True,
                            "strategy": "tensor",
                        }
                    }
                ),
                "kvcache",
            ),
            (
                _minmax_config(
                    quant_cfg={
                        "*.mlp.weights": {
                            "type": "int8",
                            "symmetric": True,
                            "strategy": "channel",
                        }
                    }
                ),
                "fuzzy",
            ),
        ],
    )
    def test_rejects_features_outside_week5_p0(self, tmp_path, config, match):
        with pytest.raises(ValueError, match=match):
            self._build(tmp_path, config)

    def test_static_activation_requires_minmax(self, tmp_path):
        config = _minmax_config(algorithm=set())
        config["quant_cfg"]["inputs"] = {
            "type": "int8",
            "symmetric": True,
            "strategy": "tensor",
        }
        with pytest.raises(ValueError, match="requires algorithm 'minmax'"):
            self._build(tmp_path, config)

    @pytest.mark.parametrize(
        "config,match",
        [
            (_minmax_config(batch_sze=2), "config fields.*batch_sze"),
            (
                _minmax_config(
                    quant_cfg={
                        "weights": {
                            "type": "int8",
                            "symmetric": True,
                            "strategy": "channel",
                            "stratgey": "tensor",
                        }
                    }
                ),
                "quant_cfg.weights.*stratgey",
            ),
            (
                _minmax_config(algorithm={"minmax": {"percentile": 0.99}}),
                "does not accept algorithm options",
            ),
        ],
    )
    def test_rejects_silently_ignored_config_fields(self, tmp_path, config, match):
        with pytest.raises(ValueError, match=match):
            self._build(tmp_path, config)

    def test_rejects_skip_layers_string(self, tmp_path):
        with pytest.raises(TypeError, match="skip_layers"):
            self._build(tmp_path, _minmax_config(skip_layers="lm_head"))


class TestPrepareLayerwiseQuantization:
    def setup_method(self):
        _FakeAdapter.calls.clear()

    def test_constructs_exactly_the_resolved_mock_adapter(self, tmp_path):
        model_dir, _ = _model_dir(tmp_path, "qwen3")
        result_dir = _result_dir(tmp_path)
        registry = _registry(("qwen3-adapter", "qwen3", _FakeAdapter))

        plan = prepare_layerwise_quantization(
            str(model_dir), _minmax_config(), str(result_dir), registry
        )

        assert isinstance(plan, LayerwiseQuantizationPlan)
        assert isinstance(plan.adapter, _FakeAdapter)
        assert plan.args.model_name == "qwen3-adapter"
        assert plan.args.quant_result_dir == str(result_dir)
        assert _FakeAdapter.calls == [plan.args]

    def test_defaults_result_directory_without_creating_it(self, tmp_path, monkeypatch):
        model_dir, _ = _model_dir(tmp_path)
        registry = _registry(("qwen3", "qwen3", _FakeAdapter))
        monkeypatch.chdir(tmp_path)

        plan = prepare_layerwise_quantization(
            str(model_dir), _minmax_config(), None, registry
        )

        assert plan.args.quant_result_dir == str(tmp_path / "quant_result")
        assert not (tmp_path / "quant_result").exists()

    def test_accepts_missing_result_directory_without_creating_it(self, tmp_path):
        model_dir, _ = _model_dir(tmp_path)
        result_dir = tmp_path / "missing-result"
        registry = _registry(("qwen3", "qwen3", _FakeAdapter))

        plan = prepare_layerwise_quantization(
            str(model_dir), _minmax_config(), str(result_dir), registry
        )

        assert plan.args.quant_result_dir == str(result_dir)
        assert not result_dir.exists()

    def test_rejects_result_file(self, tmp_path):
        model_dir, _ = _model_dir(tmp_path)
        result_file = tmp_path / "result-file"
        result_file.write_text("not a directory", encoding="utf-8")
        registry = _registry(("qwen3", "qwen3", _FakeAdapter))
        with pytest.raises(ValueError, match="directory path"):
            prepare_layerwise_quantization(
                str(model_dir), _minmax_config(), str(result_file), registry
            )

    def test_bad_config_fails_before_adapter_construction(self, tmp_path):
        model_dir, _ = _model_dir(tmp_path)
        result_dir = _result_dir(tmp_path)
        registry = _registry(("qwen3", "qwen3", _FakeAdapter))
        with pytest.raises(ValueError, match="only no algorithm or algorithm 'minmax'"):
            prepare_layerwise_quantization(
                str(model_dir),
                _minmax_config(algorithm={"gptq"}),
                str(result_dir),
                registry,
            )
        assert _FakeAdapter.calls == []


def test_model_type_specific_registration_is_lazy(monkeypatch):
    models_module = importlib.import_module("amct_pytorch.common.models.llm")
    calls = []
    monkeypatch.setattr(
        models_module,
        "import_module",
        lambda name, package=None: calls.append((name, package)),
    )
    models_module.register_llm_models(model_type="qwen3")
    assert calls == [(".qwen.qwen3.qwen3", models_module.__name__)]


def test_unknown_model_type_registration_does_not_import(monkeypatch):
    models_module = importlib.import_module("amct_pytorch.common.models.llm")
    monkeypatch.setattr(
        models_module,
        "import_module",
        lambda *args, **kwargs: pytest.fail("unknown type must not import"),
    )
    models_module.register_llm_models(model_type="not-registered")


def test_unknown_real_model_type_lists_known_adapters_without_import(
    tmp_path, monkeypatch
):
    models_module = importlib.import_module("amct_pytorch.common.models.llm")
    monkeypatch.setattr(
        models_module,
        "import_module",
        lambda *args, **kwargs: pytest.fail("unknown type must not import"),
    )
    model_dir, _ = _model_dir(tmp_path, "not-registered")
    with pytest.raises(ValueError, match=r"qwen3 \(qwen3\)"):
        resolve_model_name(str(model_dir))


def test_qwen3_registration_exposes_model_type_metadata():
    models_module = importlib.import_module("amct_pytorch.common.models.llm")
    registry_module = importlib.import_module("amct_pytorch.common.models")
    models_module.register_llm_models(model_type="qwen3")
    item = registry_module.MODEL_REGISTRY.get_item("qwen3")
    assert item.metadata["model_type"] == "qwen3"


class TestWeek6CalibrationData:
    def test_concatenates_and_cuts_exact_token_windows(self, tmp_path):
        _write_dataset(tmp_path, ["abcdef", "ghij", "klm"])
        adapter = SimpleNamespace(
            args=_mock_args(tmp_path, static_activation=True),
            tokenizer=SimpleNamespace(encode=lambda text: list(text.encode())),
        )
        samples = load_calibration_data(adapter)
        assert len(samples) == 2
        assert all(sample.shape == (1, 3) for sample in samples)
        assert samples[0].tolist() == [
            [97, 98, 99],
        ]

    def test_rejects_empty_text_and_insufficient_tokens(self, tmp_path):
        _write_dataset(tmp_path, ["", "abc"])
        adapter = SimpleNamespace(
            args=_mock_args(tmp_path, static_activation=True),
            tokenizer=SimpleNamespace(encode=lambda text: list(text.encode())),
        )
        with pytest.raises(ValueError, match="empty"):
            load_calibration_data(adapter)

        _write_dataset(tmp_path, ["abc"])
        with pytest.raises(ValueError, match="usable token"):
            load_calibration_data(adapter)

    def test_needs_extract_branches(self, tmp_path):
        assert _needs_extract(_mock_args(tmp_path, static_activation=False)) is False
        assert _needs_extract(_mock_args(tmp_path, static_activation=True)) is True


class TestSourceFingerprint:
    def test_streamed_checkpoint_digest_matches_full_model(self, tmp_path):
        tensors = _write_checkpoint(tmp_path)
        adapter = SimpleNamespace(
            args=_mock_args(tmp_path), config=_fingerprint_config()
        )
        full_model = nn.Module()
        full_model.config = _fingerprint_config()
        for name, tensor in tensors.items():
            parent, attr = name.rsplit(".", 1)
            current = full_model
            for part in parent.split("."):
                child = getattr(current, part, None)
                if child is None:
                    child = nn.Module()
                    setattr(current, part, child)
                current = child
            setattr(current, attr, nn.Parameter(tensor))
        full_model.config = adapter.config
        assert source_fingerprint_matches(adapter, full_model)


def source_fingerprint_matches(adapter, full_model):
    from amct_pytorch.classic.layerwise import _source_model_fingerprint

    actual = _source_model_fingerprint(adapter)
    expected = compute_model_fingerprint(full_model)
    print("actual", actual)
    print("expected", expected)
    return actual == expected


class TestLayerwiseEndToEnd:
    def _prepare(self, tmp_path, static_activation):
        _write_checkpoint(tmp_path)
        args = _mock_args(tmp_path, static_activation=static_activation)
        if static_activation:
            _write_dataset(tmp_path, ["abcdefghij" for _ in range(2)])
        adapter = _MockQwenAdapter(args)
        plan = LayerwiseQuantizationPlan(
            args=args, adapter=adapter, model_config={"model_type": "qwen3"}
        )
        return plan

    def _run(self, plan):
        from amct_pytorch.classic.layerwise import run_layerwise_quantization

        return run_layerwise_quantization(plan)

    def test_weight_only_uses_no_dataset_and_publishes_v2(self, tmp_path):
        plan = self._prepare(tmp_path, static_activation=False)
        result_dir = self._run(plan)

        assert plan.adapter.extract_calls == []
        assert plan.args.dataset is None
        manifest = load_quant_manifest(result_dir)
        assert len(manifest["units"]) == 6
        assert [entry["layer_name"] for entry in manifest["units"]] == [
            "model.layers.0.mlp.gate_proj",
            "model.layers.0.mlp.up_proj",
            "model.layers.0.mlp.down_proj",
            "model.layers.1.mlp.gate_proj",
            "model.layers.1.mlp.up_proj",
            "model.layers.1.mlp.down_proj",
        ]
        assert all(
            entry["quant_module_type"] == "MinMaxQuant" for entry in manifest["units"]
        )
        assert all(
            [item["key"] for item in entry["tensors"]] == ["scale_w"]
            for entry in manifest["units"]
        )

    def test_static_minmax_extracts_and_writes_activation_params(self, tmp_path):
        plan = self._prepare(tmp_path, static_activation=True)
        result_dir = self._run(plan)

        assert len(plan.adapter.extract_calls) == 3
        manifest = load_quant_manifest(result_dir)
        assert len(manifest["units"]) == 6
        assert all(
            [item["key"] for item in entry["tensors"]] == ["scale_w"]
            for entry in manifest["units"]
        )
        cache_manifest = json.loads(
            Path(plan.args.data_dir, "cache_manifest.json").read_text(encoding="utf-8")
        )
        assert len(cache_manifest["files"]) == 2
        assert cache_manifest["cache_key"].startswith("sha256:")

    def test_complete_activation_cache_is_reused(self, tmp_path):
        plan = self._prepare(tmp_path, static_activation=True)
        self._run(plan)
        first_calls = len(plan.adapter.extract_calls)

        plan.args.quant_result_dir = str(tmp_path / "result-2")
        result_2 = self._run(plan)
        assert len(plan.adapter.extract_calls) == first_calls
        assert load_quant_manifest(result_2)["format"] == "amct-quant-result-manifest"

    def test_corrupt_activation_cache_is_re_extracted(self, tmp_path):
        plan = self._prepare(tmp_path, static_activation=True)
        self._run(plan)
        first_calls = len(plan.adapter.extract_calls)
        cache_file = Path(plan.args.data_dir, "block_0_mlp_in.pkl")
        cache_file.write_bytes(b"corrupt")

        plan.args.quant_result_dir = str(tmp_path / "result-2")
        self._run(plan)
        assert len(plan.adapter.extract_calls) > first_calls

    def test_failure_leaves_no_manifest(self, tmp_path):
        plan = self._prepare(tmp_path, static_activation=False)
        original = plan.adapter.build_quant_block

        def fail_on_second(layer_idx):
            if layer_idx == 1:
                raise RuntimeError("intentional PTQ failure")
            return original(layer_idx)

        plan.adapter.build_quant_block = fail_on_second
        with pytest.raises(RuntimeError, match="intentional"):
            self._run(plan)
        assert not Path(plan.args.quant_result_dir, "quant_manifest.json").exists()

    def test_artifact_reloads_into_fresh_dense_model(self, tmp_path):
        model = nn.Module()
        model.config = _fingerprint_config()
        layers = nn.ModuleList([nn.Module() for _ in range(2)])
        backbone = nn.Module()
        backbone.layers = layers
        model.model = backbone
        for layer in layers:
            layer.mlp = _MockMLP()
        plan = self._prepare(tmp_path, static_activation=False)
        plan.args.model = _write_matching_checkpoint(tmp_path, model)
        result_dir = self._run(plan)
        reloaded = reload_quant_params(model, result_dir)
        assert reloaded is model
        assert isinstance(layers[0].mlp.gate_proj, nn.Module)
        assert type(layers[0].mlp.gate_proj).__name__ == "MinMaxQuant"

    def test_plan_accepts_quant_linear_wrapped_dense_mlp(self, tmp_path):
        _write_checkpoint(tmp_path)
        args = _mock_args(tmp_path)
        adapter = _MockQwenAdapter(args)

        adapter.build_quant_block = lambda layer_idx: SimpleNamespace(
            mlp=_WrappedMockMLP()
        )
        plan = LayerwiseQuantizationPlan(
            args=args, adapter=adapter, model_config={"model_type": "qwen3"}
        )
        result_dir = self._run(plan)
        manifest = load_quant_manifest(result_dir)
        assert len(manifest["units"]) == 6
        assert all(entry["ori_module_type"] == "Linear" for entry in manifest["units"])
