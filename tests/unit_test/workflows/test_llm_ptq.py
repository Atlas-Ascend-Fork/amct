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
"""Logic tests for LlmPtqWorkflow.

`_run_blockwise` and `_prepare_unit_batch` are end-to-end paths that require a
real model + NPU; we cover only the pure decision logic here.
"""

import importlib
import os
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

from amct_pytorch.common.models.llm.common.ptq_params import PtqParamHandler
from amct_pytorch.common.models.llm.common.ptq_units import make_ptq_unit
from amct_pytorch.workflows.llm_ptq import LlmPtqWorkflow

QUANT_TARGET_MLP = "mlp"


def _make_workflow(
    quant_target=(QUANT_TARGET_MLP,),
    granularity="block",
    output_dir=None,
    model_name="qwen3",
    **extra,
):
    workflow = LlmPtqWorkflow.__new__(LlmPtqWorkflow)
    base_kwargs = {
        "quant_target": list(quant_target),
        "granularity": granularity,
        "output_dir": output_dir or "/tmp/_ptq_test_out",
        "model_name": model_name,
        "device": torch.device("cpu"),
        "attn_linear_param_dir": "",
        "attn_cache_param_dir": "",
        "moe_mlp_param_dir": "",
        "start_block_idx": 0,
        "end_block_idx": 2,
    }
    base_kwargs.update(extra)  # explicit overrides win
    args = SimpleNamespace(**base_kwargs)
    workflow.args = args
    workflow.granularity = granularity
    workflow.model_name = model_name
    workflow.device = args.device
    if quant_target:
        workflow.quant_target = quant_target[0]
    workflow.pipeline = None
    workflow.data_provider = None
    workflow.solver_key = "blockwise"
    workflow.quant_result_dir = extra.get("quant_result_dir")
    workflow.quant_result_writer = None
    return workflow


# ---- __init__ validation -------------------------------------------------


def test_init_rejects_multiple_quant_targets():
    args = SimpleNamespace(
        quant_target=[QUANT_TARGET_MLP, "attn-linear"],
        granularity="block",
        output_dir="/tmp",
        model_name="qwen3",
        device="cpu",
    )
    with pytest.raises(ValueError, match="ptq only supports a single quant_target"):
        LlmPtqWorkflow(args)


# ---- _get_quant_param_dir_attr ------------------------------------------


@pytest.mark.parametrize(
    "target,expected",
    [
        ("attn-linear", "attn_linear_param_dir"),
        ("attn-cache", "attn_cache_param_dir"),
        (QUANT_TARGET_MLP, "moe_mlp_param_dir"),
        ("moe", "moe_mlp_param_dir"),
    ],
)
def test_get_quant_param_dir_attr_maps_target_to_args_field(target, expected):
    wf = _make_workflow(quant_target=[target])
    assert wf._get_quant_param_dir_attr() == expected


def test_get_quant_param_dir_attr_raises_for_unknown_target():
    wf = _make_workflow(quant_target=["not-a-target"])
    with pytest.raises(ValueError, match="Unsupported quant_target"):
        wf._get_quant_param_dir_attr()


# ---- _resolve_quant_param_dir -------------------------------------------


def test_resolve_quant_param_dir_uses_explicit_arg_when_provided():
    wf = _make_workflow(
        quant_target=[QUANT_TARGET_MLP], moe_mlp_param_dir="/custom/dir"
    )
    assert wf._resolve_quant_param_dir() == "/custom/dir"


def test_resolve_quant_param_dir_auto_creates_path_when_missing(tmp_path):
    wf = _make_workflow(
        quant_target=[QUANT_TARGET_MLP], output_dir=str(tmp_path), model_name="qwen3"
    )
    out = wf._resolve_quant_param_dir()
    expected = os.path.join(str(tmp_path), "ptq_params", "qwen3", QUANT_TARGET_MLP)
    assert out == expected
    # Side effect: writes back to args
    assert wf.args.moe_mlp_param_dir == expected


def test_resolve_quant_param_dir_sanitizes_slashes_in_model_name(tmp_path):
    wf = _make_workflow(
        quant_target=["attn-linear"],
        output_dir=str(tmp_path),
        model_name="org/SomeModel-7B",
    )
    out = wf._resolve_quant_param_dir()
    assert "/org_SomeModel-7B/" in out


# ---- _move_to_device -----------------------------------------------------


def test_move_to_device_floating_point_tensor_promotes_to_float32():
    wf = _make_workflow()
    out = wf._move_to_device(torch.zeros(2, dtype=torch.bfloat16))
    assert out.dtype == torch.float32
    assert out.device == wf.device


def test_move_to_device_integer_tensor_keeps_dtype():
    wf = _make_workflow()
    out = wf._move_to_device(torch.tensor([1, 2, 3], dtype=torch.int64))
    assert out.dtype == torch.int64


def test_move_to_device_traverses_nested_containers():
    wf = _make_workflow()
    nested = {
        "a": torch.tensor([1.0]),
        "b": [torch.tensor([2.0]), torch.tensor([3])],
        "c": (torch.tensor([4.0]),),
    }
    out = wf._move_to_device(nested)
    assert out["a"].device.type == "cpu"
    assert isinstance(out["b"], list) and out["b"][0].dtype == torch.float32
    assert out["b"][1].dtype == torch.int64
    assert isinstance(out["c"], tuple)


def test_move_to_device_returns_non_tensor_unchanged():
    wf = _make_workflow()
    assert wf._move_to_device("hello") == "hello"
    assert wf._move_to_device(42) == 42


# ---- _unpack_tensor_batch ------------------------------------------------


def test_unpack_tensor_batch_single_element_list():
    wf = _make_workflow()
    t = torch.zeros(2, 3)
    assert torch.equal(wf._unpack_tensor_batch([t]), t)
    assert torch.equal(wf._unpack_tensor_batch((t,)), t)


def test_unpack_tensor_batch_passthrough_for_plain_tensor():
    wf = _make_workflow()
    t = torch.zeros(2, 3)
    assert torch.equal(wf._unpack_tensor_batch(t), t)


def test_unpack_tensor_batch_rejects_two_element_batch():
    wf = _make_workflow()
    with pytest.raises(ValueError, match="exactly one tensor"):
        wf._unpack_tensor_batch([torch.zeros(1), torch.zeros(1)])


# ---- _save_unit_result ---------------------------------------------------


def test_save_unit_result_layer_indexed_filename(tmp_path):
    wf = _make_workflow(output_dir=str(tmp_path))
    wf.args.quant_param_dir = str(tmp_path)
    unit = make_ptq_unit(QUANT_TARGET_MLP, "mlp.up", layer_idx=5, module=None)
    wf._save_unit_result(unit, {"k": torch.tensor([1.0])})
    saved = torch.load(tmp_path / "layer_5_mlp_up.pt")
    assert torch.equal(saved["k"], torch.tensor([1.0]))


def test_save_unit_result_unindexed_filename_when_layer_idx_none(tmp_path):
    wf = _make_workflow(output_dir=str(tmp_path))
    wf.args.quant_param_dir = str(tmp_path)
    unit = make_ptq_unit("global", "global", layer_idx=None, module=None)
    wf._save_unit_result(unit, {"k": 1})
    assert (tmp_path / "global.pt").exists()


# ---- _build_block_solver -------------------------------------------------


def test_build_block_solver_passes_only_signature_matching_kwargs():
    wf = _make_workflow()

    captured = {}

    class _Solver:
        def __init__(self, args, layer_idx, model):
            captured.update(args=args, layer_idx=layer_idx, model=model)

    block = object()
    wf._build_block_solver(_Solver, layer_idx=4, block=block)
    assert captured["args"] is wf.args
    assert captured["layer_idx"] == 4
    assert captured["model"] is block


def test_build_block_solver_supports_block_kwarg_alias():
    wf = _make_workflow()

    captured = {}

    class _Solver:
        def __init__(self, block):
            captured["block"] = block

    block = object()
    wf._build_block_solver(_Solver, layer_idx=0, block=block)
    assert captured["block"] is block


# ---- __init__ ------------------------------------------------------------


def test_init_sets_solver_key_default_and_custom():
    args = SimpleNamespace(
        quant_target=[QUANT_TARGET_MLP],
        granularity="block",
        output_dir="/tmp/ptq",
        model_name="qwen3",
        device="cpu",
        solver="modelwise",
    )
    wf = LlmPtqWorkflow(args)
    assert wf.solver_key == "modelwise"
    assert wf.quant_target == QUANT_TARGET_MLP
    assert wf.model_name == "qwen3"
    assert wf.pipeline is None
    assert wf.data_provider is None


def test_init_solver_key_defaults_to_blockwise():
    args = SimpleNamespace(
        quant_target=["attn-linear"],
        granularity="block",
        output_dir="/tmp",
        model_name="qwen3",
        device="cpu",
    )
    wf = LlmPtqWorkflow(args)
    assert wf.solver_key == "blockwise"


# ---- run with model granularity (calls _run_modelwise) -------------------


def test_ptq_run_modelwise(monkeypatch):
    wf = _make_workflow(granularity="model")

    def setup():
        return "sink"

    wf.setup = setup
    monkeypatch.setattr(
        "amct_pytorch.workflows.llm_ptq.SOLVER_REGISTRY",
        SimpleNamespace(get=lambda k: object()),
    )
    monkeypatch.setattr(
        "amct_pytorch.workflows.llm_ptq.logger",
        SimpleNamespace(remove=lambda h: None),
    )
    with pytest.raises(ValueError, match="unsupported granularity .* for ptq"):
        wf.run()


# ---- _build_pipeline error path -------------------------------------------


def test_build_pipeline_raises_when_model_not_registered(monkeypatch):
    monkeypatch.setattr(
        "amct_pytorch.workflows.llm_ptq.MODEL_REGISTRY",
        SimpleNamespace(get=lambda k: exec("raise KeyError('unreg')")),
    )
    wf = _make_workflow(model_name="nonexistent")
    with pytest.raises(KeyError):
        wf._build_pipeline()


# ---- _prepare_unit_batch non-tuple input path -----------------------------


def test_prepare_unit_batch_non_tuple_inputs(monkeypatch):
    wf = _make_workflow()
    unit = make_ptq_unit(
        QUANT_TARGET_MLP, "test_unit", layer_idx=0, module=nn.Linear(4, 4)
    )
    wf.data_provider = MagicMock()
    wf.data_provider.load_unit_inputs = MagicMock(return_value=torch.randn(2, 4))
    wf.data_provider.materialize_gt = MagicMock(return_value=torch.randn(2, 4))
    wf.data_provider.build_unit_batch = MagicMock(return_value=object())

    observe_states = []
    monkeypatch.setattr(
        "amct_pytorch.workflows.llm_ptq.set_model_to_observe",
        lambda m, v: observe_states.append(v),
    )

    result = wf._prepare_unit_batch(unit)
    assert result is not None
    assert observe_states == [True, False]


def test_prepare_unit_batch_restores_quant_state_when_materialization_fails(
    monkeypatch,
):
    wf = _make_workflow()
    unit = make_ptq_unit(
        QUANT_TARGET_MLP, "test_unit", layer_idx=0, module=nn.Linear(4, 4)
    )
    wf.data_provider = MagicMock()
    wf.data_provider.load_unit_inputs = MagicMock(return_value=torch.randn(2, 4))
    wf.data_provider.materialize_gt = MagicMock(
        side_effect=RuntimeError("materialization failed")
    )
    observe_states = []
    monkeypatch.setattr(
        "amct_pytorch.workflows.llm_ptq.set_model_to_observe",
        lambda m, v: observe_states.append(v),
    )

    with pytest.raises(RuntimeError, match="materialization failed"):
        wf._prepare_unit_batch(unit)

    assert observe_states == [True, False]


# ---- _run_blockwise: empty units warning ----------------------------------


def test_run_blockwise_empty_units_warning(monkeypatch):
    wf = _make_workflow()
    wf.pipeline = MagicMock()
    wf.pipeline.num_layers = 10
    wf.pipeline.build_quant_block = MagicMock(return_value=nn.Linear(4, 4))
    wf.pipeline.iter_ptq_units = MagicMock(return_value=iter([]))
    wf.data_provider = MagicMock()

    warns = []
    monkeypatch.setattr(
        "amct_pytorch.workflows.llm_ptq.logger",
        MagicMock(warning=lambda msg, *args: warns.append(msg)),
    )
    monkeypatch.setattr(
        "amct_pytorch.workflows.llm_ptq.set_model_to_observe", lambda m, v: None
    )

    wf.args.start_block_idx = 0
    wf.args.end_block_idx = 1
    wf.device = "cpu"
    wf.quant_target = QUANT_TARGET_MLP

    results = wf._run_blockwise(object)  # pylint: disable=assignment-from-no-return
    assert results == {}
    assert len(warns) >= 1


# ---- _run_blockwise: skip existing params ---------------------------------


def test_run_blockwise_skip_existing_params(monkeypatch, tmp_path):
    param_dir = tmp_path / "params"
    param_dir.mkdir()
    existing_path = param_dir / "layer_0_mlp.pt"
    torch.save({"dummy": torch.tensor(1.0)}, str(existing_path))

    wf = _make_workflow(output_dir=str(tmp_path))
    wf.args.quant_param_dir = str(param_dir)
    wf.pipeline = MagicMock()
    wf.pipeline.num_layers = 10
    wf.pipeline.build_quant_block = MagicMock(return_value=nn.Linear(4, 4))
    unit = make_ptq_unit(
        QUANT_TARGET_MLP, QUANT_TARGET_MLP, layer_idx=0, module=nn.Linear(4, 4)
    )
    wf.pipeline.iter_ptq_units = MagicMock(return_value=iter([unit]))

    wf.data_provider = MagicMock()
    wf.data_provider.load_unit_inputs = MagicMock(return_value=(torch.randn(2, 4), {}))
    wf.data_provider.materialize_gt = MagicMock(return_value=torch.randn(2, 4))
    wf.data_provider.build_unit_batch = MagicMock(
        return_value=SimpleNamespace(data_loader=[(torch.randn(2, 4), {})], kwargs={})
    )

    monkeypatch.setattr(
        "amct_pytorch.workflows.llm_ptq.set_model_to_observe", lambda m, v: None
    )
    monkeypatch.setattr(
        "amct_pytorch.workflows.llm_ptq.logger",
        MagicMock(),
    )

    wf.args.start_block_idx = 0
    wf.args.end_block_idx = 1
    wf.device = "cpu"
    wf.quant_target = QUANT_TARGET_MLP

    result = wf._run_blockwise(object)  # pylint: disable=assignment-from-no-return
    assert result == {0: {}}


def test_artifact_mode_does_not_skip_existing_legacy_params(monkeypatch, tmp_path):
    param_dir = tmp_path / "params"
    param_dir.mkdir()
    torch.save({"old": torch.tensor(1.0)}, param_dir / "layer_0_mlp.pt")

    wf = _make_workflow(output_dir=str(tmp_path), algos=["lwc"])
    wf.args.quant_param_dir = str(param_dir)
    module = nn.Linear(4, 4)
    unit = make_ptq_unit(QUANT_TARGET_MLP, QUANT_TARGET_MLP, 0, module)
    wf.pipeline = SimpleNamespace(
        num_layers=1,
        ptq_param_handler=MagicMock(
            export_unit=MagicMock(return_value={"value": torch.tensor([2.0])})
        ),
        build_quant_block=MagicMock(return_value=module),
        iter_ptq_units=MagicMock(return_value=iter([unit])),
    )
    wf.quant_result_writer = MagicMock()
    wf._prepare_unit_batch = MagicMock(
        return_value=SimpleNamespace(data_loader=[], kwargs={})
    )
    solver = MagicMock()
    wf._build_block_solver = MagicMock(return_value=solver)
    wf._save_unit_result = MagicMock()
    monkeypatch.setattr("torch.npu.empty_cache", lambda: None)

    result = wf._run_blockwise(object)

    solver.solve.assert_called_once()
    wf._save_unit_result.assert_called_once()
    wf.quant_result_writer.write_unit.assert_called_once_with(
        unit, result[0][QUANT_TARGET_MLP]
    )


def test_direct_config_export_skips_calibration_solver_and_legacy_params(
    monkeypatch, tmp_path
):
    wf = _make_workflow(
        output_dir=str(tmp_path),
        quant_result_dir=str(tmp_path / "artifact"),
        algos=[],
    )
    module = nn.Linear(4, 4)
    unit = make_ptq_unit(QUANT_TARGET_MLP, QUANT_TARGET_MLP, 0, module)
    handler = PtqParamHandler()
    wf.pipeline = SimpleNamespace(
        num_layers=1,
        ptq_param_handler=handler,
        build_quant_block=MagicMock(return_value=module),
        iter_ptq_units=MagicMock(return_value=iter([unit])),
    )
    wf.args.start_block_idx = 0
    wf.args.end_block_idx = 1
    wf.args.quant_param_dir = str(tmp_path / "params")
    wf.quant_result_writer = MagicMock()
    wf._prepare_unit_batch = MagicMock(side_effect=AssertionError("calibration used"))
    wf._build_block_solver = MagicMock(side_effect=AssertionError("solver used"))
    wf._save_unit_result = MagicMock(
        side_effect=AssertionError("legacy params written")
    )

    result = wf._run_blockwise(object())

    assert result == {0: {QUANT_TARGET_MLP: {}}}
    assert all(not parameter.requires_grad for parameter in module.parameters())
    wf.quant_result_writer.write_unit.assert_called_once_with(unit, {})
    wf._prepare_unit_batch.assert_not_called()
    wf._build_block_solver.assert_not_called()
    wf._save_unit_result.assert_not_called()


def test_direct_config_export_rejects_hidden_calibrated_payload(tmp_path):
    wf = _make_workflow(
        output_dir=str(tmp_path),
        quant_result_dir=str(tmp_path / "artifact"),
        algos=[],
    )
    module = nn.Linear(4, 4)
    unit = make_ptq_unit(QUANT_TARGET_MLP, QUANT_TARGET_MLP, 0, module)
    handler = MagicMock()
    handler.export_unit.return_value = {"unexpected": torch.ones(1)}
    wf.pipeline = SimpleNamespace(
        num_layers=1,
        ptq_param_handler=handler,
        build_quant_block=MagicMock(return_value=module),
        iter_ptq_units=MagicMock(return_value=iter([unit])),
    )
    wf.args.start_block_idx = 0
    wf.args.end_block_idx = 1
    wf.args.quant_param_dir = str(tmp_path / "params")
    wf.quant_result_writer = MagicMock()

    with pytest.raises(ValueError, match="found calibrated parameters"):
        wf._run_blockwise(object())


# ---- _prepare_experiment_dirs --------------------------------------------


def test_prepare_experiment_dirs_creates_log_dir_and_quant_param_dir(tmp_path):
    wf = _make_workflow(output_dir=str(tmp_path), quant_target=[QUANT_TARGET_MLP])
    wf._prepare_experiment_dirs()
    assert os.path.isdir(wf.args.log_dir)
    assert os.path.isdir(wf.args.quant_param_dir)
    assert wf.args.log_dir.endswith("logs")


def test_prepare_experiment_dirs_direct_config_does_not_create_param_dir(tmp_path):
    wf = _make_workflow(
        output_dir=str(tmp_path),
        quant_result_dir=str(tmp_path / "artifact"),
        algos=[],
    )
    wf._resolve_quant_param_dir = MagicMock(
        side_effect=AssertionError("legacy param directory resolved")
    )

    wf._prepare_experiment_dirs()

    assert os.path.isdir(wf.args.log_dir)
    assert wf.args.quant_param_dir == ""
    wf._resolve_quant_param_dir.assert_not_called()


def test_llm_ptq_run_blockwise(monkeypatch):
    wf = _make_workflow(quant_target=[QUANT_TARGET_MLP], granularity="block")

    def setup():
        return "sink"

    wf.setup = setup

    class FakeBlockwiseSolver:
        pass

    monkeypatch.setattr(
        "amct_pytorch.workflows.llm_ptq.SOLVER_REGISTRY.get",
        lambda k: FakeBlockwiseSolver if k == "blockwise" else None,
    )
    called = {}

    def _run_blockwise(*a, **k):
        called.update({"run": True})

    wf._run_blockwise = _run_blockwise
    monkeypatch.setattr(
        "amct_pytorch.workflows.llm_ptq.logger",
        importlib.import_module("types").SimpleNamespace(remove=lambda h: None),
    )
    wf.run()
    assert called.get("run") is True


def test_llm_ptq_run_publishes_artifact_after_blockwise_success(monkeypatch):
    wf = _make_workflow(quant_target=[QUANT_TARGET_MLP], granularity="block")
    wf.setup = lambda: "sink"
    wf.quant_result_writer = MagicMock()
    wf._run_blockwise = MagicMock(return_value={0: {"mlp": {}}})
    monkeypatch.setattr(
        "amct_pytorch.workflows.llm_ptq.SOLVER_REGISTRY.get", lambda key: object()
    )
    monkeypatch.setattr(
        "amct_pytorch.workflows.llm_ptq.logger", SimpleNamespace(remove=lambda _: None)
    )

    result = wf.run()

    assert result == {0: {"mlp": {}}}
    wf.quant_result_writer.publish.assert_called_once_with()


def test_llm_ptq_run_never_publishes_after_unit_failure(monkeypatch):
    wf = _make_workflow(quant_target=[QUANT_TARGET_MLP], granularity="block")
    wf.setup = lambda: "sink"
    wf.quant_result_writer = MagicMock()
    wf._run_blockwise = MagicMock(side_effect=RuntimeError("unit failed"))
    monkeypatch.setattr(
        "amct_pytorch.workflows.llm_ptq.SOLVER_REGISTRY.get", lambda key: object()
    )
    monkeypatch.setattr(
        "amct_pytorch.workflows.llm_ptq.logger", SimpleNamespace(remove=lambda _: None)
    )

    with pytest.raises(RuntimeError, match="unit failed"):
        wf.run()

    wf.quant_result_writer.publish.assert_not_called()


def test_llm_ptq_run_unknown_granularity(monkeypatch):
    wf = _make_workflow(granularity="unknown")

    def setup():
        return "sink"

    wf.setup = setup
    monkeypatch.setattr(
        "amct_pytorch.workflows.llm_ptq.SOLVER_REGISTRY.get", lambda k: None
    )
    monkeypatch.setattr(
        "amct_pytorch.workflows.llm_ptq.logger",
        importlib.import_module("types").SimpleNamespace(remove=lambda h: None),
    )
    with pytest.raises(ValueError, match="Unsupported .*granularity"):
        wf.run()


def test_llm_ptq_setup(monkeypatch):
    wf = _make_workflow(quant_target=[QUANT_TARGET_MLP])
    registered = {}
    monkeypatch.setattr(
        wf, "_register_components", lambda: registered.update({"reg": True})
    )
    monkeypatch.setattr(
        wf, "_prepare_experiment_dirs", lambda: registered.update({"dirs": True})
    )
    monkeypatch.setattr(
        wf, "_build_pipeline", lambda: registered.update({"pipeline": True})
    )
    monkeypatch.setattr(
        wf, "_build_data_provider", lambda: registered.update({"data": True})
    )
    monkeypatch.setattr(
        "amct_pytorch.workflows.llm_ptq.setup_run_logging",
        lambda log_dir, name: ("sink_id", None),
    )
    wf.setup()
    assert registered.get("reg") is True
    assert registered.get("pipeline") is True
    assert registered.get("data") is True


def test_get_quant_param_dir_attr_raises_on_unsupported_target():
    wf = _make_workflow(quant_target=["unsupported"], output_dir="/tmp/fake")
    with pytest.raises(ValueError, match="Unsupported quant_target"):
        wf._get_quant_param_dir_attr()


def test_resolve_quant_param_dir_auto_generates_when_not_configured():
    wf = _make_workflow(quant_target=[QUANT_TARGET_MLP], output_dir="/tmp/fake")
    wf.model_name = "test_model"
    setattr(wf.args, "moe_mlp_param_dir", "")
    result = wf._resolve_quant_param_dir()
    assert "ptq_params" in result
    assert "test_model" in result
    assert QUANT_TARGET_MLP in result


def test_unpack_tensor_batch_raises_on_multi_element_tuple():
    wf = _make_workflow(quant_target=[QUANT_TARGET_MLP], output_dir="/tmp/fake")
    with pytest.raises(ValueError, match="contain exactly one tensor"):
        wf._unpack_tensor_batch((torch.tensor(1), torch.tensor(2)))


def test_unpack_tensor_batch_returns_single_tensor():
    wf = _make_workflow(quant_target=[QUANT_TARGET_MLP], output_dir="/tmp/fake")
    t = torch.tensor([1.0])
    assert wf._unpack_tensor_batch((t,)) is t
    assert wf._unpack_tensor_batch(t) is t


def test_build_block_solver_passes_kwargs():
    wf = _make_workflow(quant_target=[QUANT_TARGET_MLP], output_dir="/tmp/fake")
    captured = {}

    class _Solver:
        def __init__(self, args, layer_idx, block):
            captured["layer_idx"] = layer_idx
            captured["block"] = block

    block = nn.Linear(4, 4)
    wf._build_block_solver(_Solver, layer_idx=3, block=block)
    assert captured["layer_idx"] == 3
    assert captured["block"] is block


def test_save_unit_result_constructs_path(tmp_path):
    from amct_pytorch.common.models.llm.common.ptq_units import (
        make_ptq_unit as mk_ptq_unit,
    )

    wf = _make_workflow(quant_target=[QUANT_TARGET_MLP])
    wf.args.quant_param_dir = str(tmp_path)
    unit = mk_ptq_unit(
        QUANT_TARGET_MLP, QUANT_TARGET_MLP, layer_idx=2, module=nn.Linear(4, 4)
    )
    wf._save_unit_result(unit, torch.randn(4, 4))
    assert (tmp_path / "layer_2_mlp.pt").exists()


# ---- _run_blockwise (mocked pipeline) ------------------------------------


def test_ptq_run_blockwise_mocked(monkeypatch):
    wf = _make_workflow(quant_target=[QUANT_TARGET_MLP])
    wf.pipeline = MagicMock()
    wf.pipeline.num_layers = 10
    wf.pipeline.build_quant_block = MagicMock(return_value=nn.Linear(4, 4))

    unit = make_ptq_unit(
        QUANT_TARGET_MLP, QUANT_TARGET_MLP, layer_idx=0, module=nn.Linear(4, 4)
    )
    wf.pipeline.iter_ptq_units = MagicMock(return_value=iter([unit]))

    wf.data_provider = MagicMock()
    wf.data_provider.load_unit_inputs = MagicMock(return_value=(torch.randn(2, 4), {}))
    wf.data_provider.materialize_gt = MagicMock(return_value=torch.randn(2, 4))
    wf.data_provider.build_unit_batch = MagicMock(
        return_value=SimpleNamespace(data_loader=[(torch.randn(2, 4), {})], kwargs={})
    )

    monkeypatch.setattr(
        "amct_pytorch.workflows.llm_ptq.set_model_to_observe", lambda m, v: None
    )
    monkeypatch.setattr(
        "amct_pytorch.workflows.llm_ptq.logger",
        MagicMock(),
    )
    monkeypatch.setattr(torch, "save", lambda obj, f: None)

    wf.args.start_block_idx = 0
    wf.args.end_block_idx = 1
    wf.device = "cpu"
    wf.quant_target = QUANT_TARGET_MLP
    wf.args.quant_param_dir = "/tmp/fake"

    class FakeSolver:
        def __init__(self, **kwargs):
            pass

        def solve(self, data_loader, **forward_kwargs):
            pass

        def finalize(self):
            return {}

    results = wf._run_blockwise(FakeSolver)  # pylint: disable=assignment-from-no-return
    assert 0 in results


def test_register_components_runs_without_error(monkeypatch):
    monkeypatch.setattr(
        "amct_pytorch.workflows.llm_ptq.register_algorithms", lambda: None
    )
    monkeypatch.setattr(
        "amct_pytorch.workflows.llm_ptq.register_llm_models", lambda: None
    )
    monkeypatch.setattr("amct_pytorch.workflows.llm_ptq.register_dtype", lambda: None)
    monkeypatch.setattr("amct_pytorch.workflows.llm_ptq.register_solvers", lambda: None)
    workflow = _make_workflow()
    workflow._register_components()


def test_build_pipeline_uses_registry(monkeypatch):
    def fake_cls(args):
        return SimpleNamespace(args=args)

    monkeypatch.setattr(
        "amct_pytorch.workflows.llm_ptq.MODEL_REGISTRY",
        SimpleNamespace(get=lambda k: fake_cls),
    )
    workflow = _make_workflow()
    pipeline = workflow._build_pipeline()
    assert pipeline is not None


def test_artifact_setup_rejects_partial_layer_range_before_fingerprinting(
    monkeypatch, tmp_path
):
    workflow = _make_workflow(
        quant_target=[QUANT_TARGET_MLP],
        quant_result_dir=str(tmp_path / "artifact"),
        start_block_idx=0,
        end_block_idx=1,
    )
    workflow.args.model = str(tmp_path / "model")
    workflow._register_components = MagicMock()
    workflow._prepare_experiment_dirs = MagicMock()
    workflow._build_pipeline = MagicMock(return_value=SimpleNamespace(num_layers=2))
    workflow._build_data_provider = MagicMock(return_value=MagicMock())
    fingerprint = MagicMock()
    monkeypatch.setattr(
        "amct_pytorch.workflows.llm_ptq.compute_checkpoint_fingerprint", fingerprint
    )

    with pytest.raises(ValueError, match="full model layer range"):
        workflow.setup()

    fingerprint.assert_not_called()
    assert not (tmp_path / "artifact").exists()
