"""Logic tests for LlmPtqWorkflow.

`_run_blockwise` and `_prepare_unit_batch` are end-to-end paths that require a
real model + NPU; we cover only the pure decision logic here.
"""

import os
from types import SimpleNamespace

import pytest
import torch

from amct_pytorch.common.models.llm.common.ptq_units import make_ptq_unit
from amct_pytorch.workflows.llm_ptq import LlmPtqWorkflow


def _make_workflow(
    quant_target=("mlp",),
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
    return workflow


# ---- __init__ validation -------------------------------------------------


def test_init_rejects_multiple_quant_targets():
    args = SimpleNamespace(
        quant_target=["mlp", "attn-linear"],
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
        ("mlp", "moe_mlp_param_dir"),
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
    wf = _make_workflow(quant_target=["mlp"], moe_mlp_param_dir="/custom/dir")
    assert wf._resolve_quant_param_dir() == "/custom/dir"


def test_resolve_quant_param_dir_auto_creates_path_when_missing(tmp_path):
    wf = _make_workflow(
        quant_target=["mlp"], output_dir=str(tmp_path), model_name="qwen3"
    )
    out = wf._resolve_quant_param_dir()
    expected = os.path.join(str(tmp_path), "ptq_params", "qwen3", "mlp")
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
    unit = make_ptq_unit("mlp", "mlp.up", layer_idx=5, module=None)
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


# ---- _prepare_experiment_dirs --------------------------------------------


def test_prepare_experiment_dirs_creates_log_dir_and_quant_param_dir(tmp_path):
    wf = _make_workflow(output_dir=str(tmp_path), quant_target=["mlp"])
    wf._prepare_experiment_dirs()
    assert os.path.isdir(wf.args.log_dir)
    assert os.path.isdir(wf.args.quant_param_dir)
    assert wf.args.log_dir.endswith("logs")
