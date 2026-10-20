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

import sys
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch import nn

from amct_pytorch.classic.optimizer import InsertQuantizeModulePass
from amct_pytorch.classic.quant_result import (
    MANIFEST_FORMAT,
    MANIFEST_NAME,
    SCHEMA_VERSION,
    compute_model_fingerprint,
    is_quant_result_supported,
    load_quant_manifest,
    read_quant_unit,
    validate_result_directory,
)
from amct_pytorch.classic.quantize import (
    algorithm_register,
    convert,
    quantize,
)
from amct_pytorch.classic.quantize_op.minmax_module import MinMaxQuant
from amct_pytorch.classic.quantize_op.linear_awq_module import LinearAWQuant
from amct_pytorch.classic.quantize_op.gptq_module import GPTQuant
from amct_pytorch.classic.quantize_op.smooth_quant_module import SmoothQuant
from amct_pytorch.common.utils.model_util import ModuleHelper
from amct_pytorch.classic.quantize_op.quant_result_mixin import QuantResultSupportMixin
from amct_pytorch.quantize_op import BaseQuantizeModule

QUANTIZE_MODULE = sys.modules["amct_pytorch.classic.quantize"]
QUANT_RESULT_MODULE = sys.modules["amct_pytorch.classic.quant_result"]


def test_base_quantize_module_keeps_public_import_paths():
    from amct_pytorch.quantize_op import BaseQuantizeModule as PackageBaseQuantizeModule

    assert PackageBaseQuantizeModule is BaseQuantizeModule


def test_classic_quantize_imports_without_ptq_package_layer():
    from amct_pytorch.classic.quantize import (
        algorithm_register as classic_algorithm_register,
    )

    assert classic_algorithm_register is algorithm_register


def test_quant_result_lifecycle_is_not_a_public_api():
    import amct_pytorch

    for name in ("finalize_quant_result", "cancel_quant_result"):
        assert name not in amct_pytorch.__all__
        assert name not in vars(amct_pytorch)
        assert not hasattr(amct_pytorch.classic, name)


def test_artifact_support_is_declared_by_the_registered_quant_class():
    assert is_quant_result_supported(MinMaxQuant)
    assert is_quant_result_supported(LinearAWQuant)
    assert is_quant_result_supported(GPTQuant)
    assert not is_quant_result_supported(SmoothQuant)

    class _DeclaredWithoutParams(QuantResultSupportMixin, BaseQuantizeModule):
        supports_quant_result = True

    # A declaration alone cannot make an unloadable class eligible.
    assert not is_quant_result_supported(_DeclaredWithoutParams)

    class _PlainBase(BaseQuantizeModule):
        supports_quant_result = True

    # Artifact methods are not part of the shared base-class contract.
    assert not hasattr(_PlainBase, 'export_quant_params')
    assert not is_quant_result_supported(_PlainBase)


class _DummyQuantOp(BaseQuantizeModule):
    pass


class _DummyDeployOp(nn.Module):
    pass


def test_algorithm_register_delegates_to_registry():
    with patch("amct_pytorch.algorithms.AlgorithmRegistry.register") as mock_reg:
        algorithm_register("alg_x", "Linear", _DummyQuantOp, _DummyDeployOp)
    mock_reg.assert_called_once_with(
        "alg_x", "Linear", _DummyQuantOp, deploy_op=_DummyDeployOp
    )


def test_algorithm_register_allows_none_deploy_op():
    with patch("amct_pytorch.algorithms.AlgorithmRegistry.register") as mock_reg:
        algorithm_register("alg_y", "Linear", _DummyQuantOp)
    mock_reg.assert_called_once_with("alg_y", "Linear", _DummyQuantOp, deploy_op=None)


def test_algorithm_register_rejects_non_module_quant_op():
    class _NotAQuantOp:
        pass

    with pytest.raises(TypeError):
        algorithm_register("alg_z", "Linear", _NotAQuantOp, _DummyDeployOp)


def test_convert_runs_replace_pass_on_model():
    model = nn.Linear(4, 4)
    with (
        patch.object(QUANTIZE_MODULE, "ModelOptimizer") as mock_opt_cls,
        patch.object(QUANTIZE_MODULE, "ApplyScalingFusionPass"),
        patch.object(QUANTIZE_MODULE, "ReplaceNpuQuantModulePass") as mock_pass_cls,
    ):
        opt = mock_opt_cls.return_value
        convert(model)

    mock_pass_cls.assert_called_once_with()
    assert mock_pass_cls.return_value in [
        call.args[0] for call in opt.add_pass.call_args_list
    ]
    opt.do_optimizer.assert_called_once_with(model)


def test_convert_rejects_non_module():
    with pytest.raises(TypeError):
        convert("not_a_module")


def test_quantize_uses_default_config_when_none():
    model = nn.Linear(4, 4)
    sentinel_default = {"_default": True}
    sentinel_layer = {"layer": "cfg"}
    with (
        patch.object(
            QUANTIZE_MODULE, "set_default_config", return_value=sentinel_default
        ) as mock_default,
        patch.object(
            QUANTIZE_MODULE, "parse_config", return_value=sentinel_layer
        ) as mock_parse,
        patch.object(QUANTIZE_MODULE, "ModelOptimizer") as mock_opt_cls,
        patch.object(QUANTIZE_MODULE, "InsertQuantizeModulePass") as mock_pass_cls,
        patch.object(
            QUANTIZE_MODULE, "InsertLayerNormQuantPass"
        ) as mock_layer_norm_pass_cls,
    ):
        opt = mock_opt_cls.return_value
        quantize(model, None)

    mock_default.assert_called_once_with()
    mock_parse.assert_called_once()
    args, _ = mock_parse.call_args
    assert args[0] is model
    assert args[1] is sentinel_default
    mock_pass_cls.assert_called_once_with(sentinel_layer, quant_result_writer=None)
    mock_layer_norm_pass_cls.assert_called_once_with()
    assert [call.args[0] for call in opt.add_pass.call_args_list] == [
        mock_pass_cls.return_value,
        mock_layer_norm_pass_cls.return_value,
    ]
    opt.do_optimizer.assert_called_once_with(model)


def test_quantize_passes_user_config_through():
    model = nn.Linear(4, 4)
    user_cfg = {"granularity": "tensor"}
    with (
        patch.object(QUANTIZE_MODULE, "set_default_config") as mock_default,
        patch.object(
            QUANTIZE_MODULE, "parse_config", return_value={"layer": "cfg"}
        ) as mock_parse,
        patch.object(QUANTIZE_MODULE, "ModelOptimizer"),
        patch.object(QUANTIZE_MODULE, "InsertQuantizeModulePass"),
    ):
        quantize(model, user_cfg)

    mock_default.assert_not_called()
    args, _ = mock_parse.call_args
    assert args[1] is user_cfg


def test_quantize_rejects_non_module_model():
    with pytest.raises(TypeError):
        quantize("not_a_module")


def test_quantize_rejects_non_dict_config():
    model = nn.Linear(4, 4)
    with pytest.raises(TypeError):
        quantize(model, "bad_config")


# --------------------------------------------------------------------------
# quant_result_dir: quant result artifact persistence.
#
# The default (None) path keeps the old user-visible behaviour: no directory and
# no artifact writes. With a directory, the manifest appears only once every
# expected unit has been dumped.
# --------------------------------------------------------------------------

WEIGHT_ONLY_CFG = {
    'batch_num': 1,
    'quant_cfg': {
        'weights': {'type': 'int8', 'symmetric': True, 'strategy': 'channel'}
    },
    'algorithm': {'minmax'},
}

ACTIVATION_CFG = {
    'batch_num': 2,
    'quant_cfg': {
        'weights': {'type': 'int8', 'symmetric': True, 'strategy': 'channel'},
        'inputs': {'type': 'int8', 'symmetric': True, 'strategy': 'tensor'},
    },
    'algorithm': {'minmax'},
}


class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 16)

    def forward(self, inputs):
        return self.fc2(self.fc1(inputs))


def _tiny_model():
    # int8 minmax only accepts bf16/fp16 source weights.
    return _TinyModel().to(torch.bfloat16)


def _tiny_inputs(batch=8):
    return torch.randn(batch, 64).to(torch.bfloat16)


def test_quantize_without_result_dir_creates_nothing(tmp_path):
    model = _tiny_model()
    before = sorted(p.name for p in tmp_path.iterdir())
    quantize(model, WEIGHT_ONLY_CFG)
    assert sorted(p.name for p in tmp_path.iterdir()) == before
    assert not hasattr(model.fc1, 'quant_result_writer')
    assert not hasattr(model.fc1, 'quant_params_dumped')


def test_quantize_without_result_dir_keeps_pass_signature():
    model = nn.Linear(4, 4)
    with (
        patch.object(QUANTIZE_MODULE, "parse_config", return_value={"layer": "cfg"}),
        patch.object(QUANTIZE_MODULE, "ModelOptimizer"),
        patch.object(QUANTIZE_MODULE, "InsertQuantizeModulePass") as mock_pass_cls,
        patch.object(QUANTIZE_MODULE, "QuantResultWriter") as mock_writer_cls,
    ):
        quantize(model, {"granularity": "tensor"})

    mock_pass_cls.assert_called_once_with({"layer": "cfg"}, quant_result_writer=None)
    mock_writer_cls.create.assert_not_called()


@pytest.mark.parametrize("bad_dir", [1, 1.5, b"x", object()])
def test_quantize_rejects_non_str_result_dir(bad_dir):
    with pytest.raises(TypeError):
        quantize(_tiny_model(), WEIGHT_ONLY_CFG, quant_result_dir=bad_dir)


def test_quantize_rejects_empty_result_dir():
    with pytest.raises(ValueError, match="non-empty path"):
        quantize(_tiny_model(), WEIGHT_ONLY_CFG, quant_result_dir="")


def test_quantize_rejects_missing_result_dir(tmp_path):
    missing = tmp_path / "missing"
    with pytest.raises(ValueError, match="must be an existing directory"):
        quantize(_tiny_model(), WEIGHT_ONLY_CFG, quant_result_dir=str(missing))


def test_quantize_preserves_unmanaged_files_in_result_dir(tmp_path):
    existing = tmp_path / "with_other_files"
    existing.mkdir()
    old_file = existing / "old.safetensors"
    old_file.write_bytes(b"old")
    quantize(_tiny_model(), WEIGHT_ONLY_CFG, quant_result_dir=str(existing))
    assert (existing / MANIFEST_NAME).is_file()
    assert old_file.read_bytes() == b"old"


def test_quantize_rejects_result_dir_that_is_a_file(tmp_path):
    path = tmp_path / "a_file"
    path.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="not a file"):
        quantize(_tiny_model(), WEIGHT_ONLY_CFG, quant_result_dir=str(path))


def test_quantize_rejects_result_dir_when_no_layer_is_quantizable(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    # float32 weights are not accepted by int8 minmax, so nothing matches.
    with pytest.raises(ValueError, match="no quantizable layer"):
        quantize(_TinyModel(), WEIGHT_ONLY_CFG, quant_result_dir=str(out))
    assert list(out.iterdir()) == []


def test_quantize_rejects_result_dir_when_the_root_is_the_quant_layer(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    # parse_config names the root layer '', and module replacement is keyed by
    # name, so the root is never actually replaced: its unit could never be
    # dumped and the manifest would stay unpublished forever.
    root_linear = nn.Linear(64, 32).to(torch.bfloat16)
    with pytest.raises(ValueError, match="itself a quantizable layer"):
        quantize(root_linear, WEIGHT_ONLY_CFG, quant_result_dir=str(out))
    assert list(out.iterdir()) == []


def test_quantize_rejects_root_quant_layer_without_result_dir():
    root_linear = nn.Linear(64, 32).to(torch.bfloat16)
    with pytest.raises(ValueError, match="itself a quantizable layer"):
        quantize(root_linear, WEIGHT_ONLY_CFG)


def test_weight_only_publishes_without_any_forward(tmp_path):
    out = tmp_path / "wo"
    out.mkdir()
    model = _tiny_model()
    quantize(model, WEIGHT_ONLY_CFG, quant_result_dir=str(out))

    assert sorted(p.name for p in out.iterdir()) == sorted(
        [MANIFEST_NAME, "fc1.safetensors", "fc2.safetensors"]
    )
    manifest = load_quant_manifest(str(out))
    validate_result_directory(str(out), manifest)
    assert manifest["format"] == MANIFEST_FORMAT
    assert manifest["schema_version"] == SCHEMA_VERSION
    assert [unit["layer_name"] for unit in manifest["units"]] == ["fc1", "fc2"]
    assert all(u["quant_module_type"] == "MinMaxQuant" for u in manifest["units"])
    assert all(u["ori_module_type"] == "Linear" for u in manifest["units"])
    # The algorithm is recorded inside the full layer-level quant_config.
    assert all(
        u["quant_config"]["algorithm"] == {"minmax": {}} for u in manifest["units"]
    )
    assert all("batch_num" in u["quant_config"] for u in manifest["units"])

    tensors, _, _ = read_quant_unit(str(out), "fc1")
    assert torch.equal(tensors["scale_w"], model.fc1.scale_w.cpu())


def test_activation_publishes_only_after_the_last_calibration_batch(tmp_path):
    out = tmp_path / "act"
    out.mkdir()
    model = _tiny_model()
    quantize(model, ACTIVATION_CFG, quant_result_dir=str(out))
    manifest_path = out / MANIFEST_NAME

    assert list(out.iterdir()) == [], "nothing is final until calibration runs"

    inputs = _tiny_inputs()
    model(inputs)
    assert not manifest_path.exists(), "batch_num-1 forwards must not publish"

    model(inputs)
    assert manifest_path.exists()
    manifest = load_quant_manifest(str(out))
    validate_result_directory(str(out), manifest)
    assert len(manifest["units"]) == 2

    tensors, _, _ = read_quant_unit(str(out), "fc2")
    assert sorted(tensors) == ["scale_d", "scale_w"]
    assert torch.equal(tensors["scale_d"], model.fc2.scale_d.cpu())


def test_publish_hook_is_removed_after_publishing(tmp_path):
    out = tmp_path / "hook"
    out.mkdir()
    model = _tiny_model()
    quantize(model, ACTIVATION_CFG, quant_result_dir=str(out))
    assert model._forward_hooks, "a publish hook must be armed while incomplete"

    inputs = _tiny_inputs()
    model(inputs)
    model(inputs)
    assert not model._forward_hooks, "hook must detach once published"
    # Further inference must neither fail nor touch the published artifact.
    digest_before = (out / MANIFEST_NAME).read_bytes()
    model(inputs)
    assert (out / MANIFEST_NAME).read_bytes() == digest_before


def test_failed_calibration_forward_leaves_no_manifest(tmp_path):
    out = tmp_path / "failed"
    out.mkdir()
    model = _tiny_model()
    quantize(model, ACTIVATION_CFG, quant_result_dir=str(out))
    with pytest.raises(RuntimeError):
        model(torch.randn(8, 999).to(torch.bfloat16))  # wrong in_features
    assert not (out / MANIFEST_NAME).exists()


def test_one_failed_unit_write_leaves_no_manifest(tmp_path):
    out = tmp_path / "partial"
    out.mkdir()
    model = _tiny_model()
    from safetensors.torch import save_file as real_save

    calls = {"n": 0}

    def fail_on_second(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 2:
            raise OSError("disk full")
        return real_save(*args, **kwargs)

    import safetensors.torch as safetensors_torch

    with patch.object(safetensors_torch, "save_file", fail_on_second):
        with pytest.raises(OSError):
            quantize(model, WEIGHT_ONLY_CFG, quant_result_dir=str(out))

    names = sorted(p.name for p in out.iterdir())
    assert MANIFEST_NAME not in names, "an incomplete run must not publish"
    assert names == ["fc1.safetensors"], "no temp file may survive the failure"
    with pytest.raises(FileNotFoundError, match="not a valid quant result"):
        load_quant_manifest(str(out))


def test_quantize_records_source_model_fingerprint(tmp_path):
    out = tmp_path / "fp"
    out.mkdir()
    model = _tiny_model()
    expected = compute_model_fingerprint(_tiny_model_like(model))
    quantize(model, WEIGHT_ONLY_CFG, quant_result_dir=str(out))
    manifest = load_quant_manifest(str(out))
    # Fingerprint is taken before MinMaxQuant replaces the Linear modules, so
    # it describes the source model, not the rewritten one.
    assert manifest["source_model"] == expected
    assert manifest["source_model"]["model_type"] == "_TinyModel"


def _tiny_model_like(model):
    """A pristine copy of the pre-quantize model, for fingerprint comparison."""
    clone = _tiny_model()
    clone.load_state_dict(
        {
            "fc1.weight": model.fc1.weight.detach().clone(),
            "fc1.bias": model.fc1.bias.detach().clone(),
            "fc2.weight": model.fc2.weight.detach().clone(),
            "fc2.bias": model.fc2.bias.detach().clone(),
        }
    )
    return clone


AWQ_CFG = {
    'batch_num': 1,
    'quant_cfg': {
        'weights': {
            'type': 'int4',
            'symmetric': True,
            'strategy': 'channel',
        }
    },
    'algorithm': {'awq': {'grids_num': 20}},
}


SMOOTHQUANT_CFG = {
    'batch_num': 1,
    'quant_cfg': {
        'weights': {'type': 'int8', 'symmetric': True, 'strategy': 'channel'},
        'inputs': {'type': 'int8', 'symmetric': True, 'strategy': 'tensor'},
    },
    'algorithm': {'smoothquant': {'smooth_strength': 0.5}},
}


class _LinearShapedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(64, 64, bias=False)

    def forward(self, inputs):
        return self.fc1(inputs)


def test_quantize_rejects_an_algorithm_without_export_support(tmp_path):
    """An algorithm that cannot dump must fail before writing any result.

    Regression guard: every quantize module inherits BaseQuantizeModule, so a
    mere isinstance or hasattr check lets an unwired algorithm through.
    quantize() then returned normally, the result directory stayed empty, and the
    manifest was pending forever -- an API that reports success while producing
    nothing loadable. SmoothQuant stands in for "not wired yet" now that AWQ and
    GPTQ are.
    """
    out = tmp_path / "smoothquant"
    out.mkdir()
    model = _LinearShapedModel().to(torch.bfloat16)
    with pytest.raises(ValueError, match="not supported yet for algorithm") as ctx:
        quantize(model, SMOOTHQUANT_CFG, quant_result_dir=str(out))

    assert "_supports_quant_result_export" not in str(ctx.value)
    assert list(out.iterdir()) == [], "a rejected algorithm must write nothing"
    assert type(model.fc1).__name__ == "Linear", "the model must be left untouched"


def test_the_same_unsupported_algorithm_still_works_without_a_result_dir(tmp_path):
    model = _LinearShapedModel().to(torch.bfloat16)
    quantize(model, SMOOTHQUANT_CFG)
    # The legacy in-memory path is unaffected by the export screening.
    assert type(model.fc1).__name__ == "SmoothQuant"


def test_awq_is_now_accepted_for_export(tmp_path):
    """AWQ is wired in Week 4: it must publish a real artifact."""
    out = tmp_path / "awq_ok"
    out.mkdir()
    model = _LinearShapedModel().to(torch.bfloat16)
    quantize(model, AWQ_CFG, quant_result_dir=str(out))
    model(torch.randn(8, 64).to(torch.bfloat16))
    assert (out / "quant_manifest.json").is_file()


def test_artifact_is_consumable_by_the_reload_contract(tmp_path):
    """Walk the spec's reload validation chain and rebuild from the manifest.

    Week 4 owns ``reload_quant_params()``; this asserts the Week 3 artifact
    already carries everything that chain needs, so the schema cannot drift out
    from under it. The rebuild uses only manifest fields -- no knowledge of how
    the artifact was produced.
    """
    out = tmp_path / "consume"
    out.mkdir()
    model = _tiny_model()
    reference = _tiny_model_like(model)
    quantize(model, ACTIVATION_CFG, quant_result_dir=str(out))
    inputs = _tiny_inputs()
    model(inputs)
    model(inputs)
    expected = model(inputs)

    # 1. Manifest format, version, required fields, uniqueness, safe paths.
    manifest = load_quant_manifest(str(out))
    # 2. The directory holds the manifest and every unit it manages.
    validate_result_directory(str(out), manifest)
    # 3. The source model identity matches the untouched float model.
    assert manifest["source_model"] == compute_model_fingerprint(reference)

    helper = ModuleHelper(reference)
    for entry in manifest["units"]:
        layer_name = entry["layer_name"]
        assert layer_name in helper.named_module_dict
        # 4-6. Size, digest, header and tensor schema, all checked by the reader.
        tensors, non_tensor_params, _ = read_quant_unit(str(out), layer_name, manifest)

        # A Week 4 loader resolves the class from quant_module_type; MinMaxQuant
        # is the only registered name here, so assert the recorded value and use
        # it rather than hardcoding the class at the call site.
        assert entry["quant_module_type"] == "MinMaxQuant"
        assert entry["ori_module_type"] == "Linear"
        ori_module = helper.named_module_dict[layer_name]
        rebuilt = MinMaxQuant(ori_module, layer_name, entry["quant_config"])
        rebuilt.load_quant_params(tensors, non_tensor_params)
        ModuleHelper.replace_module_by_name(reference, layer_name, rebuilt)

    # The rebuilt model fake-quants identically, without recalibrating.
    assert torch.allclose(reference(inputs), expected)


def _npu_available():
    """Return whether torch exposes a usable, non-mocked NPU backend."""
    if not hasattr(torch, "npu"):
        return False
    try:
        if torch.npu.is_available() is not True:
            return False
        torch.zeros(1).npu()
    except Exception:
        return False
    return True


def test_npu_available_rejects_mocked_backend():
    with patch.object(torch, "npu", MagicMock(), create=True):
        assert not _npu_available()


@pytest.mark.npu
def test_npu_round_trip_restores_params_to_the_device(tmp_path):
    """On real NPU hardware, units persist as CPU tensors and reload to npu.

    Units are always stored CPU-side (device is not part of the artifact), so
    this is the check that ``_quant_param_device()`` puts them back on the
    module's device rather than leaving a CPU tensor in an NPU module.
    """
    if not _npu_available():
        pytest.skip("no Ascend NPU available")

    out = tmp_path / "npu"
    out.mkdir()
    model = _tiny_model().npu()
    reference = _tiny_model_like(model).npu()
    quantize(model, ACTIVATION_CFG, quant_result_dir=str(out))

    inputs = _tiny_inputs().npu()
    model(inputs)
    model(inputs)
    expected = model(inputs)

    manifest = load_quant_manifest(str(out))
    validate_result_directory(str(out), manifest)
    helper = ModuleHelper(reference)
    for entry in manifest["units"]:
        layer_name = entry["layer_name"]
        tensors, non_tensor_params, _ = read_quant_unit(str(out), layer_name, manifest)
        assert all(t.device.type == "cpu" for t in tensors.values())
        rebuilt = MinMaxQuant(
            helper.named_module_dict[layer_name], layer_name, entry["quant_config"]
        )
        rebuilt.load_quant_params(tensors, non_tensor_params)
        assert rebuilt.scale_w.device == rebuilt.weight.device
        assert rebuilt.scale_d.device == rebuilt.weight.device
        ModuleHelper.replace_module_by_name(reference, layer_name, rebuilt)

    assert torch.equal(reference(inputs), expected)


def test_pass_attaches_the_writer_without_exposing_capability_flags():
    class _RecordingQuantOp(QuantResultSupportMixin, nn.Module):
        def __init__(self, ori_module, layer_name, quant_config):
            super().__init__()
            self.ori_module = ori_module
            self.attached_writer = None

        def attach_quant_result_writer(self, writer):
            self.attached_writer = writer

        def forward(self, inputs):
            return self.ori_module(inputs)

    model = _tiny_model()
    writer = MagicMock()
    quant_pass = InsertQuantizeModulePass(
        {"fc1": {"algorithm": {"minmax": {}}}}, quant_result_writer=writer
    )
    quant_pass.quantize_ops["fc1"] = _RecordingQuantOp
    quant_pass.do_pass(model, model.fc1, "fc1")
    assert model.fc1.attached_writer is writer


def test_quantize_rejects_shared_input_fused_quantization_before_model_rewrite(
    tmp_path,
):
    """Fused activation calibration cannot yet be represented by an artifact."""
    out = tmp_path / "fused"
    out.mkdir()
    model = _tiny_model()
    # Simulate the state after InsertLayerNormQuantPass: members are managed by
    # LayerNormQuant and never run their own activation finalization branch.
    model.fc1.shared_param_mode = "pending"
    with pytest.raises(ValueError, match="shared-input fused quantization") as ctx:
        quantize(model, ACTIVATION_CFG, quant_result_dir=str(out))

    assert "Run without quant_result_dir" in str(ctx.value)
    assert list(out.iterdir()) == []
    assert type(model.fc1).__name__ == "Linear"


def test_shared_input_fusion_still_works_without_a_result_dir():
    model = _tiny_model()
    model.fc1.shared_param_mode = "pending"
    quantize(model, ACTIVATION_CFG)
    assert type(model.fc1).__name__ == "MinMaxQuant"


def test_incomplete_calibration_error_mentions_unpublished_units(tmp_path):
    out = tmp_path / "incomplete"
    out.mkdir()
    model = _tiny_model()
    quantize(model, ACTIVATION_CFG, quant_result_dir=str(out))
    assert list(out.iterdir()) == []

    # The first forward of a two-batch calibration has not finalized the
    # activation parameters, so no unit or manifest exists yet. Create one
    # representative unit file to model the partial-run directory that
    # convert() must diagnose.
    model(_tiny_inputs())
    (out / "fc1.safetensors").write_bytes(b"partial unit")
    assert (out / "fc1.safetensors").is_file()
    assert not (out / MANIFEST_NAME).exists()

    with pytest.raises(FileNotFoundError, match="Unpublished unit file") as error:
        load_quant_manifest(str(out))
    message = str(error.value)
    assert "quant_manifest.json" in message
    assert "fc1.safetensors" in message
    assert "Rerun calibration" in message
