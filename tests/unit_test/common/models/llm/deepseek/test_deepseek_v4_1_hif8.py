# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
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
)
from amct_pytorch.common.models.llm.deepseek.deepseek_v4_1.deepseekv4_1 import (
    DeepseekV41,
)
from amct_pytorch.quantization.dtypes import hifp_impl
from amct_pytorch.quantization.dtypes.hifp_impl import (
    HIF8_WEIGHT_SCOPE,
    hif8_pack,
    is_native_hifloat8_cast_available,
)

SHARD = "model.safetensors"
GROUPS, RANK, COLUMNS = 2, 32, 64
WO_A, PLAIN = "layers.0.attn.wo_a", "layers.0.attn.wq_a"
UNSELECTED = "layers.0.attn.wkv"
UNTOUCHED = "layers.0.attn.norm.weight"

requires_hifloat8_npu = pytest.mark.skipif(
    not is_native_hifloat8_cast_available(),
    reason="HiFloat8 pack requires native HiFloat8 cast support in torch_npu",
)


@pytest.fixture
def fake_hifloat8_cast(monkeypatch):
    import torch_npu

    def cast(tensor, dtype, **kwargs):
        return tensor.float().abs().clamp(max=127).to(torch.uint8)

    monkeypatch.setattr(hifp_impl, "is_native_hifloat8_cast_available", lambda: True)
    monkeypatch.setattr(torch_npu, "npu_dtype_cast", cast, raising=False)
    monkeypatch.setattr(torch_npu, "hifloat8", object(), raising=False)
    monkeypatch.setattr(torch.Tensor, "npu", lambda self, *a, **k: self, raising=False)


@pytest.mark.npu
@requires_hifloat8_npu
def test_hif8_pack_round_trips_on_npu():
    import torch_npu

    weight = torch.randn(8, 64) * torch.arange(1, 9).view(-1, 1)
    codes, scale = hif8_pack(weight)
    decoded = torch_npu.npu_dtype_cast(
        codes.npu(), torch.float32, input_dtype=torch_npu.hifloat8
    ).cpu()
    peak = decoded.abs().amax(dim=1)
    assert torch.allclose(peak, torch.full((8,), HIF8_WEIGHT_SCOPE), rtol=0.1)
    assert torch.allclose(decoded * scale, weight, rtol=0.1, atol=1e-3)


def test_hif8_pack_returns_codes_and_row_scales(fake_hifloat8_cast):
    weight = torch.randn(8, 64) * torch.arange(1, 9).view(-1, 1)
    codes, scale = hif8_pack(weight)

    assert codes.dtype == torch.uint8 and codes.shape == weight.shape
    assert scale.dtype == torch.float32 and scale.shape == (8, 1)
    expected = weight.abs().amax(dim=1, keepdim=True) / HIF8_WEIGHT_SCOPE
    assert torch.allclose(scale, expected)


def test_hif8_pack_rejects_non_2d_input():
    with pytest.raises(RuntimeError, match="2-D"):
        hif8_pack(torch.ones(4))


def test_hif8_pack_rejects_non_uint8_codes(fake_hifloat8_cast, monkeypatch):
    import torch_npu

    monkeypatch.setattr(torch_npu, "npu_dtype_cast", lambda t, d, **k: t)
    with pytest.raises(RuntimeError, match="uint8"):
        hif8_pack(torch.ones(2, 2))


def test_hif8_pack_requires_native_cast(monkeypatch):
    monkeypatch.setattr(hifp_impl, "is_native_hifloat8_cast_available", lambda: False)
    with pytest.raises(RuntimeError, match="torch_npu"):
        hif8_pack(torch.ones(2, 2))


def _block_fp8(rows, columns=COLUMNS):
    weight = torch.ones((rows, columns), dtype=torch.float8_e4m3fn)
    scale = torch.full((rows // 32, columns // 32), 127, dtype=torch.uint8)
    return weight, scale.view(torch.float8_e8m0fnu)


def _hif8_model(tmp_path, selected=(PLAIN, WO_A), groups=GROUPS):
    model = DeepseekV41.__new__(DeepseekV41)
    model.args = SimpleNamespace(
        model_name="deepseek_v4_1",
        deploy_platform="ascend950",
        quant_dtype="hifp",
        deploy_selection=DeploySelection(selected),
    )
    model.model_path, model.output_dir = tmp_path / "model", tmp_path / "output"
    model.model_path.mkdir()
    model.output_dir.mkdir()
    model.processed_modules = set()
    model.config = {
        "text_config": {"o_groups": groups, "o_lora_rank": RANK},
        "quantization_config": {"quant_method": "fp8"},
    }

    tensors = {UNTOUCHED: torch.ones(COLUMNS, dtype=torch.bfloat16)}
    for name in (PLAIN, WO_A, UNSELECTED):
        weight, scale = _block_fp8(GROUPS * RANK)
        tensors[f"{name}.weight"], tensors[f"{name}.scale"] = weight, scale
    path = model.model_path / SHARD
    save_file(tensors, str(path), metadata={"format": "pt"})
    model.safetensors_files = frozenset({path.resolve()})
    return model, tensors


def test_convert_hif8_shard_writes_codes_and_decodes_unselected_weights(
    tmp_path, fake_hifloat8_cast
):
    model, source = _hif8_model(tmp_path)
    source["orphan.scale"] = torch.tensor([1.25, 2.5], dtype=torch.float32)
    save_file(source, str(model.model_path / SHARD))
    routing = model.convert_tensorwise_shard(
        SHARD, model.model_path, dict.fromkeys(source, SHARD), {PLAIN: 8, WO_A: 8}, {}
    )

    expected_names = {
        f"{PLAIN}.weight",
        f"{PLAIN}.weight_scale",
        f"{WO_A}.weight",
        f"{WO_A}.weight_scale",
        f"{UNSELECTED}.weight",
        UNTOUCHED,
        "orphan.scale",
    }
    assert set(routing) == expected_names
    assert model.processed_modules == {PLAIN, WO_A}
    with safe_open(str(model.output_dir / SHARD), framework="pt") as shard:
        assert shard.get_slice(f"{PLAIN}.weight").get_dtype() == "U8"
        assert shard.get_slice(f"{PLAIN}.weight_scale").get_dtype() == "F32"
        assert shard.get_slice(f"{WO_A}.weight").get_shape() == [GROUPS * RANK, COLUMNS]
        scale = shard.get_tensor(f"{WO_A}.weight_scale")
        unselected = shard.get_tensor(f"{UNSELECTED}.weight")
        untouched = shard.get_tensor(UNTOUCHED)
        orphan_scale = shard.get_tensor("orphan.scale")
    per_group = scale.view(GROUPS, RANK)
    assert torch.equal(per_group, per_group[0].expand(GROUPS, RANK))
    assert unselected.dtype == torch.bfloat16
    assert torch.equal(unselected, torch.ones_like(unselected))
    assert torch.equal(untouched, source[UNTOUCHED])
    assert orphan_scale.dtype == torch.float32
    assert torch.equal(orphan_scale, source["orphan.scale"])


def test_convert_hif8_reads_scale_from_another_shard(tmp_path, fake_hifloat8_cast):
    model, _ = _hif8_model(tmp_path, selected=(PLAIN,))
    weight, scale = _block_fp8(GROUPS * RANK)
    other = "model-00002.safetensors"
    save_file({f"{PLAIN}.weight": weight}, str(model.model_path / SHARD))
    save_file(
        {f"{PLAIN}.scale": scale, UNTOUCHED: torch.ones(COLUMNS)},
        str(model.model_path / other),
    )
    model.safetensors_files = frozenset(
        (model.model_path / name).resolve() for name in (SHARD, other)
    )
    weight_map = {f"{PLAIN}.weight": SHARD, f"{PLAIN}.scale": other, UNTOUCHED: other}

    routing = {}
    for shard in (SHARD, other):
        routing.update(
            model.convert_tensorwise_shard(
                shard, model.model_path, weight_map, {PLAIN: 8}, {}
            )
        )

    assert routing == {
        f"{PLAIN}.weight": SHARD,
        f"{PLAIN}.weight_scale": SHARD,
        UNTOUCHED: other,
    }
    with safe_open(str(model.output_dir / SHARD), framework="pt") as shard:
        assert shard.get_slice(f"{PLAIN}.weight").get_dtype() == "U8"
        assert shard.get_slice(f"{PLAIN}.weight_scale").get_dtype() == "F32"
    with safe_open(str(model.output_dir / other), framework="pt") as shard:
        assert set(shard.keys()) == {UNTOUCHED}


def test_convert_hif8_shard_rejects_missing_scale(tmp_path, fake_hifloat8_cast):
    model, _ = _hif8_model(tmp_path)
    weight, _ = _block_fp8(32)
    save_file({f"{PLAIN}.weight": weight}, str(model.model_path / SHARD))

    with pytest.raises(ValueError, match="Missing source scale"):
        model.convert_tensorwise_shard(SHARD, model.model_path, {}, {PLAIN: 8}, {})


def test_convert_hif8_shard_rejects_wo_a_not_split_by_groups(
    tmp_path, fake_hifloat8_cast
):
    model, source = _hif8_model(tmp_path, groups=GROUPS + 1)

    with pytest.raises(ValueError, match="o_groups \\* o_lora_rank"):
        model.convert_tensorwise_shard(
            SHARD, model.model_path, dict.fromkeys(source, SHARD), {WO_A: 8}, {}
        )


def test_prepare_hif8_plan_keeps_only_selected_modules(tmp_path):
    model, source = _hif8_model(tmp_path)
    plan = model.prepare_tensor_deploy_plan(dict.fromkeys(source, SHARD))

    assert plan.selected_modules == {PLAIN, WO_A}
    assert UNSELECTED in plan.unselected_weight_modules


def test_prepare_hif8_plan_rejects_selected_non_matrix_module(tmp_path):
    model, source = _hif8_model(tmp_path, selected=(PLAIN, "layers.0.attn.norm"))

    with pytest.raises(ValueError, match="not nonempty 2-D weights"):
        model.prepare_tensor_deploy_plan(dict.fromkeys(source, SHARD))


def test_refresh_config_hif8_writes_source_config_unchanged(tmp_path):
    model, _ = _hif8_model(tmp_path)
    original = json.loads(json.dumps(model.config))

    model.refresh_config([], tensor_plan=None)

    written = json.loads((model.output_dir / "config.json").read_text())
    assert model.config == original
    expected = dict(original)
    expected.pop("quantization_config", None)
    assert written == expected


@pytest.mark.parametrize("value", [None, "2"])
def test_convert_hif8_shard_rejects_invalid_o_groups(
    tmp_path, fake_hifloat8_cast, value
):
    model, source = _hif8_model(tmp_path)
    model.config["text_config"]["o_groups"] = value

    with pytest.raises(ValueError, match="integer 'o_groups'"):
        model.convert_tensorwise_shard(
            SHARD, model.model_path, dict.fromkeys(source, SHARD), {WO_A: 8}, {}
        )


def test_validate_deploy_args_accepts_hifp_tensor_deploy(tmp_path):
    from amct_pytorch.common.models.llm.common.deployment.deploy_selection import (
        QuantLayerRule,
        QuantSpec,
    )

    model, _ = _hif8_model(tmp_path)
    model.args.granularity = "tensor"
    model.args.quant_layers_config = None
    model.args.deploy_selection = DeploySelection(
        (QuantLayerRule(PLAIN, QuantSpec("hifp", 8, 8)),)
    )
    DeepseekV41.validate_deploy_args(model.args)
    assert model.args.quant_dtype == "hifp"
