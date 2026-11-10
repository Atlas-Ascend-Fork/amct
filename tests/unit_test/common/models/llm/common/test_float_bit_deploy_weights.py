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

"""CPU mixed-bit deployment payloads, before any external model loader."""

import importlib
import types

import pytest
import torch
from safetensors.torch import load_file, save_file


class HostMLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_size = self.intermediate_size = 2
        self.act_fn = torch.nn.functional.silu
        self.gate_proj = torch.nn.Linear(2, 2, bias=False)
        self.up_proj = torch.nn.Linear(2, 2, bias=False)
        self.down_proj = torch.nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            for module in [self.gate_proj, self.up_proj, self.down_proj]:
                module.weight.copy_(torch.tensor([[1.0, -0.5], [0.25, 0.75]]))


class HostExportBoundary:
    """Supplied CPU weights bypass external checkpoint/model loading only."""

    def __init__(self, block):
        self.block = block
        self.args = types.SimpleNamespace(device="cpu")

    def build_quant_block(self, layer_idx):
        return self.block

    def load_selected_layer_ptq_params(self, *args, **kwargs):
        return None

    def get_layer_weight_prefix(self, layer_idx):
        return "model.layers.0.mlp."


def load_export_modules():
    importlib.import_module("amct_pytorch.quantization.dtypes.int")
    return {
        "policy": importlib.import_module("amct_pytorch.quantization.bit_policy"),
        "apply": importlib.import_module(
            "amct_pytorch.common.models.llm.common.quant_apply"
        ),
        "export": importlib.import_module(
            "amct_pytorch.common.models.llm.common.deployment.deploy_export"
        ),
        "linear": importlib.import_module(
            "amct_pytorch.quantization.modules.quant_linear"
        ),
    }


@pytest.fixture
def export_api(monkeypatch):
    loaded = load_export_modules()
    base = importlib.import_module("amct_pytorch.common.models.llm.common.base")
    monkeypatch.setattr(
        HostExportBoundary,
        "iter_deploy_bindings",
        base.BaseModel.iter_deploy_bindings,
        raising=False,
    )
    return loaded


def quant_block(api, config):
    arguments = types.SimpleNamespace(
        w_bits=8,
        a_bits=8,
        quant_dtype="int",
        algos=[],
        bit_policy=api.get("policy").BitPolicy(config),
    )
    with torch.random.fork_rng(devices=[]):
        return api.get("apply").QuantGatedMLP(arguments, HostMLP())


def export_payload(api, block):
    ignored = []
    tensors, routes = api.get("export").export_block_deploy(
        HostExportBoundary(block), 0, ignored
    )
    return tensors, routes, ignored


def test_mixed_policy_exports_float_gate_and_marks_it_ignored(export_api):
    block = quant_block(
        export_api,
        {"w_bits": 8, "a_bits": 8, "mlp": {"gate_proj": {"w_bits": 16, "a_bits": 16}}},
    )

    tensors, routes, ignored = export_payload(export_api, block)

    key = "model.layers.0.mlp.gate_proj.weight"
    assert tensors.get(key).dtype == torch.float32
    assert torch.equal(tensors.get(key), block.gate_proj.linear.weight)
    assert key.removesuffix(".weight") in ignored
    assert key.removesuffix(".weight") in export_api.get(
        "export"
    ).generate_quant_config(None, ignored).get("ignore")
    assert key.removesuffix(".weight") + ".weight_scale" not in tensors
    assert routes.get(key) == key
    assert tensors.get("model.layers.0.mlp.up_proj.weight").dtype == torch.int8


def test_all_float_policy_preserves_every_weight_and_ignores_every_projection(
    export_api,
):
    block = quant_block(export_api, {"w_bits": 16, "a_bits": 16})

    tensors, routes, ignored = export_payload(export_api, block)

    names = ["gate_proj", "up_proj", "down_proj"]
    assert set(tensors) == {f"model.layers.0.mlp.{name}.weight" for name in names}
    assert set(ignored) == {f"model.layers.0.mlp.{name}" for name in names}
    for name in names:
        key = f"model.layers.0.mlp.{name}.weight"
        assert tensors.get(key).dtype == torch.float32
        assert torch.equal(tensors.get(key), getattr(block, name).linear.weight)
        assert routes.get(key) == key


def test_int8_policy_keeps_quantized_payload_and_scale(export_api):
    block = quant_block(export_api, {"w_bits": 8, "a_bits": 8})

    tensors, _, ignored = export_payload(export_api, block)

    assert ignored == []
    for name in ["gate_proj", "up_proj", "down_proj"]:
        key = f"model.layers.0.mlp.{name}.weight"
        assert torch.equal(
            tensors.get(key), torch.tensor([[127, -64], [42, 127]], dtype=torch.int8)
        )
        assert torch.allclose(
            tensors.get(key.replace(".weight", ".weight_scale")),
            torch.tensor([[1 / 127], [0.75 / 127]], dtype=torch.float32),
        )


def test_float_export_preserves_supported_structure_transform_and_forward(export_api):
    block = quant_block(export_api, {"w_bits": 16, "a_bits": 16})

    def transform(weight, *, inv_t, name):
        assert inv_t and name == "gate_proj"
        return weight + 0.125

    inputs = torch.tensor([[1.0, -1.0]])

    output = block.gate_proj(inputs, structure_transform=transform)
    payload = block.gate_proj.export_deploy(structure_transform=transform)

    expected = torch.tensor([[1.125, -0.375], [0.375, 0.875]])
    assert torch.equal(payload.get("qweight"), expected)
    assert set(payload) == {"qweight"}
    assert torch.equal(
        torch.nn.functional.linear(inputs, payload.get("qweight")), output
    )


def test_plain_and_raw_linears_keep_ignore_entries(export_api):
    with torch.random.fork_rng(devices=[]):
        block = torch.nn.ModuleDict(
            {
                "plain": export_api.get("apply").PlainLinear(torch.nn.Linear(2, 2)),
                "raw": torch.nn.Linear(2, 2),
            }
        )

    ignored = export_api.get("export").get_quant_ignore_linear_names(
        block, "model.layers.0."
    )

    assert set(ignored) == {"model.layers.0.plain", "model.layers.0.raw"}


def test_mixed_payload_safetensors_roundtrip_keeps_float_weight(export_api, tmp_path):
    block = quant_block(
        export_api,
        {"w_bits": 8, "a_bits": 8, "mlp": {"gate_proj": {"w_bits": 16, "a_bits": 16}}},
    )
    tensors, _, _ = export_payload(export_api, block)
    checkpoint = tmp_path / "mixed.safetensors"

    save_file(tensors, str(checkpoint))
    restored = load_file(str(checkpoint))

    gate = restored.get("model.layers.0.mlp.gate_proj.weight")
    assert gate.dtype == torch.float32 and torch.equal(
        gate, block.gate_proj.linear.weight
    )
    assert restored.get("model.layers.0.mlp.up_proj.weight").dtype == torch.int8


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_float_autoround_export_keeps_learned_clipping(export_api, dtype):
    importlib.import_module("amct_pytorch.algorithms.quant.auto_round")
    base = importlib.import_module("amct_pytorch.quantization.modules.quant_base")
    arguments = types.SimpleNamespace(
        w_bits=16,
        a_bits=16,
        w_size=(2, 2),
        quant_dtype="int",
        algos=["autoround"],
        quant_target=["weight"],
    )
    quantizer = base.WeightQuantizer(arguments)
    algorithm = quantizer.algorithms["autoround"]
    with torch.no_grad():
        algorithm.min_scale.fill_(0.5)
        algorithm.max_scale.fill_(0.5)
    weight = torch.tensor([[1.0, -0.5], [0.25, 0.75]], dtype=dtype)
    expected = quantizer(weight).detach()
    assert not torch.equal(expected, weight), "fixture must exercise learned clipping"

    payload = quantizer.export_deploy(weight)

    assert set(payload) == {"qweight"}
    assert payload["qweight"].dtype == dtype
    assert payload["qweight"].device.type == "cpu"
    assert torch.equal(payload["qweight"], expected)
    inputs = torch.tensor([[0.25, -0.5]], dtype=dtype)
    assert torch.equal(
        torch.nn.functional.linear(inputs, payload["qweight"]),
        torch.nn.functional.linear(inputs, expected),
    )


@pytest.mark.parametrize("weight_bits,activation_bits", [(16, 8), (8, 16)])
def test_mixed_precision_metadata_keeps_each_active_quantizer(
    export_api, tmp_path, weight_bits, activation_bits
):
    import json

    block = quant_block(export_api, {"w_bits": weight_bits, "a_bits": activation_bits})
    ignored, linear_bits = [], {}
    tensors, _ = export_api.get("export").export_block_deploy(
        HostExportBoundary(block), 0, ignored, linear_bits=linear_bits
    )
    config = export_api.get("export").generate_quant_config(
        None, ignored, linear_bits=linear_bits
    )
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"quantization_config": config}), encoding="utf-8")
    restored = json.loads(path.read_text(encoding="utf-8"))["quantization_config"]
    assert ignored == [] and restored["ignore"] == []
    by_target = {
        target: group
        for group in restored["config_groups"].values()
        for target in group["targets"]
    }
    assert "Linear" not in by_target
    for name in ("gate_proj", "up_proj", "down_proj"):
        target = "model.layers.0.mlp." + name
        module = getattr(block, name)
        assert (module.w_bits, module.a_bits) == (weight_bits, activation_bits)
        assert linear_bits[target] == (weight_bits, activation_bits)
        group = by_target[target]
        if weight_bits == 16:
            assert group["weights"] is None
            assert torch.equal(tensors[target + ".weight"], module.linear.weight)
        else:
            assert group["weights"]["num_bits"] == weight_bits
            assert tensors[target + ".weight"].dtype == torch.int8
        if activation_bits == 16:
            assert group["input_activations"] is None
        else:
            assert group["input_activations"]["num_bits"] == activation_bits
    assert block.input_quant.bits == block.gate_proj.a_bits
    assert block.hidden_quant.bits == block.down_proj.a_bits
    inputs = torch.tensor([[0.31, -0.97]])
    quantized_inputs = block.input_quant(inputs)
    assert torch.isfinite(block(inputs)).all()
    assert torch.equal(quantized_inputs, inputs) == (activation_bits == 16)


def test_shared_gate_input_metadata_uses_real_activation_bits(export_api):
    block = quant_block(
        export_api,
        {"w_bits": 8, "a_bits": 8, "mlp": {"gate_proj": {"w_bits": 16, "a_bits": 16}}},
    )
    # Both gate and up projections consume input_quant, whose policy is gate.a.
    assert block.up_proj.a_bits == block.input_quant.bits == 16
    assert block.down_proj.a_bits == block.hidden_quant.bits == 8


def test_w16a8_keeps_real_activation_quantization_enabled(export_api):
    block = quant_block(export_api, {"w_bits": 16, "a_bits": 8})
    tensors, _, ignored = export_payload(export_api, block)
    assert block.input_quant.bits == block.hidden_quant.bits == 8
    assert ignored == [], (
        "floating weights do not disable 8-bit activation quantization"
    )
    for name in ("gate_proj", "up_proj", "down_proj"):
        target = "model.layers.0.mlp." + name + ".weight"
        assert torch.equal(tensors[target], getattr(block, name).linear.weight)


def test_activation_only_export_parses_with_compressed_tensors_schema(export_api):
    from compressed_tensors.quantization import QuantizationConfig

    block = quant_block(export_api, {"w_bits": 16, "a_bits": 8})
    ignored, bits = [], {}
    export_api.get("export").export_block_deploy(
        HostExportBoundary(block), 0, ignored, linear_bits=bits
    )
    config = export_api.get("export").generate_quant_config(
        None, ignored, linear_bits=bits
    )
    parsed = QuantizationConfig.model_validate(config)
    target = "model.layers.0.mlp.gate_proj"
    matches = [
        group for group in parsed.config_groups.values() if target in group.targets
    ]
    assert len(matches) == 1
    assert matches[0].weights is None
    assert matches[0].input_activations.num_bits == 8


@pytest.mark.parametrize(
    "weight_bits,activation_bits", [(8, 8), (16, 8), (8, 16), (16, 16)]
)
def test_moe_config_targets_match_real_export_bindings(
    export_api, weight_bits, activation_bits
):
    from amct_pytorch.common.models.llm.qwen.qwen3.qwen3_moe import Qwen3Moe

    class MoeBoundary(HostExportBoundary):
        iter_deploy_bindings = Qwen3Moe.iter_deploy_bindings

        def get_layer_weight_prefix(self, layer_idx):
            return "model.layers.0."

    block = torch.nn.Module()
    block.mlp = torch.nn.Module()
    block.mlp.experts = torch.nn.Module()
    block.mlp.experts.expert_modules = torch.nn.ModuleList(
        [
            quant_block(export_api, {"w_bits": weight_bits, "a_bits": activation_bits})
            for _ in range(2)
        ]
    )
    ignored, bits = [], {}
    tensors, _ = export_api["export"].export_block_deploy(
        MoeBoundary(block), 0, ignored, linear_bits=bits
    )
    config = export_api["export"].generate_quant_config(None, ignored, linear_bits=bits)
    expected = {
        f"model.layers.0.mlp.experts.{idx}.{projection}"
        for idx in range(2)
        for projection in ("gate_proj", "up_proj", "down_proj")
    }
    assert set(bits) == expected
    assert {
        key.removesuffix(".weight") for key in tensors if key.endswith(".weight")
    } == expected
    assert all("expert_modules" not in name for name in bits)
    targets = {
        target: group
        for group in config["config_groups"].values()
        for target in group["targets"]
    }
    assert "Linear" not in targets
    if (weight_bits, activation_bits) == (16, 16):
        assert set(ignored) == expected
        assert expected.isdisjoint(targets)
    else:
        assert ignored == []
        for target in expected:
            assert bits[target] == (weight_bits, activation_bits)
            assert (
                targets[target]["weights"] is None
                if weight_bits == 16
                else targets[target]["weights"]["num_bits"] == weight_bits
            )
            assert (
                targets[target]["input_activations"] is None
                if activation_bits == 16
                else targets[target]["input_activations"]["num_bits"] == activation_bits
            )


def test_quantlinear_requires_explicit_activation_bits(export_api):
    # CLI arguments do not define a_bits; omitted metadata must never silently become A16.
    args = types.SimpleNamespace(quant_dtype="int", algos=[], w_bits=8)
    with pytest.raises(TypeError, match="a_bits"):
        export_api["linear"].QuantLinear(args, torch.nn.Linear(2, 2), w_bits=8)
