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

import ast
import importlib
import sys
import types
from pathlib import Path

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


def install_package_namespaces(root, monkeypatch):
    packages = [
        "amct_pytorch",
        "amct_pytorch.common",
        "amct_pytorch.common.utils",
        "amct_pytorch.common.models",
        "amct_pytorch.common.models.llm",
        "amct_pytorch.common.models.llm.common",
        "amct_pytorch.quantization",
        "amct_pytorch.quantization.modules",
        "amct_pytorch.algorithms",
    ]
    for name in packages:
        package = types.ModuleType(name)
        package.__path__ = [str(root / name.replace(".", "/"))]
        monkeypatch.setitem(sys.modules, name, package)


def load_export_modules():
    importlib.import_module("amct_pytorch.quantization.dtypes.int")
    return {
        "policy": importlib.import_module("amct_pytorch.quantization.bit_policy"),
        "apply": importlib.import_module(
            "amct_pytorch.common.models.llm.common.quant_apply"
        ),
        "export": importlib.import_module(
            "amct_pytorch.common.models.llm.common.deploy_export"
        ),
        "linear": importlib.import_module(
            "amct_pytorch.quantization.modules.quant_linear"
        ),
    }


def install_complete_binding_method(root, loaded, monkeypatch):
    text = (root / "amct_pytorch/common/models/llm/common/base.py").read_text()
    owner = next(
        node
        for node in ast.parse(text).body
        if isinstance(node, ast.ClassDef) and node.name == "BaseModel"
    )
    binding = next(
        node
        for node in owner.body
        if isinstance(node, ast.FunctionDef) and node.name == "iter_deploy_bindings"
    )
    namespace = {"QuantLinear": loaded.get("linear").QuantLinear}
    exec(
        compile(
            ast.Module(body=[binding], type_ignores=[]),
            "complete BaseModel.iter_deploy_bindings",
            "exec",
        ),
        namespace,
    )
    method = namespace.get("iter_deploy_bindings")
    assert callable(method)
    monkeypatch.setattr(
        HostExportBoundary, "iter_deploy_bindings", method, raising=False
    )


@pytest.fixture
def export_api(monkeypatch):
    root = Path(__file__).resolve().parents[6]
    previous = {
        name: value
        for name, value in sys.modules.items()
        if name.startswith("amct_pytorch")
    }
    try:
        install_package_namespaces(root, monkeypatch)
        loaded = load_export_modules()
        install_complete_binding_method(root, loaded, monkeypatch)
        yield loaded
    finally:
        for name in list(sys.modules):
            if name.startswith("amct_pytorch") and name not in previous:
                sys.modules.pop(name, None)
        sys.modules.update(previous)


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
