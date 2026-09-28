# coding=utf-8
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

import importlib

import pytest
import torch


def adapt(weight_name, payload):
    module = importlib.import_module(
        "amct_pytorch.common.models.llm.common.deploy_export"
    )
    return module.adapt_ascend_payload(weight_name, payload)


def test_dynamic_payload_preserves_weight_and_original_scale():
    weight = torch.tensor([[64, -127], [0, 0]], dtype=torch.int8)
    scale = torch.tensor([[2.0 / 127], [1e-9]], dtype=torch.float32)
    payload = {"x.weight": weight, "x.weight_scale": scale}
    result = adapt("x.weight", payload)
    assert set(result) == {"x.weight", "x.weight_scale", "x.weight_offset"}
    assert result["x.weight"] is weight
    assert result["x.weight_scale"].dtype == torch.bfloat16
    assert result["x.weight_scale"].shape == (2, 1)
    assert torch.equal(result["x.weight_scale"], scale.to(torch.bfloat16))
    assert result["x.weight_offset"].dtype == torch.bfloat16
    assert torch.equal(result["x.weight_offset"], torch.zeros(2, 1))
    assert payload["x.weight_scale"] is scale
    assert scale.dtype == torch.float32
    assert "x.weight_offset" not in payload


@pytest.mark.parametrize(
    "weight,scale",
    [
        (torch.ones(2, 3), torch.ones(2, 1)),
        (torch.ones(2, dtype=torch.int8), torch.ones(2, 1)),
        (torch.ones(2, 3, dtype=torch.int8), torch.ones(2)),
        (torch.ones(2, 3, dtype=torch.int8), torch.ones(3, 1)),
        (torch.ones(2, 3, dtype=torch.int8), torch.ones(2, 1, dtype=torch.int32)),
    ],
)
def test_rejects_invalid_payload_dtype_or_shape(weight, scale):
    with pytest.raises(ValueError, match="x.weight"):
        adapt("x.weight", {"x.weight": weight, "x.weight_scale": scale})


@pytest.mark.parametrize(
    "value", [0.0, -1.0, float("nan"), float("inf"), 1e-45, 3.4e38]
)
def test_rejects_scale_invalid_after_bf16_conversion(value):
    with pytest.raises(ValueError, match="scale"):
        adapt(
            "x.weight",
            {
                "x.weight": torch.ones(2, 3, dtype=torch.int8),
                "x.weight_scale": torch.full((2, 1), value),
            },
        )


@pytest.mark.parametrize("extra", [None, "weight_bias", "input_scale", "weight_offset"])
def test_rejects_missing_scale_or_unexpected_payload(extra):
    payload = {"x.weight": torch.ones(2, 3, dtype=torch.int8)}
    if extra:
        payload.update(
            {"x.weight_scale": torch.ones(2, 1), "x." + extra: torch.ones(2, 1)}
        )
    with pytest.raises(ValueError, match="payload"):
        adapt("x.weight", payload)


def test_description_tracks_actual_tensors():
    module = importlib.import_module(
        "amct_pytorch.common.models.llm.common.deploy_export"
    )
    names = {
        "x.weight",
        "x.weight_scale",
        "x.weight_offset",
        "x.bias",
        "model.layers.2.eh_proj.weight",
        "model.norm.weight",
    }
    desc = module.make_ascend_description(names, frozenset({"x"}))
    assert desc == {
        "version": "1.0.0",
        "model_quant_type": "W8A8_DYNAMIC",
        "metadata": {},
        "group_size": 0,
        "is_rot_used": False,
        "x.weight": "W8A8_DYNAMIC",
        "x.weight_scale": "W8A8_DYNAMIC",
        "x.weight_offset": "W8A8_DYNAMIC",
        "x.bias": "FLOAT",
        "model.layers.2.eh_proj.weight": "FLOAT",
        "model.norm.weight": "FLOAT",
    }


@pytest.mark.parametrize("missing", ["weight", "weight_scale", "weight_offset"])
def test_description_rejects_incomplete_selected_payload(missing):
    module = importlib.import_module(
        "amct_pytorch.common.models.llm.common.deploy_export"
    )
    names = {"x.weight", "x.weight_scale", "x.weight_offset"} - {"x." + missing}
    with pytest.raises(ValueError, match="missing"):
        module.make_ascend_description(names, frozenset({"x"}))
