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

"""Ascend deployment payload adaptation without quantization or shard I/O."""

import torch


def adapt_ascend_payload(
    weight_name: str, payload: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    """Adapt symmetric INT8 payload to the W8A8_DYNAMIC checkpoint contract."""
    if not weight_name.endswith(".weight"):
        raise ValueError(f"Ascend payload requires a .weight key: {weight_name}")
    prefix = weight_name.removesuffix(".weight")
    scale_name = prefix + ".weight_scale"
    if set(payload) != {weight_name, scale_name}:
        raise ValueError(
            f"Ascend payload for {weight_name} must contain only weight and weight_scale"
        )
    weight = payload[weight_name]
    scale = payload[scale_name]
    if weight.dtype != torch.int8 or weight.ndim != 2 or weight.numel() == 0:
        raise ValueError(
            f"Ascend payload {weight_name} requires a nonempty 2-D INT8 weight"
        )
    if not scale.is_floating_point() or scale.shape != (weight.shape[0], 1):
        raise ValueError(
            f"Ascend payload {weight_name} requires floating scale shape [out, 1]"
        )
    scale = scale.to(torch.bfloat16)
    if not torch.isfinite(scale).all() or not (scale > 0).all():
        raise ValueError(
            f"Ascend scale for {weight_name} must remain finite and positive in BF16"
        )
    result = dict(payload)
    result[scale_name] = scale
    result[prefix + ".weight_offset"] = torch.zeros_like(scale)
    return result


def make_ascend_description(
    output_tensor_names: set[str], selected_modules: frozenset[str]
) -> dict:
    """Describe exactly the tensors written by the shared shard exporter."""
    dynamic_names = {
        module + suffix
        for module in selected_modules
        for suffix in (".weight", ".weight_scale", ".weight_offset")
    }
    missing = dynamic_names - output_tensor_names
    if missing:
        raise ValueError(
            f"Ascend output is missing quantized tensors: {sorted(missing)}"
        )
    description = {
        "version": "1.0.0",
        "model_quant_type": "W8A8_DYNAMIC",
        "metadata": {},
        "group_size": 0,
        "is_rot_used": False,
    }
    description.update(
        {
            name: "W8A8_DYNAMIC" if name in dynamic_names else "FLOAT"
            for name in sorted(output_tensor_names)
        }
    )
    return description
