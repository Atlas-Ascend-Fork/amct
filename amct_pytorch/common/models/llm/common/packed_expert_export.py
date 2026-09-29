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

"""Checkpoint-side expansion for packed gated MoE experts."""

import re
from dataclasses import dataclass

import torch

_PACKED_EXPERT_RE = re.compile(
    r"^(?P<prefix>.+\.experts)\.(?P<projection>gate_up_proj|down_proj)$"
)
_NUM_EXPERT_FIELDS = ("num_experts", "n_routed_experts", "num_local_experts")
_INTERMEDIATE_FIELDS = ("moe_intermediate_size", "intermediate_size")


def _config_value(config, fields):
    for candidate in (config, getattr(config, "text_config", None)):
        if candidate is None:
            continue
        for field in fields:
            value = getattr(candidate, field, None)
            if value is not None:
                return value
    return None


@dataclass(frozen=True)
class PackedGatedExpertLayout:
    """Describe the standard [experts, rows, columns] gated-MoE layout."""

    num_experts: int
    moe_intermediate_size: int | None = None

    @staticmethod
    def _match(name: str):
        return _PACKED_EXPERT_RE.fullmatch(name)

    @staticmethod
    def _expert_name(prefix: str, expert_idx: int, projection: str) -> str:
        return f"{prefix}.{expert_idx}.{projection}.weight"

    def _validate_tensor(self, name: str, tensor: torch.Tensor, projection: str):
        if tensor.ndim != 3 or tensor.shape[0] != self.num_experts:
            raise ValueError(
                f"Packed expert tensor {name} must have shape "
                f"[{self.num_experts}, ...], got {tuple(tensor.shape)}"
            )
        if projection == "gate_up_proj":
            if tensor.shape[1] % 2:
                raise ValueError(
                    f"Packed gate_up_proj {name} requires an even gate/up dimension, "
                    f"got {tensor.shape[1]}"
                )
            if (
                self.moe_intermediate_size is not None
                and tensor.shape[1] != 2 * self.moe_intermediate_size
            ):
                raise ValueError(
                    f"Packed gate_up_proj {name} has dimension {tensor.shape[1]}, "
                    f"expected {2 * self.moe_intermediate_size}"
                )
        elif (
            self.moe_intermediate_size is not None
            and tensor.shape[2] != self.moe_intermediate_size
        ):
            raise ValueError(
                f"Packed down_proj {name} has input dimension {tensor.shape[2]}, "
                f"expected {self.moe_intermediate_size}"
            )

    def expand_weight_map(self, weight_map: dict[str, str]) -> dict[str, str]:
        """Expand packed index entries into vLLM per-expert weight names."""
        expanded = {}
        for name, filename in weight_map.items():
            match = self._match(name)
            if match is None:
                expanded[name] = filename
                continue
            projections = (
                ("gate_proj", "up_proj")
                if match["projection"] == "gate_up_proj"
                else ("down_proj",)
            )
            for expert_idx in range(self.num_experts):
                for projection in projections:
                    expanded[
                        self._expert_name(match["prefix"], expert_idx, projection)
                    ] = filename
        return expanded

    def expand_tensors(
        self,
        state_dict: dict[str, torch.Tensor],
        selected_weight_keys: set[str] | frozenset[str] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Expand packed tensors and clone slices that will remain floating point."""
        expanded = {}
        for name, tensor in state_dict.items():
            match = self._match(name)
            if match is None:
                expanded[name] = tensor
                continue
            projection = match["projection"]
            self._validate_tensor(name, tensor, projection)
            if projection == "gate_up_proj":
                midpoint = tensor.shape[1] // 2
                slices = {
                    "gate_proj": tensor[:, :midpoint],
                    "up_proj": tensor[:, midpoint:],
                }
            else:
                slices = {"down_proj": tensor}
            for expert_idx in range(self.num_experts):
                for output_projection, packed_tensor in slices.items():
                    output_name = self._expert_name(
                        match["prefix"], expert_idx, output_projection
                    )
                    value = packed_tensor[expert_idx]
                    if (
                        selected_weight_keys is None
                        or output_name not in selected_weight_keys
                    ):
                        value = value.clone()
                    expanded[output_name] = value
        return expanded


def detect_packed_gated_expert_layout(
    weight_map: dict[str, str], config
) -> PackedGatedExpertLayout | None:
    """Detect the standard packed gated-expert checkpoint contract."""
    projections_by_prefix = {}
    for name in weight_map:
        match = _PACKED_EXPERT_RE.fullmatch(name)
        if match is not None:
            projections_by_prefix.setdefault(match["prefix"], set()).add(
                match["projection"]
            )
    if not projections_by_prefix:
        return None

    incomplete = sorted(
        prefix
        for prefix, projections in projections_by_prefix.items()
        if projections != {"gate_up_proj", "down_proj"}
    )
    if incomplete:
        raise ValueError(
            "Packed gated-expert entries must contain both gate_up_proj and "
            f"down_proj: {incomplete}"
        )

    num_experts = _config_value(config, _NUM_EXPERT_FIELDS)
    if num_experts is None or int(num_experts) <= 0:
        raise ValueError(
            "Packed gated-expert checkpoint requires a positive num_experts "
            "or n_routed_experts config field"
        )
    intermediate = _config_value(config, _INTERMEDIATE_FIELDS)
    return PackedGatedExpertLayout(
        num_experts=int(num_experts),
        moe_intermediate_size=None if intermediate is None else int(intermediate),
    )
