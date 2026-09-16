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

import re

import torch

from amct_pytorch.common.models import MODEL_REGISTRY
from amct_pytorch.common.models.llm.qwen.qwen3_5.qwen3_5_moe import Qwen3_5Moe


@MODEL_REGISTRY.register(
    name="qwen3_6_moe",
    task="llm",
    family="qwen",
    description="Qwen3.6 moe model adapter",
)
class Qwen3_6Moe(Qwen3_5Moe):
    """Qwen3.6 extensions used by Ascend tensorwise deployment."""

    _LAYER_RE = re.compile(r"model\.language_model\.layers\.(\d+)\.(.+)")

    def _is_full_attention_layer(self, layer_idx: int) -> bool:
        layer_types = getattr(self.config, "layer_types", ())
        return (
            layer_idx < len(layer_types) and layer_types[layer_idx] == "full_attention"
        )

    def get_ascend_deploy_mtp_modules(self, module_names: set[str]) -> set[str]:
        return {name for name in module_names if name.startswith("mtp.layers.")}

    @staticmethod
    def _expanded_expert_name(name: str, expert_idx: int, projection: str) -> str:
        return (
            re.sub(
                r"mlp\.experts\.(?:gate_up_proj|down_proj)$",
                f"mlp.experts.{expert_idx}.{projection}",
                name,
            )
            + ".weight"
        )

    def expand_ascend_deploy_weight_map(
        self, weight_map: dict[str, str]
    ) -> dict[str, str]:
        expanded = {}
        num_experts = int(getattr(self.config, "num_experts", 0))
        for name, filename in weight_map.items():
            match = re.search(r"mlp\.experts\.(gate_up_proj|down_proj)$", name)
            if match is None:
                expanded[name] = filename
                continue
            projections = (
                ("gate_proj", "up_proj")
                if match[1] == "gate_up_proj"
                else ("down_proj",)
            )
            for expert_idx in range(num_experts):
                for projection in projections:
                    expanded[
                        self._expanded_expert_name(name, expert_idx, projection)
                    ] = filename
        return expanded

    def expand_ascend_deploy_tensors(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        expanded = {}
        num_experts = int(getattr(self.config, "num_experts", 0))
        for name, tensor in state_dict.items():
            match = re.search(r"mlp\.experts\.(gate_up_proj|down_proj)$", name)
            if match is None:
                expanded[name] = tensor
                continue
            if tensor.ndim != 3 or tensor.shape[0] != num_experts:
                raise ValueError(
                    f"Invalid packed Qwen3.6 expert tensor: {name} {tuple(tensor.shape)}"
                )
            clone_slices = name.startswith("mtp.")
            if match[1] == "gate_up_proj":
                if tensor.shape[1] % 2:
                    raise ValueError(
                        f"Packed gate_up_proj has odd intermediate dimension: {name}"
                    )
                midpoint = tensor.shape[1] // 2
                for expert_idx in range(num_experts):
                    gate = tensor[expert_idx, :midpoint]
                    up = tensor[expert_idx, midpoint:]
                    expanded[
                        self._expanded_expert_name(name, expert_idx, "gate_proj")
                    ] = gate.clone() if clone_slices else gate
                    expanded[
                        self._expanded_expert_name(name, expert_idx, "up_proj")
                    ] = up.clone() if clone_slices else up
            else:
                for expert_idx in range(num_experts):
                    down = tensor[expert_idx]
                    expanded[
                        self._expanded_expert_name(name, expert_idx, "down_proj")
                    ] = down.clone() if clone_slices else down
        return expanded

    def _ascend_deploy_role(self, name: str) -> str | None:
        match = self._LAYER_RE.fullmatch(name)
        if match is None:
            return None
        layer_idx, suffix = int(match[1]), match[2]
        if not 0 <= layer_idx < int(self.config.num_hidden_layers):
            return None
        expert = re.fullmatch(
            r"mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)", suffix
        )
        if expert and int(expert[1]) < int(self.config.num_experts):
            return "moe.routed"
        if self._is_full_attention_layer(layer_idx) and suffix in {
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
        }:
            return "attn-linear"
        return None

    def get_ascend_deploy_candidates(self, module_names: set[str]) -> set[str]:
        return {
            name for name in module_names if self._ascend_deploy_role(name) is not None
        }

    def validate_ascend_deploy_selection(self, selected: set[str]) -> None:
        unsupported = sorted(
            name for name in selected if self._ascend_deploy_role(name) is None
        )
        if unsupported:
            raise ValueError(f"Unsupported Ascend deployment modules: {unsupported}")
        for layer_idx in range(int(self.config.num_hidden_layers)):
            prefix = f"model.language_model.layers.{layer_idx}.self_attn."
            qkv = {prefix + projection for projection in ("q_proj", "k_proj", "v_proj")}
            if (
                self._is_full_attention_layer(layer_idx)
                and qkv & selected
                and not qkv <= selected
            ):
                raise ValueError(
                    f"Qwen3.6 full-attention QKV must be selected together at layer {layer_idx}"
                )
            expert_prefix = f"model.language_model.layers.{layer_idx}.mlp.experts."
            experts = {
                f"{expert_prefix}{expert_idx}.{projection}"
                for expert_idx in range(int(self.config.num_experts))
                for projection in ("gate_proj", "up_proj", "down_proj")
            }
            if selected & experts and not experts <= selected:
                raise ValueError(
                    f"Qwen3.6 requires all experts and projections at layer {layer_idx}"
                )

    def get_ascend_deploy_bits(self, name: str) -> int:
        if self._ascend_deploy_role(name) is None:
            raise ValueError(f"Unsupported Ascend deployment module: {name}")
        return 8

    def get_ascend_deploy_quant_type(self, name: str) -> str:
        self.get_ascend_deploy_bits(name)
        return "W8A8_DYNAMIC"
