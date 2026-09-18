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
