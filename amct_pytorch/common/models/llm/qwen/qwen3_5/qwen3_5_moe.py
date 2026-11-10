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

from __future__ import annotations

import re

import torch
from compressed_tensors.utils.safetensors_load import get_weight_mappings
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeDecoderLayer,
    Qwen3_5MoeTextConfig,
)

from amct_pytorch.common.models import MODEL_REGISTRY
from amct_pytorch.common.models.llm.common.base import PtqUnit
from amct_pytorch.common.models.llm.common.deployment.schemes import (
    PLATFORM_QUANT_DTYPES,
)
from amct_pytorch.common.models.llm.common.moe_unpack import find_moe_module
from amct_pytorch.common.models.llm.common.quant_apply import build_no_algo_args
from amct_pytorch.common.models.llm.qwen.moe_common import (
    QuantGatedExperts,
    is_packed_experts,
)
from amct_pytorch.common.models.llm.qwen.qwen3_5.qwen3_5 import QuantQwen35MLP, Qwen3_5
from amct_pytorch.quantization.modules.quant_linear import QuantLinear


@MODEL_REGISTRY.register(
    name="qwen3_5_moe",
    task="llm",
    family="qwen",
    description="Qwen3.5 moe model adapter",
)
class Qwen3_5Moe(Qwen3_5):
    _LAYER_RE = re.compile(r"model\.language_model\.layers\.(\d+)\.(.+)")
    _LINEAR_ATTN_PROJECTIONS = frozenset(
        {
            "linear_attn.in_proj_qkv",
            "linear_attn.in_proj_z",
            "linear_attn.in_proj_b",
            "linear_attn.in_proj_a",
            "linear_attn.out_proj",
        }
    )

    def __init__(self, args):
        super().__init__(args)
        self._weight_map = get_weight_mappings(self.model_path)
        self.cls = Qwen3_5MoeDecoderLayer
        self.textconfig = Qwen3_5MoeTextConfig

    def parse_quant_mode(self):
        if "mlp" in self.quant_target:
            raise ValueError(
                "Qwen3.5-moe is a moe model and does not support quant_target='mlp'."
            )

    @staticmethod
    def get_deploy_capabilities():
        return PLATFORM_QUANT_DTYPES

    def _is_full_attention_layer(self, layer_idx: int) -> bool:
        layer_types = getattr(self.config, "layer_types", ())
        return (
            layer_idx < len(layer_types) and layer_types[layer_idx] == "full_attention"
        )

    def _is_linear_attention_layer(self, layer_idx: int) -> bool:
        layer_types = getattr(self.config, "layer_types", ())
        return (
            layer_idx < len(layer_types)
            and layer_types[layer_idx] == "linear_attention"
        )

    def _deploy_role(self, name: str) -> str | None:
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
        if (
            self._is_linear_attention_layer(layer_idx)
            and suffix in self._LINEAR_ATTN_PROJECTIONS
        ):
            return "attn-linear"
        return None

    def get_deploy_candidates(self, module_names: set[str]) -> set[str]:
        return {name for name in module_names if self._deploy_role(name) is not None}

    def validate_deploy_selection(self, selected: set[str]) -> None:
        unsupported = sorted(
            name for name in selected if self._deploy_role(name) is None
        )
        if unsupported:
            raise ValueError(f"Unsupported Deployment modules: {unsupported}")
        for layer_idx in range(int(self.config.num_hidden_layers)):
            prefix = f"model.language_model.layers.{layer_idx}.self_attn."
            qkv = {prefix + projection for projection in ("q_proj", "k_proj", "v_proj")}
            if (
                self._is_full_attention_layer(layer_idx)
                and qkv & selected
                and not qkv <= selected
            ):
                raise ValueError(
                    f"{type(self).__name__} full-attention QKV must be selected together at layer {layer_idx}"
                )
            expert_prefix = f"model.language_model.layers.{layer_idx}.mlp.experts."
            experts = {
                f"{expert_prefix}{expert_idx}.{projection}"
                for expert_idx in range(int(self.config.num_experts))
                for projection in ("gate_proj", "up_proj", "down_proj")
            }
            if selected & experts and not experts <= selected:
                raise ValueError(
                    f"{type(self).__name__} requires all experts and projections at layer {layer_idx}"
                )

    def float_model(self):
        return super().float_model()

    def empty_weights_model(self):
        return super().empty_weights_model()

    def get_layer_weight_prefix(self, layer_idx: int) -> str:
        return super().get_layer_weight_prefix(layer_idx)

    def load_layer_weight(self, prefix):
        return super().load_layer_weight(prefix)

    def load_embed_state_dict(self):
        return super().load_embed_state_dict()

    def block(self, layer_idx):
        return super().block(layer_idx)

    def do_embedding_forward(self, samples, dtype=torch.bfloat16, hook_name=None):
        return super().do_embedding_forward(samples, dtype=dtype, hook_name=hook_name)

    def do_block_forward(
        self,
        layer_idx,
        samples,
        hook_name=None,
        use_quant_block=False,
        enable_quant=False,
    ):
        return super().do_block_forward(
            layer_idx,
            samples,
            hook_name=hook_name,
            use_quant_block=use_quant_block,
            enable_quant=enable_quant,
        )

    def do_head_forward(self, inps):
        return super().do_head_forward(inps)

    def build_quant_block(self, layer_idx):
        decoder_layer = self.block(layer_idx)
        if "attn-linear" in self.quant_target or "attn-cache" in self.quant_target:
            self.apply_quant_attn(decoder_layer)

        if "moe" in self.quant_target:
            quant_moe = find_moe_module(decoder_layer)
            if quant_moe is not None:
                experts = getattr(quant_moe, "experts", None)
                if experts is not None and is_packed_experts(experts):
                    quant_moe.experts = QuantGatedExperts(
                        self.args, experts, group="moe.routed"
                    )
                    shared_expert = getattr(quant_moe, "shared_expert", None)
                    if shared_expert is not None and not isinstance(
                        shared_expert, QuantQwen35MLP
                    ):
                        shared_expert_args = build_no_algo_args(self.args)
                        quant_moe.shared_expert = QuantQwen35MLP(
                            shared_expert_args, shared_expert, group="moe.shared"
                        )

        return decoder_layer

    def iter_ptq_units(self, layer_idx, block, *, for_load=False):
        yield from super().iter_ptq_units(layer_idx, block, for_load=for_load)

    def iter_deploy_bindings(self, layer_idx, block):
        weight_prefix = self.get_layer_weight_prefix(layer_idx)
        for name, module in block.named_modules():
            if not isinstance(module, QuantLinear):
                continue

            if name.startswith("mlp.experts.expert_modules."):
                parts = name.split(".")
                if len(parts) != 5:
                    raise ValueError(
                        f"Unexpected Qwen3.5 MoE expert module name: {name}"
                    )
                _, _, _, expert_idx, proj_name = parts
                yield (
                    f"{weight_prefix}mlp.experts.{expert_idx}.{proj_name}.weight",
                    module,
                )
                continue

            yield f"{weight_prefix}{name}.weight", module

    def load_unit_inputs(self, data_dir, unit: PtqUnit):
        return super().load_unit_inputs(data_dir, unit)
