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

import torch
from accelerate import init_empty_weights

from amct_pytorch.common.models import MODEL_REGISTRY
from amct_pytorch.common.models.llm.common.base import BaseModel
from amct_pytorch.common.models.llm.common.quant_apply import apply_quant_to_moe_mlp
from amct_pytorch.common.models.llm.kimi.kimi_vl.modeling.configuration_kimi_vl import (
    DeepseekV3Config,
)
from amct_pytorch.common.models.llm.kimi.kimi_vl.modeling.modeling_kimi_vl import (
    DeepseekV3DecoderLayer,
    DeepseekV3ForCausalLM,
)
from amct_pytorch.common.models.llm.kimi.kimi_vl.quant_module import QuantKimiVLMLP


@MODEL_REGISTRY.register(
    name="kimi_vl",
    task="llm",
    family="kimi",
    description="Kimi-VL A3B multimodal model adapter (language side)",
)
class KimiVL(BaseModel):
    """Adapter for the language side of Kimi-VL-A3B.

    Only the text backbone is instantiated, from the vendored modeling in
    ``modeling/`` — the official remote code targets transformers 4.x and does
    not import on the version AMCT pins. MoonViT and the multimodal projector
    are never built: quantizing them is out of scope, and ``deploy`` writes
    through every tensor it did not replace, so their weights survive the
    export untouched.

    The checkpoint still nests the backbone under the multimodal wrapper, so
    module paths are the plain HuggingFace ones while every checkpoint prefix
    carries an extra ``language_model.`` segment.
    """

    def __init__(self, args):
        super().__init__(args)
        self.vl_config = self.config
        self.config = self._text_config()
        self.num_layers = self.config.num_hidden_layers
        self.cls = DeepseekV3DecoderLayer
        self.model = self.empty_weights_model()
        self.parse_quant_mode()

    def _text_config(self):
        text_config = DeepseekV3Config(**self.vl_config.text_config.to_dict())
        # The vendored backbone ships the eager attention path only.
        text_config._attn_implementation = "eager"
        return text_config

    def parse_quant_mode(self):
        if "attn-linear" in self.quant_target or "attn-cache" in self.quant_target:
            raise ValueError(
                "Kimi-VL adapter does not support quant_target='attn-linear' or "
                "'attn-cache' yet. The MLA attention of the language backbone "
                "(q_lora_rank=None) needs its own quant module. Use "
                "quant_target='moe' and/or 'mlp'."
            )

    def empty_weights_model(self):
        with init_empty_weights():
            model = DeepseekV3ForCausalLM(self.config)
        return model.to(torch.bfloat16)

    def _embed_base_prefix(self) -> str:
        return "language_model.model."

    def get_layer_weight_prefix(self, layer_idx: int) -> str:
        return f"language_model.model.layers.{layer_idx}."

    def get_embed_load_specs(self):
        # Same module paths as the base class; only the checkpoint-side
        # `lm_head` prefix differs, because it sits under the VL wrapper.
        base = self._embed_base_prefix()
        tie = bool(getattr(self.config, "tie_word_embeddings", False))
        lm_head_prefix = f"{base}embed_tokens." if tie else "language_model.lm_head."
        return [
            (self.model.model.embed_tokens, f"{base}embed_tokens."),
            (self.model.model.norm, f"{base}norm."),
            (self.model.lm_head, lm_head_prefix),
        ]

    def build_quant_block(self, layer_idx):
        decoder_layer = self.block(layer_idx)
        if "moe" in self.quant_target or "mlp" in self.quant_target:
            self.apply_quant_moe_mlp(decoder_layer)
        return decoder_layer

    def apply_quant_moe_mlp(self, decoder_layer):
        # Layers below first_k_dense_replace carry a plain DeepseekV3MLP and are
        # wrapped under the `mlp` bit-policy group; the rest carry routed
        # experts plus shared experts (`moe.routed` / `moe.shared`).
        return apply_quant_to_moe_mlp(
            self.args,
            decoder_layer,
            cls=QuantKimiVLMLP,
        )
