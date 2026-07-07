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
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5DecoderLayer,
    Qwen3_5TextConfig,
)

from amct_pytorch.common.models import MODEL_REGISTRY
from amct_pytorch.common.models.llm.common.base import BaseModel, PtqUnit
from amct_pytorch.common.models.llm.common.quant_apply import (
    apply_quant_to_attn,
    apply_quant_to_moe_mlp,
)
from amct_pytorch.common.models.llm.qwen.qwen3_5.quant_module import (
    QuantQwen35Attn, QuantQwen35LinearAttn, QuantQwen35MLP,
)


# for reference, qwen3.5-4b ppl is 9.5375
@MODEL_REGISTRY.register(
    name="qwen3_5",
    task="llm",
    family="qwen",
    description="Qwen3.5 dense model adapter",
)
class Qwen3_5(BaseModel):
    def __init__(self, args):
        super().__init__(args)
        self.textconfig = Qwen3_5TextConfig
        self._update_config()
        self.num_layers = self.config.num_hidden_layers
        self.cls = Qwen3_5DecoderLayer
        self.model = self.empty_weights_model()
        self.parse_quant_mode()

    def parse_quant_mode(self):
        if "moe" in self.quant_target:
            raise ValueError("Qwen3.5 < 7B is a dense model and does not support quant_target='moe'.")

    def float_model(self):
        return super().float_model()

    def empty_weights_model(self):
        return super().empty_weights_model()

    def get_layer_weight_prefix(self, layer_idx: int) -> str:
        return f"model.language_model.layers.{layer_idx}."

    def load_layer_weight(self, prefix):
        return super().load_layer_weight(prefix)

    def load_embed_state_dict(self):
        return super().load_embed_state_dict()

    def block(self, layer_idx):
        return super().block(layer_idx)

    def do_embedding_forward(self, samples, dtype=torch.bfloat16, hook_name=None):
        return super().do_embedding_forward(samples, dtype=dtype, hook_name=hook_name)

    def do_block_forward(self, layer_idx, samples, hook_name=None, use_quant_block=False, enable_quant=False):
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
        if "mlp" in self.quant_target:
            self.apply_quant_moe_mlp(decoder_layer)
        return decoder_layer

    def apply_quant_attn(self, decoder_layer):
        layer_type = decoder_layer.layer_type
        if layer_type == "linear_attention":
            attn_cls = QuantQwen35LinearAttn
            quant_attn = getattr(decoder_layer, "linear_attn", None)
            quant_attn.config = self.config
        else:
            attn_cls = QuantQwen35Attn
        return apply_quant_to_attn(self.args, decoder_layer, attn_cls)

    def apply_quant_moe_mlp(self, decoder_layer):
        return apply_quant_to_moe_mlp(
            self.args,
            decoder_layer,
            cls=QuantQwen35MLP,
        )

    def iter_ptq_units(self, layer_idx, block):
        yield from super().iter_ptq_units(layer_idx, block)

    def iter_deploy_bindings(self, layer_idx, block):
        yield from super().iter_deploy_bindings(layer_idx, block)

    def load_unit_inputs(self, data_dir, unit: PtqUnit):
        return super().load_unit_inputs(data_dir, unit)

    def get_block_attention_mask(self, block):
        return None

    def _update_config(self):
        self.config = self.textconfig(**self.config.text_config.to_dict())

    def _embed_base_prefix(self) -> str:
        return "model.language_model."