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

import os
import json
from pathlib import Path
import torch
from loguru import logger
from safetensors import safe_open
from tqdm import tqdm
from amct_pytorch.common.models import MODEL_REGISTRY
from amct_pytorch.common.models.llm.common.base import BaseModel
from amct_pytorch.common.models.llm.common.capture import Catcher
from amct_pytorch.common.models.llm.common.quant_apply import apply_quant_to_attn, apply_quant_to_moe_mlp
from amct_pytorch.common.models.llm.deepseek.deepseek_v3_2.modeling.modeling_deepseek_v3_2 import Block
from amct_pytorch.common.models.llm.deepseek.deepseek_v3_2.quant_module import (
    QuantDeepseekV3MLP, QuantDeepseekV3Attention)
from amct_pytorch.common.datasets.ptq_io import save_ptq_kwargs



@MODEL_REGISTRY.register(
    name="deepseek_v3_2",
    task="llm",
    family="deepseek",
    description="DeepSeek V3.2 model adapter",
)
class DeepseekV32(BaseModel):
    def __init__(self, args):
        super().__init__(args)
        self.quant_target = args.quant_target
        self.init_cls()
        self.cls = Block
        self.model = self.empty_weights_model()
        self.num_layers = self.config.num_hidden_layers

    def float_model(self):
        return super().float_model()

    def empty_weights_model(self):
        return super().empty_weights_model()

    def get_layer_weight_prefix(self, layer_idx: int) -> str:
        return f"model.layers.{layer_idx}."

    def get_embed_load_specs(self):
        tie = bool(getattr(self.config, "tie_word_embeddings", False))
        lm_head_prefix = f"{self.base_prefix}embed_tokens." if tie else "lm_head."
        return [
            (self.model.embed, f"{self.base_prefix}embed_tokens."),
            (self.model.norm, f"{self.base_prefix}norm."),
            (self.model.head, lm_head_prefix),
        ]

    def load_layer_weight(self, prefix):
        return super().load_layer_weight(prefix)

    def load_embed_state_dict(self):
        return super().load_embed_state_dict()

    def block(self, layer_idx):
        decoder_layer = self.cls(layer_idx, self.config)
        state_dict = self.load_layer_weight(self.get_layer_weight_prefix(layer_idx))
        decoder_layer.load_state_dict(state_dict, strict=True)
        decoder_layer.eval().bfloat16()
        return decoder_layer

    def do_embedding_forward(self, samples, dtype=torch.bfloat16, hook_name=None):
        outs = []
        self.load_embed_state_dict()
        layers = self.model.layers
        layers[0] = self.block(0)
        layers[0] = layers[0].bfloat16()
        layers[0] = Catcher(layers[0], outs)
        with torch.no_grad():
            for bs, inputs in tqdm(enumerate(samples), total=len(samples), desc=f"Embedding Processing..."):
                try:
                    self.model(inputs)
                except ValueError:
                    pass
        self.position_ids = layers[0].position_ids
        self.position_embeddings = layers[0].position_embeddings
        self.attention_mask = layers[0].attention_mask
        if hook_name is not None:
            save_ptq_kwargs(self.position_ids, self.position_embeddings, self.attention_mask, self.args.data_dir)
        layers[0] = layers[0].module
        return outs

    def do_block_forward(self, layer_idx, samples, hook_name=None, use_quant_block=False, enable_quant=False):
        return super().do_block_forward(
            layer_idx,
            samples,
            hook_name=hook_name,
            use_quant_block=use_quant_block,
            enable_quant=enable_quant,
        )

    def do_head_forward(self, inps):
        self.model.norm.to(self.args.device)
        self.model.head.to(self.args.device)
        preds = []
        with torch.no_grad():
            for idx, inp in tqdm(enumerate(inps), total=len(inps), desc='Head Processing...'):
                inp = inp.to(self.args.device)
                out = self.model.norm(inp)
                out = self.model.head(out)[:, :-1, :].contiguous()
                preds.append(out.to('cpu'))
        return preds

    def build_quant_block(self, layer_idx):
        decoder_layer = self.block(layer_idx)

        if "attn-linear" in self.quant_target or "attn-cache" in self.quant_target:
            self.apply_quant_attn(decoder_layer)
        if "moe" in self.quant_target:
            self.apply_quant_moe_mlp(decoder_layer)
        if "mlp" in self.quant_target:
            raise ValueError("Deepseek V32 is a moe model and does not support quant_target='mlp'.")

        return decoder_layer

    def apply_quant_attn(self, decoder_layer):
        return apply_quant_to_attn(self.args, decoder_layer, QuantDeepseekV3Attention)

    def apply_quant_moe_mlp(self, decoder_layer):
        return apply_quant_to_moe_mlp(
            self.args,
            decoder_layer,
            cls=QuantDeepseekV3MLP,
        )

    def iter_ptq_units(self, layer_idx, block):
        yield from super().iter_ptq_units(layer_idx, block)

    def iter_deploy_bindings(self, layer_idx, block):
        yield from super().iter_deploy_bindings(layer_idx, block)

    def load_unit_inputs(self, data_dir, unit):
        return super().load_unit_inputs(data_dir, unit)

    def get_scale_name(self, weight_name):
        scale_prefix = "_scale_inv"
        scale_inv_name = f"{weight_name}_scale_inv"
        return scale_prefix, scale_inv_name