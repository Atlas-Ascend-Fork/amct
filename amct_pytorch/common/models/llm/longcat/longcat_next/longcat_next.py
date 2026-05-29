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
import torch.nn as nn

from accelerate import init_empty_weights
from transformers.dynamic_module_utils import get_class_from_dynamic_module

from amct_pytorch.common.models.llm.common.base import PtqUnit
from amct_pytorch.common.models.llm.longcat.longcat_lite.longcat_lite import LongcatLite
from amct_pytorch.common.models import MODEL_REGISTRY


@MODEL_REGISTRY.register(
    name="longcat_next",
    task="llm",
    family="longcat",
    description="LongCat-Next model adapter",
)
class LongcatNext(LongcatLite):
    def __init__(self, args):
        super().__init__(args)

    def float_model(self):
        return super().float_model()

    def empty_weights_model(self):
        model_cls = get_class_from_dynamic_module(
            "modeling_longcat_ngram.LongcatFlashNgramForCausalLM",
            self.model_path,
        )
        with init_empty_weights():
            model = model_cls(self.config)
            model = model.bfloat16()
            lm_head_vocab_size = getattr(
                self.config,
                "text_vocab_plus_multimodal_special_token_size",
                self.config.vocab_size,
            )
            model.lm_head = nn.Linear(self.config.hidden_size, lm_head_vocab_size, bias=False, dtype=torch.bfloat16)
        return model

    def get_layer_weight_prefix(self, layer_idx: int) -> str:
        return super().get_layer_weight_prefix(layer_idx)

    def get_embed_load_specs(self):
        return super().get_embed_load_specs()

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
        return super().build_quant_block(layer_idx)

    def iter_ptq_units(self, layer_idx, block):
        yield from super().iter_ptq_units(layer_idx, block)

    def iter_deploy_bindings(self, layer_idx, block):
        yield from super().iter_deploy_bindings(layer_idx, block)

    def load_unit_inputs(self, data_dir, unit: PtqUnit):
        return super().load_unit_inputs(data_dir, unit)
