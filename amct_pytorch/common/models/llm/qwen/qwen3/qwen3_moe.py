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
from compressed_tensors.utils.safetensors_load import get_weight_mappings
from transformers.models.qwen3_moe.modeling_qwen3_moe import (
    Qwen3MoeConfig,
    Qwen3MoeDecoderLayer,
)

from amct_pytorch.common.models.llm.common.base import BaseModel, PtqUnit
from amct_pytorch.common.models.llm.common.quant_apply import apply_quant_to_attn
from amct_pytorch.common.models.llm.qwen.moe_common import QuantGatedExperts, pack_gated_expert_weights
from amct_pytorch.common.models.llm.qwen.qwen3.quant_module import QuantQwen3Attn, QuantQwen3MLP
from amct_pytorch.quantization.modules.quant_linear import QuantLinear
from amct_pytorch.common.models import MODEL_REGISTRY


@MODEL_REGISTRY.register(
    name='qwen3_moe',
    task='llm',
    family='qwen',
    description='Qwen3 MoE model adapter',
)
class Qwen3Moe(BaseModel):
    def __init__(self, args):
        super().__init__(args)
        self.quant_target = args.quant_target
        self.textconfig = Qwen3MoeConfig
        self._weight_map = get_weight_mappings(self.model_path)
        self.config._attn_implementation = 'eager'
        self.num_layers = self.config.num_hidden_layers
        self.cls = Qwen3MoeDecoderLayer
        self.model = self.empty_weights_model()
        self.parse_quant_mode()

    def parse_quant_mode(self):
        if "mlp" in self.quant_target:
            raise ValueError("Qwen3-MoE is a moe model and does not support quant_target='mlp'.")

    def float_model(self):
        return super().float_model()

    def empty_weights_model(self):
        return super().empty_weights_model()

    def get_layer_weight_prefix(self, layer_idx: int) -> str:
        return f'model.layers.{layer_idx}.'

    def load_layer_weight(self, prefix):
        state_dict = super().load_layer_weight(prefix)
        state_dict = pack_gated_expert_weights(state_dict, expert_prefix='mlp.experts')
        return state_dict

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
            apply_quant_to_attn(self.args, decoder_layer, QuantQwen3Attn)
        if 'moe' in self.quant_target:
            mlp = getattr(decoder_layer, 'mlp', None)
            if mlp is not None and hasattr(mlp, 'experts'):
                mlp.experts = QuantGatedExperts(self.args, mlp.experts)
            elif mlp is not None:
                decoder_layer.mlp = QuantQwen3MLP(self.args, mlp)
        return decoder_layer

    def iter_ptq_units(self, layer_idx, block):
        yield from super().iter_ptq_units(layer_idx, block)

    def iter_deploy_bindings(self, layer_idx, block):
        weight_prefix = self.get_layer_weight_prefix(layer_idx)
        for name, module in block.named_modules():
            if not isinstance(module, QuantLinear):
                continue

            if name.startswith("mlp.experts.expert_modules."):
                parts = name.split(".")
                if len(parts) != 5:
                    raise ValueError(f"Unexpected Qwen3 MoE expert module name: {name}")
                _, _, _, expert_idx, proj_name = parts
                yield f"{weight_prefix}mlp.experts.{expert_idx}.{proj_name}.weight", module
                continue

            yield f"{weight_prefix}{name}.weight", module

    def load_unit_inputs(self, data_dir, unit: PtqUnit):
        return super().load_unit_inputs(data_dir, unit)
