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
import re
from compressed_tensors.utils.safetensors_load import get_weight_mappings
from loguru import logger
from tqdm import tqdm

from transformers.models.hy_v3.modeling_hy_v3 import (
    HYV3Config,
    HYV3DecoderLayer,
)

from amct_pytorch.common.models.llm.common.base import BaseModel
from amct_pytorch.common.models.llm.common.capture import Catcher
from amct_pytorch.common.models.llm.common.quant_apply import apply_quant_to_attn
from amct_pytorch.common.models import MODEL_REGISTRY
from amct_pytorch.common.datasets.ptq_io import save_ptq_kwargs
from amct_pytorch.quantization.modules.quant_linear import QuantLinear
from amct_pytorch.common.models.llm.hyv3.quant_module import (
    QuantHYV3Attn,
    QuantHYV3MLP,
    QuantHYV3MoE,
)


def pack_hyv3_expert_weights(state_dict, expert_prefix: str = "mlp.experts"):
    gate_proj = {}
    up_proj = {}
    down_proj = {}
    packed_state = {}
    
    pattern = re.compile(
        rf"{re.escape(expert_prefix)}\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$"
    )

    for key, tensor in state_dict.items():
        match = pattern.match(key)
        if match is None:
            packed_state[key] = tensor
            continue
        expert_idx = int(match.group(1))
        proj_name = match.group(2)
        if proj_name == "gate_proj":
            gate_proj[expert_idx] = tensor
        elif proj_name == "up_proj":
            up_proj[expert_idx] = tensor
        else:
            down_proj[expert_idx] = tensor

    if not gate_proj and not up_proj and not down_proj:
        return state_dict

    expert_indices = sorted(set(gate_proj) | set(up_proj) | set(down_proj))
    if set(gate_proj) != set(up_proj) or set(gate_proj) != set(down_proj):
        raise KeyError(f"Inconsistent expert weights while packing {expert_prefix}.")

    packed_state[f"{expert_prefix}.gate_up_proj"] = torch.stack(
        [torch.cat([gate_proj[idx], up_proj[idx]], dim=0) for idx in expert_indices],
        dim=0,
    )
    packed_state[f"{expert_prefix}.down_proj"] = torch.stack(
        [down_proj[idx] for idx in expert_indices],
        dim=0,
    )
    return packed_state


def remap_hyv3_keys(state_dict):
    new_state_dict = {}
    for key, tensor in state_dict.items():
        new_key = key
        
        if "mlp.router.gate.weight" in key:
            new_key = key.replace("mlp.router.gate.weight", "mlp.gate.weight")
        elif "mlp.expert_bias" in key:
            new_key = key.replace("mlp.expert_bias", "mlp.e_score_correction_bias")
        elif "mlp.shared_mlp." in key:
            new_key = key.replace("mlp.shared_mlp.", "mlp.shared_experts.")
        
        new_state_dict[new_key] = tensor
    
    return new_state_dict


@MODEL_REGISTRY.register(
    name="hy_v3",
    task="llm",
    family="hyv3",
    description="Tencent HunYuan V3 (Hy3-preview) model adapter",
    force=True,
)
class HyV3(BaseModel):
    def __init__(self, args):
        super().__init__(args)
        self.quant_target = args.quant_target
        self.textconfig = HYV3Config
        self._weight_map = get_weight_mappings(self.model_path)
        self.config._attn_implementation = "eager"
        self.num_layers = self.config.num_hidden_layers
        self.cls = HYV3DecoderLayer
        self.model = self.empty_weights_model()
        self.parse_quant_mode()

    def parse_quant_mode(self):
        if "mlp" in self.quant_target:
            raise ValueError("HyV3 is a MoE model and does not support quant_target='mlp'. Use 'moe' instead.")

    def float_model(self):
        return super().float_model()

    def empty_weights_model(self):
        return super().empty_weights_model()

    def get_layer_weight_prefix(self, layer_idx: int) -> str:
        return f"model.layers.{layer_idx}."

    def load_layer_weight(self, prefix):
        state_dict = super().load_layer_weight(prefix)
        state_dict = remap_hyv3_keys(state_dict)
        if "mlp.experts.0.gate_proj.weight" in state_dict:
            state_dict = pack_hyv3_expert_weights(state_dict, expert_prefix="mlp.experts")
        return state_dict

    def load_embed_state_dict(self):
        return super().load_embed_state_dict()

    def block(self, layer_idx):
        return super().block(layer_idx)

    def do_embedding_forward(self, samples, dtype=torch.bfloat16, hook_name=None):
        outs = []
        self.load_embed_state_dict()
        layers = self.model.model.layers
        layers[0] = self.block(0)
        layers[0] = layers[0].bfloat16()
        layers[0] = Catcher(layers[0], outs)
        with torch.no_grad():
            for bs, inputs in tqdm(enumerate(samples), total=len(samples), desc="Embedding Processing..."):
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
        return super().do_head_forward(inps)

    def build_quant_block(self, layer_idx):
        decoder_layer = self.block(layer_idx)
        
        if "attn-linear" in self.quant_target or "attn-cache" in self.quant_target:
            apply_quant_to_attn(self.args, decoder_layer, QuantHYV3Attn)
        
        if "moe" in self.quant_target:
            mlp = getattr(decoder_layer, "mlp", None)
            if mlp is not None:
                if hasattr(mlp, "experts"):
                    decoder_layer.mlp = QuantHYV3MoE(self.args, mlp)
                else:
                    decoder_layer.mlp = QuantHYV3MLP(self.args, mlp, group="mlp")
        
        return decoder_layer

    def iter_ptq_units(self, layer_idx, block):
        from amct_pytorch.common.models.llm.common.ptq_units import make_ptq_unit, iter_indexed_units
        
        if "attn-linear" in self.quant_target or "attn-cache" in self.quant_target:
            quant_attn = getattr(block, "self_attn", None)
            yield make_ptq_unit("attn", "self_attn", layer_idx, quant_attn)
            return
        
        mlp = getattr(block, "mlp", None)
        if mlp is None:
            return
        
        if "moe" in self.quant_target:
            if hasattr(mlp, "experts"):
                experts = getattr(mlp, "experts", [])
                if hasattr(experts, "iter_ptq_expert_modules"):
                    expert_list = experts.iter_ptq_expert_modules()
                elif hasattr(experts, "expert_modules"):
                    expert_list = experts.expert_modules
                else:
                    expert_list = list(experts)
                
                yield from iter_indexed_units(
                    kind="moe",
                    name_prefix="expert",
                    layer_idx=layer_idx,
                    items=expert_list,
                    metadata_fn=lambda idx, _: {"expert_idx": idx},
                )
            
            if hasattr(mlp, "shared_experts"):
                shared_experts = mlp.shared_experts
                yield make_ptq_unit("moe.shared", "shared_experts", layer_idx, shared_experts)

    def iter_deploy_bindings(self, layer_idx, block):
        weight_prefix = self.get_layer_weight_prefix(layer_idx)
        for name, module in block.named_modules():
            if not isinstance(module, QuantLinear):
                continue
            yield f"{weight_prefix}{name}.weight", module

    def load_unit_inputs(self, data_dir, unit):
        return super().load_unit_inputs(data_dir, unit)

    def get_scale_name(self, weight_name):
        scale_prefix = "_scale_inv"
        scale_inv_name = f"{weight_name}_scale_inv"
        return scale_prefix, scale_inv_name