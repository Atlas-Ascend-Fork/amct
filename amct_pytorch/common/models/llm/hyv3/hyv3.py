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

from compressed_tensors.utils.safetensors_load import get_weight_mappings

from transformers.models.hy_v3.modeling_hy_v3 import (
    HYV3Config,
    HYV3DecoderLayer,
)

from amct_pytorch.common.models import MODEL_REGISTRY
from amct_pytorch.common.models.llm.common.base import BaseModel
from amct_pytorch.common.models.llm.common.quant_apply import apply_quant_to_attn
from amct_pytorch.quantization.modules.quant_linear import QuantLinear
from amct_pytorch.common.models.llm.hyv3.quant_module import (
    QuantHYV3Attn,
    QuantHYV3MLP,
    QuantHYV3MoE,
)
from amct_pytorch.common.models.llm.qwen.moe_common import pack_gated_expert_weights


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

    def get_layer_weight_prefix(self, layer_idx: int) -> str:
        return f"model.layers.{layer_idx}."

    def load_layer_weight(self, prefix):
        state_dict = super().load_layer_weight(prefix)
        state_dict = remap_hyv3_keys(state_dict)
        if "mlp.experts.0.gate_proj.weight" in state_dict:
            state_dict = pack_gated_expert_weights(state_dict, expert_prefix="mlp.experts")
        return state_dict

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
            
            # shared_experts uses direct quantization (build_no_algo_args), skip PTQ
            # if hasattr(mlp, "shared_experts"):
            #     shared_experts = mlp.shared_experts
            #     yield make_ptq_unit("moe.shared", "shared_experts", layer_idx, shared_experts)

    def iter_deploy_bindings(self, layer_idx, block):
        weight_prefix = self.get_layer_weight_prefix(layer_idx)
        for name, module in block.named_modules():
            if not isinstance(module, QuantLinear):
                continue
            yield f"{weight_prefix}{name}.weight", module

    def get_scale_name(self, weight_name):
        scale_prefix = "_scale_inv"
        scale_inv_name = f"{weight_name}_scale_inv"
        return scale_prefix, scale_inv_name