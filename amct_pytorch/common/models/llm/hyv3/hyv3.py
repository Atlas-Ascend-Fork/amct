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
from amct_pytorch.common.models.llm.common.deploy_export import generate_quant_group
from amct_pytorch.common.models.llm.common.quant_apply import apply_quant_to_attn
from amct_pytorch.quantization.bit_policy import ensure_bit_policy
from amct_pytorch.quantization.modules.quant_linear import QuantLinear
from amct_pytorch.common.models.llm.hyv3.quant_module import (
    QuantHYV3Attn,
    QuantHYV3MLP,
    QuantHYV3MoE,
)
from amct_pytorch.common.models.llm.qwen.moe_common import pack_gated_expert_weights


def _min_supported_bit(bits):
    resolved_bits = {int(bit) for bit in bits}
    unsupported_bits = resolved_bits - {4, 8, 16}
    if unsupported_bits:
        raise ValueError(f"HY3 tensorwise deploy only supports 4-bit, 8-bit, or 16-bit config bits, "
                         f"got {sorted(resolved_bits)}.")
    quant_bits = [bit for bit in resolved_bits if bit < 16]
    return min(quant_bits) if quant_bits else 8


def _add_tensorwise_layer(quant_layers, weight_map, layer_name, bit=8):
    if weight_map is not None and f"{layer_name}.weight" not in weight_map:
        return
    quant_layers[layer_name] = bit


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
    ATTN_LINEAR_NAMES = ("q_proj", "k_proj", "v_proj", "o_proj")
    MLP_LINEAR_NAMES = ("gate_proj", "down_proj", "up_proj")

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
        yield from super().iter_ptq_units(layer_idx, block)

    def iter_deploy_bindings(self, layer_idx, block):
        weight_prefix = self.get_layer_weight_prefix(layer_idx)
        for name, module in block.named_modules():
            if not isinstance(module, QuantLinear):
                continue
            if name.startswith("mlp.experts.expert_modules."):
                parts = name.split(".")
                if len(parts) != 5:
                    raise ValueError(f"Unexpected HyV3 MoE expert module name: {name}")
                _, _, _, expert_idx, proj_name = parts
                yield f"{weight_prefix}mlp.experts.{expert_idx}.{proj_name}.weight", module
                continue
            yield f"{weight_prefix}{name}.weight", module

    def get_scale_name(self, weight_name):
        scale_prefix = "_scale"
        scale_name = f"{weight_name}_scale"
        return scale_prefix, scale_name

    def generate_tensorwise_quant_layers(self, config: dict, weight_map=None):
        self._tensorwise_weight_map = dict(weight_map or {})
        bit_policy = ensure_bit_policy(self.args)
        quant_targets = set(getattr(self.args, "quant_target", []) or [])
        enable_attn_linear = "attn-linear" in quant_targets
        enable_moe = "moe" in quant_targets
        enable_mlp = "mlp" in quant_targets
        num_hidden_layers = int(config["num_hidden_layers"])
        num_nextn_predict_layers = int(config.get("num_nextn_predict_layers", 0))
        first_k_dense_replace = int(config.get("first_k_dense_replace", 0))
        num_layers = num_hidden_layers + num_nextn_predict_layers
        num_experts = int(config.get("num_experts", config.get("n_routed_experts", 0)))
        quant_layers = {}

        for layer_idx in range(num_layers):
            if enable_moe and layer_idx >= first_k_dense_replace:
                self._add_tensorwise_moe_layers(quant_layers, weight_map, bit_policy, layer_idx, num_experts)
            elif enable_mlp and layer_idx < first_k_dense_replace:
                self._add_tensorwise_mlp_layers(quant_layers, weight_map, bit_policy, layer_idx)

            if enable_attn_linear:
                self._add_tensorwise_attn_layers(quant_layers, weight_map, bit_policy, layer_idx)

        return quant_layers

    def generate_tensorwise_quant_config(self, config: dict, quant_layers=None):
        bit_policy = ensure_bit_policy(self.args)
        quant_type = self._get_tensorwise_quant_type()
        attn_a_bit, attn_w_bit = self._get_tensorwise_attn_bits(bit_policy)
        moe_a_bit, moe_w_bit = self._get_tensorwise_moe_bits(bit_policy)
        ignored_layers = self._get_tensorwise_ignore_layers(config, quant_layers)
        return {
            "config_groups": {
                "group_0": {
                    "targets": ["Linear"],
                    **generate_quant_group(
                        a_num_bits=attn_a_bit,
                        w_num_bits=attn_w_bit,
                        qtype=quant_type,
                        input_group_size=32,
                        weight_group_size=32,
                        input_observer="minmax",
                        input_strategy="group",
                        weight_strategy="group",
                    ),
                },
                "group_1": {
                    "targets": ["MoEGMM"],
                    **generate_quant_group(
                        a_num_bits=moe_a_bit,
                        w_num_bits=moe_w_bit,
                        qtype=quant_type,
                        input_group_size=32,
                        weight_group_size=32,
                        input_observer="minmax",
                        input_strategy="group",
                        weight_strategy="group",
                    ),
                },
            },
            "format": "int-quantized" if quant_type == "int" else "float-quantized",
            "global_compression_ratio": "null",
            "ignore": ignored_layers,
            "kv_cache_scheme": {
                "num_bits": 8,
                "type": quant_type,
                "strategy": "token",
                "group_size": -1,
                "dynamic": "true",
                "symmetric": "true",
            },
            "quant_method": "compressed-tensors",
            "quantization_status": "compressed",
            "weight_block_size": [1, 32],
        }

    def _get_tensorwise_quant_type(self):
        quant_dtype = str(getattr(self.args, "quant_dtype", "") or "")
        if quant_dtype.startswith("int"):
            return "int"
        return "float"

    def _get_tensorwise_attn_bits(self, bit_policy):
        layer_bits = [bit_policy["attn-linear"][name] for name in self.ATTN_LINEAR_NAMES]
        return (
            _min_supported_bit(bit.a for bit in layer_bits),
            _min_supported_bit(bit.w for bit in layer_bits),
        )

    def _get_tensorwise_moe_bits(self, bit_policy):
        layer_bits = []
        for group in ["moe.routed", "moe.shared"]:
            layer_bits.extend(bit_policy[group][name] for name in self.MLP_LINEAR_NAMES)
        return (
            _min_supported_bit(bit.a for bit in layer_bits),
            _min_supported_bit(bit.w for bit in layer_bits),
        )

    def _get_tensorwise_ignore_layers(self, config, quant_layers=None):
        weight_map = getattr(self, "_tensorwise_weight_map", {}) or {}
        quant_layer_names = set(quant_layers or {})
        ignored_layers = []
        for weight_name in weight_map:
            if not weight_name.endswith(".weight"):
                continue
            layer_name = weight_name.removesuffix(".weight")
            if layer_name in quant_layer_names:
                continue
            if layer_name not in ignored_layers:
                ignored_layers.append(layer_name)
        if ignored_layers:
            return ignored_layers

        num_hidden_layers = int(config["num_hidden_layers"])
        return ["lm_head", f"model.layers.{num_hidden_layers}"]

    def _add_tensorwise_moe_layers(self, quant_layers, weight_map, bit_policy, layer_idx, num_experts):
        for expert_idx in range(num_experts):
            for linear_name in self.MLP_LINEAR_NAMES:
                bit = bit_policy["moe.routed"][linear_name].w
                layer_name = f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.{linear_name}"
                _add_tensorwise_layer(quant_layers, weight_map, layer_name, bit=bit)
        for linear_name in self.MLP_LINEAR_NAMES:
            bit = bit_policy["moe.shared"][linear_name].w
            layer_name = f"model.layers.{layer_idx}.mlp.shared_mlp.{linear_name}"
            _add_tensorwise_layer(quant_layers, weight_map, layer_name, bit=bit)

    def _add_tensorwise_mlp_layers(self, quant_layers, weight_map, bit_policy, layer_idx):
        for linear_name in self.MLP_LINEAR_NAMES:
            bit = bit_policy["mlp"][linear_name].w
            layer_name = f"model.layers.{layer_idx}.mlp.{linear_name}"
            _add_tensorwise_layer(quant_layers, weight_map, layer_name, bit=bit)

    def _add_tensorwise_attn_layers(self, quant_layers, weight_map, bit_policy, layer_idx):
        for linear_name in self.ATTN_LINEAR_NAMES:
            bit = bit_policy["attn-linear"][linear_name].w
            layer_name = f"model.layers.{layer_idx}.self_attn.{linear_name}"
            _add_tensorwise_layer(quant_layers, weight_map, layer_name, bit=bit)
