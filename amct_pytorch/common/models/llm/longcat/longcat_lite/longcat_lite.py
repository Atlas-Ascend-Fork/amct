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
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from transformers.models.longcat_flash.modeling_longcat_flash import LongcatFlashDecoderLayer

try:
    from transformers.models.longcat_flash.modeling_longcat_flash import is_flash_attention_requested
except ImportError:
    def is_flash_attention_requested(config):
        return getattr(config, "_attn_implementation", None) == "flash_attention_2"

from amct_pytorch.common.models.llm.common.base import BaseModel, PtqUnit
from amct_pytorch.common.models.llm.common.ptq_units import iter_indexed_units
from amct_pytorch.quantization.modules.quant_linear import QuantLinear
from amct_pytorch.common.models import MODEL_REGISTRY
from amct_pytorch.common.models.llm.longcat.longcat_lite.quant_module import QuantLongcatMLA, QuantLongcatMLP
from amct_pytorch.common.models.llm.common.moe_unpack import find_moe_module
from amct_pytorch.common.models.llm.longcat.moe_common import QuantLongcatExperts


def _save_tensor(path: str, tensors: list[torch.Tensor]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    stacked = torch.cat([tensor.detach().cpu() for tensor in tensors], dim=0)
    torch.save(stacked, path)


@MODEL_REGISTRY.register(
    name="longcat_lite",
    task="llm",
    family="longcat",
    description="LongCat-Flash-Lite model adapter",
)
class LongcatLite(BaseModel):
    def __init__(self, args):
        super().__init__(args)
        self.quant_target = args.quant_target
        self._set_safe_attn_impl()
        self.model = self.empty_weights_model()
        self.num_layers = self.config.num_layers
        self.cls = LongcatFlashDecoderLayer

    def block(self, layer_idx):
        decoder_layer = self.cls(self.config, layer_idx)
        state_dict = self._load_longcat_layer_weight(layer_idx)
        decoder_layer.load_state_dict(state_dict, strict=True)
        decoder_layer.eval().bfloat16()
        return decoder_layer

    def build_quant_block(self, layer_idx):
        decoder_layer = self.block(layer_idx)
        if "attn-linear" in self.quant_target or "attn-cache" in self.quant_target:
            decoder_layer.self_attn = nn.ModuleList(
                [QuantLongcatMLA(self.args, module) for module in decoder_layer.self_attn]
            )
        if "mlp" in self.quant_target:
            decoder_layer.mlps = nn.ModuleList(
                [QuantLongcatMLP(self.args, module) for module in decoder_layer.mlps]
            )
        if "moe" in self.quant_target:
            quant_moe = find_moe_module(decoder_layer)
            if quant_moe is None:
                raise TypeError("LongCat MoE quantization expects a module with 'experts'.")
            quant_moe.experts = QuantLongcatExperts(
                self.args,
                quant_moe.experts,
                quant_mlp_cls=QuantLongcatMLP,
            )
        return decoder_layer

    def do_block_forward(self, layer_idx, samples, hook_name=None, use_quant_block=False, enable_quant=False):
        return super().do_block_forward(
            layer_idx,
            samples,
            hook_name=hook_name,
            use_quant_block=use_quant_block,
            enable_quant=enable_quant,
        )

    def do_embedding_forward(self, samples, dtype=torch.bfloat16, hook_name=None):
        return super().do_embedding_forward(samples, dtype=dtype, hook_name=hook_name)

    def do_head_forward(self, inps):
        return super().do_head_forward(inps)

    def empty_weights_model(self):
        return super().empty_weights_model()

    def float_model(self):
        return super().float_model()

    def get_embed_load_specs(self):
        specs = super().get_embed_load_specs()
        specs.extend([
            (self.model.model.ngram_embeddings.embedders, "model.ngram_embeddings.embedders."),
            (self.model.model.ngram_embeddings.post_projs, "model.ngram_embeddings.post_projs."),
        ])
        return specs

    def get_layer_weight_prefix(self, layer_idx: int) -> str:
        return f"model.layers.{layer_idx}."

    def iter_deploy_bindings(self, layer_idx, block):
        weight_prefix = self.get_layer_weight_prefix(layer_idx)
        for name, module in block.named_modules():
            if not isinstance(module, QuantLinear):
                continue

            if name.startswith("mlp.experts.expert_modules."):
                parts = name.split(".")
                if len(parts) != 5:
                    raise ValueError(f"Unexpected LongCat expert module name: {name}")
                _, _, _, expert_idx, proj_name = parts
                yield f"{weight_prefix}mlp.experts.{expert_idx}.{proj_name}.weight", module
                continue

            yield f"{weight_prefix}{name}.weight", module

    def iter_ptq_units(self, layer_idx, block):
        if "attn-linear" in self.quant_target or "attn-cache" in self.quant_target:
            yield from iter_indexed_units(
                kind="attn",
                name_prefix="self_attn",
                layer_idx=layer_idx,
                items=block.self_attn,
                metadata_fn=lambda attn_idx, _: {"input_name": f"attn_{attn_idx}"},
            )
            return

        if "moe" in self.quant_target:
            quant_moe = find_moe_module(block)
            if quant_moe is None:
                raise TypeError("LongCat MoE PTQ expects a module with 'experts'.")
            experts = getattr(quant_moe, "experts", None)
            if experts is None:
                raise TypeError("LongCat MoE PTQ expected 'experts' on the located MoE module.")

            if hasattr(experts, "iter_ptq_expert_modules"):
                ptq_items = experts.iter_ptq_expert_modules()
            elif hasattr(experts, "expert_modules"):
                num_routed = getattr(experts, "num_routed_experts", None)
                expert_modules = experts.expert_modules
                ptq_items = expert_modules if num_routed is None else expert_modules[:num_routed]
            else:
                raise TypeError("LongCat MoE PTQ expects experts to expose PTQ iterable expert modules.")

            yield from iter_indexed_units(
                kind="moe",
                name_prefix="expert",
                layer_idx=layer_idx,
                items=ptq_items,
                metadata_fn=lambda expert_idx, _: {"expert_idx": expert_idx, "input_name": "moe"},
            )
            return

        if "mlp" in self.quant_target:
            yield from iter_indexed_units(
                kind="mlp",
                name_prefix="mlp",
                layer_idx=layer_idx,
                items=block.mlps,
                metadata_fn=lambda mlp_idx, _: {"input_name": f"mlp_{mlp_idx}"},
            )
            return

        yield from super().iter_ptq_units(layer_idx, block)

    def load_embed_state_dict(self):
        super().load_embed_state_dict()
        tie_weights = getattr(self.model, "tie_weights", None)
        if callable(tie_weights):
            tie_weights()

    def load_layer_weight(self, prefix):
        return super().load_layer_weight(prefix)

    def load_unit_inputs(self, data_dir, unit: PtqUnit):
        cached_inps, kwargs = super().load_unit_inputs(data_dir, unit)
        input_name = unit.metadata.get("input_name") if unit.metadata else None
        if input_name is None:
            return super().load_unit_inputs(data_dir, unit)
        path = Path(data_dir) / f"block_{unit.layer_idx}_{input_name}_in.pkl"
        if not path.exists():
            raise FileNotFoundError(f"PTQ inputs not found for unit '{unit.name}': {path}")
        cached_inps = torch.load(path, weights_only=True)
        return cached_inps, kwargs

    def register_block_forward_hooks(self, block, hook_name, hooks, act_stat):
        if "attn-linear" in self.quant_target or "attn-cache" in self.quant_target:
            name_map = {
                "input_layernorm.0": "attn_0",
                "input_layernorm.1": "attn_1",
            }
        elif "moe" in self.quant_target:
            name_map = {
                "post_attention_layernorm.0": "moe",
            }
        else:
            name_map = {
                "post_attention_layernorm.0": "mlp_0",
                "post_attention_layernorm.1": "mlp_1",
            }
        self._register_named_hook(block, name_map, act_stat, hooks)

    def save_block_hook_inputs(self, act_stat, hook_name, layer_idx):
        self._save_ptq_inputs(layer_idx, act_stat)

    def _load_longcat_layer_weight(self, layer_idx: int):
        prefix = self.get_layer_weight_prefix(layer_idx)
        state_dict = self.load_layer_weight(prefix)
        return self._pack_expert_weights(state_dict)

    def _pack_expert_weights(self, state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        gate_keys = [key for key in state_dict if key.startswith("mlp.experts.") and key.endswith(".gate_proj.weight")]
        if not gate_keys:
            return state_dict

        num_routed_experts = self.config.n_routed_experts
        zero_expert_num = getattr(self.config, "zero_expert_num", 0) or 0
        total_experts = num_routed_experts + zero_expert_num
        first_gate = state_dict["mlp.experts.0.gate_proj.weight"]
        first_down = state_dict["mlp.experts.0.down_proj.weight"]

        gate_up_proj = torch.zeros(
            total_experts,
            first_gate.shape[0] * 2,
            first_gate.shape[1],
            dtype=first_gate.dtype,
        )
        down_proj = torch.zeros(
            num_routed_experts,
            first_down.shape[0],
            first_down.shape[1],
            dtype=first_down.dtype,
        )

        for expert_idx in range(num_routed_experts):
            gate_key = f"mlp.experts.{expert_idx}.gate_proj.weight"
            up_key = f"mlp.experts.{expert_idx}.up_proj.weight"
            down_key = f"mlp.experts.{expert_idx}.down_proj.weight"
            if gate_key not in state_dict or up_key not in state_dict or down_key not in state_dict:
                raise KeyError(f"Missing routed expert weights for expert {expert_idx}.")
            gate_up_proj[expert_idx] = torch.cat(
                [state_dict.pop(gate_key), state_dict.pop(up_key)],
                dim=0,
            )
            down_proj[expert_idx] = state_dict.pop(down_key)

        state_dict["mlp.experts.gate_up_proj"] = gate_up_proj
        state_dict["mlp.experts.down_proj"] = down_proj
        return state_dict

    def _register_named_hook(self, block, name_map: dict[str, str], storage: dict[str, list[torch.Tensor]], hooks):
        def save_output(_, __, output, save_name):
            if isinstance(output, (tuple, list)):
                output = output[0]
            storage.setdefault(save_name, []).append(output.detach().cpu())

        for module_name, module in block.named_modules():
            if module_name not in name_map:
                continue
            hooks.append(module.register_forward_hook(lambda m, x, y, n=name_map[module_name]: save_output(m, x, y, n)))

    def _save_ptq_inputs(self, layer_idx: int, storage: dict[str, list[torch.Tensor]]):
        for save_name, tensors in storage.items():
            _save_tensor(
                os.path.join(self.args.data_dir, f"block_{layer_idx}_{save_name}_in.pkl"),
                tensors,
            )

    def _set_safe_attn_impl(self):
        self.config._attn_implementation = "eager"
