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
import torch.nn as nn
from tqdm import tqdm
from loguru import logger
from safetensors import safe_open
from compressed_tensors.utils.safetensors_load import get_weight_mappings
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeDecoderLayer, Qwen3_5MoeTextConfig

from amct_pytorch.common.models import MODEL_REGISTRY

from amct_pytorch.common.models.llm.common.base import BaseModel, PtqUnit
from amct_pytorch.common.models.llm.common.moe_unpack import find_moe_module
from amct_pytorch.common.models.llm.common.ptq_units import iter_indexed_units, make_ptq_unit
from amct_pytorch.common.models.llm.qwen.moe_common import QuantGatedExperts, is_packed_experts
from amct_pytorch.common.models.llm.qwen.qwen3_5.qwen3_5 import Qwen3_5, QuantQwen35MLP

from amct_pytorch.common.models.llm.common.quant_apply import apply_quant_to_attn, build_no_algo_args
from amct_pytorch.quantization.modules.quant_linear import QuantLinear
from amct_pytorch.quantization.bit_policy import ensure_bit_policy


@MODEL_REGISTRY.register(
    name="qwen3_5_moe",
    task="llm",
    family="qwen",
    description="Qwen3.5 moe model adapter",
)
class Qwen3_5Moe(Qwen3_5):
    def __init__(self, args):
        super().__init__(args)
        self.cls = Qwen3_5MoeDecoderLayer
        self.textconfig = Qwen3_5MoeTextConfig

    def parse_quant_mode(self):
        if "mlp" in self.quant_target:
            raise ValueError("Qwen3.5-moe is a moe model and does not support quant_target='mlp'.")

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

        if "moe" in self.quant_target:
            quant_moe = find_moe_module(decoder_layer)
            if quant_moe is not None:
                experts = getattr(quant_moe, "experts", None)
                if experts is not None and is_packed_experts(experts):
                    quant_moe.experts = QuantGatedExperts(self.args, experts, group="moe.routed")
                    shared_expert = getattr(quant_moe, "shared_expert", None)
                    if shared_expert is not None and not isinstance(shared_expert, QuantQwen35MLP):
                        shared_expert_args = build_no_algo_args(self.args)
                        quant_moe.shared_expert = QuantQwen35MLP(shared_expert_args, shared_expert, group="moe.shared")

        return decoder_layer

    def iter_ptq_units(self, layer_idx, block):
        yield from super().iter_ptq_units(layer_idx, block)

    def bits_scheme(self):
        """Per-group (targets, w_bits, a_bits) for the deploy config groups.

        Declares one config group per quantized target so infer's
        find_matched_target resolves every non-ignored QuantLinear:

        - ``attn-linear`` -> ``Linear`` group at the top-level bit-widths
          (attention linears).
        - ``moe`` -> a ``MoEGMM`` group for the routed experts (W4A8, from
          ``moe.routed``) AND a ``Linear`` group for the shared_expert, which
          build_quant_block wraps as a QuantLinear at ``moe.shared`` bit-widths
          (W8A8). Without the Linear group the shared_expert weights are
          quantized but unmatched, so infer aborts with "Unable to find
          matching target". This mirrors deepseek_v4's two-group W4A8 recipe
          (Linear=W8A8 for shared_expert, MoEGMM=W4A8 for routed).

        A ``Linear`` target is emitted at most once; when both attn-linear and
        moe are requested the attention (top-level) bit-widths win. Returns
        ``None`` when nothing matches so generate_quant_config keeps its
        default.
        """
        bit_policy = ensure_bit_policy(self.args)
        linear_bits = None
        moe_gmm_bits = None
        if "moe" in self.quant_target:
            shared = bit_policy["moe.shared"].default
            linear_bits = (shared.w, shared.a)
            routed = bit_policy["moe.routed"].default
            moe_gmm_bits = (routed.w, routed.a)
        if "attn-linear" in self.quant_target:
            # attention (top-level) bit-widths take precedence for the Linear group
            linear_bits = (bit_policy.w_bits, bit_policy.a_bits)

        groups = []
        if linear_bits is not None:
            # linear_attn (Gated DeltaNet) projections are quantized alongside
            # the attention linears at the same bit-widths, but infer represents
            # them as custom modules whose class name is not "Linear", so the
            # bare "Linear" target never matches them. Declare them explicitly
            # via regex targets or infer aborts with "Unable to find matching
            # target" for the quantized-but-unmatched in_proj/out_proj weights.
            linear_targets = ["Linear"]
            if "attn-linear" in self.quant_target:
                linear_targets += [
                    r"re:.*\.linear_attn\.in_proj_qkv",
                    r"re:.*\.linear_attn\.in_proj_z",
                    r"re:.*\.linear_attn\.in_proj_a",
                    r"re:.*\.linear_attn\.in_proj_b",
                    r"re:.*\.linear_attn\.out_proj",
                ]
            groups.append({"targets": linear_targets,
                           "w_bits": linear_bits[0], "a_bits": linear_bits[1]})
        if moe_gmm_bits is not None:
            groups.append({"targets": ["MoEGMM"],
                           "w_bits": moe_gmm_bits[0], "a_bits": moe_gmm_bits[1]})
        return groups or None

    def generate_deploy_ignore_layers(self):
        """Top-level (non-block) modules that deploy leaves unquantized.

        The per-block ignore scan (get_quant_ignore_linear_names) only sees
        modules inside decoder blocks, so lm_head - which is never quantized -
        must be declared here. Without it infer matches lm_head against the
        Linear target group and aborts for the missing weight_scale. When
        weights are tied, lm_head shares embed_tokens and has no standalone
        tensor to ignore.
        """
        ignore = []
        if not bool(getattr(self.config, "tie_word_embeddings", False)):
            ignore.append("lm_head")
        return ignore

    def iter_deploy_bindings(self, layer_idx, block):
        weight_prefix = self.get_layer_weight_prefix(layer_idx)
        for name, module in block.named_modules():
            if not isinstance(module, QuantLinear):
                continue

            if name.startswith("mlp.experts.expert_modules."):
                parts = name.split(".")
                if len(parts) != 5:
                    raise ValueError(f"Unexpected Qwen3.5 MoE expert module name: {name}")
                _, _, _, expert_idx, proj_name = parts
                yield f"{weight_prefix}mlp.experts.{expert_idx}.{proj_name}.weight", module
                continue

            yield f"{weight_prefix}{name}.weight", module

    def load_unit_inputs(self, data_dir, unit: PtqUnit):
        return super().load_unit_inputs(data_dir, unit)
