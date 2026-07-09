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

import functools

import torch
import torch.nn as nn
from compressed_tensors.utils.safetensors_load import get_weight_mappings
from transformers.models.glm_moe_dsa.configuration_glm_moe_dsa import GlmMoeDsaConfig
from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import GlmMoeDsaDecoderLayer

from amct_pytorch.common.models import MODEL_REGISTRY
from amct_pytorch.common.models.llm.common.base import BaseModel
from amct_pytorch.common.models.llm.common.quant_apply import (
    QuantGatedMLP,
    apply_quant_to_attn,
)
from amct_pytorch.common.models.llm.glm.glm5_2.quant_module import QuantGlmMoeDsaAttention
from amct_pytorch.common.models.llm.qwen.moe_common import (
    QuantGatedExperts,
    is_packed_experts,
    pack_gated_expert_weights,
)
from amct_pytorch.quantization.modules.quant_linear import QuantLinear


@MODEL_REGISTRY.register(
    name="glm5_2",
    task="llm",
    family="glm",
    description="GLM-5.2 MoE DSA model adapter",
)
class GLM5_2(BaseModel):
    """Adapter for GLM-5.2 MoE model with DeepSeek Sparse Attention (DSA) + IndexShare.

    A completely independent adapter (does NOT inherit from DeepseekV32).
    Handles MLA attention, DSA indexer, MoE gated experts (packed), and
    shared experts.
    """

    def __init__(self, args):
        super().__init__(args)
        self.config._attn_implementation = "eager"
        self.textconfig = GlmMoeDsaConfig
        self._weight_map = get_weight_mappings(self.model_path)
        self.num_layers = self.config.num_hidden_layers
        self.cls = GlmMoeDsaDecoderLayer
        self.model = self.empty_weights_model()
        self._dsa_topk_indices = None
        self.parse_quant_mode()

    def parse_quant_mode(self):
        if "mlp" in self.quant_target:
            raise ValueError(
                "GLM-5.2 is a moe model and does not support quant_target='mlp'."
            )

    def float_model(self):
        return super().float_model()

    def empty_weights_model(self):
        return super().empty_weights_model()

    def get_layer_weight_prefix(self, layer_idx: int) -> str:
        return f"model.layers.{layer_idx}."

    def load_layer_weight(self, prefix):
        state_dict = super().load_layer_weight(prefix)
        state_dict = pack_gated_expert_weights(
            state_dict, expert_prefix="mlp.experts"
        )
        return state_dict

    def load_embed_state_dict(self):
        return super().load_embed_state_dict()

    def do_embedding_forward(self, samples, dtype=torch.bfloat16, hook_name=None):
        return super().do_embedding_forward(samples, dtype=dtype, hook_name=hook_name)

    def do_block_forward(self, layer_idx, samples, hook_name=None,
                         use_quant_block=False, enable_quant=False):
        return super().do_block_forward(
            layer_idx, samples,
            hook_name=hook_name,
            use_quant_block=use_quant_block,
            enable_quant=enable_quant,
        )

    def get_block_forward_kwargs(self):
        kwargs = super().get_block_forward_kwargs()
        if self._dsa_topk_indices is not None:
            kwargs["prev_topk_indices"] = self._dsa_topk_indices
        return kwargs

    def do_head_forward(self, inps):
        return super().do_head_forward(inps)

    def build_quant_block(self, layer_idx):
        decoder_layer = self.block(layer_idx)

        if "attn-linear" in self.quant_target or "attn-cache" in self.quant_target:
            apply_quant_to_attn(self.args, decoder_layer, QuantGlmMoeDsaAttention)

        if "moe" in self.quant_target or "mlp" in self.quant_target:
            mlp = getattr(decoder_layer, "mlp", None)
            if mlp is None:
                raise ValueError(
                    f"MLP not found on decoder layer {layer_idx}"
                )
            if hasattr(mlp, "experts") and is_packed_experts(mlp.experts):
                mlp.experts = QuantGatedExperts(self.args, mlp.experts)
                if hasattr(mlp, "shared_experts"):
                    mlp.shared_experts = QuantGatedMLP(
                        self.args, mlp.shared_experts, group="moe.shared"
                    )
            else:
                decoder_layer.mlp = QuantGatedMLP(self.args, mlp, group="mlp")

        return decoder_layer

    def get_scale_name(self, weight_name):
        """Return (scale_prefix, scale_inv_name) for locating scale tensors during dequantization."""
        scale_prefix = ".scale"
        scale_inv_name = weight_name.replace(".weight", ".scale")
        return scale_prefix, scale_inv_name


    def cache_scheme(self):
        """Return cache scheme dict for deploy config.

        Aligns with infer repo convert_model.py:
        - kv_cache_scheme: only quantized for mxfp (8-bit float, group-wise).
          For non-mxfp (c16/int) the key is kept but set to None to signal
          "no KV cache quantization".
        - li_cache_scheme: always present, type follows quant_dtype
          (mxfp -> "float", otherwise "int").
        """
        is_mxfp = self.args.quant_dtype == "mxfp"
        li_cache_scheme = {
            "type": "float" if is_mxfp else "int",
            "num_bits": 8,
        }
        if not is_mxfp:
            return {"kv_cache_scheme": None, "li_cache_scheme": li_cache_scheme}
        return {
            "kv_cache_scheme": {
                "num_bits": 8,
                "type": "float",
                "strategy": "group",
                "group_size": 128,
                "dynamic": True,
                "symmetric": True,
            },
            "li_cache_scheme": li_cache_scheme,
        }

    def iter_deploy_bindings(self, layer_idx, block):
        weight_prefix = self.get_layer_weight_prefix(layer_idx)
        for name, module in block.named_modules():
            if not isinstance(module, QuantLinear):
                continue

            # MoE experts: translate packed module path to unpacked checkpoint path
            #   packed path:  mlp.experts.expert_modules.{idx}.gate_proj
            #   unpacked:     mlp.experts.{idx}.gate_proj.weight
            if name.startswith("mlp.experts.expert_modules."):
                parts = name.split(".")
                if len(parts) == 5:
                    _, _, _, expert_idx, proj_name = parts
                    yield (
                        f"{weight_prefix}mlp.experts.{expert_idx}.{proj_name}.weight",
                        module,
                    )
                    continue

            yield f"{weight_prefix}{name}.weight", module

    def _build_block_for_forward(self, layer_idx, use_quant_block=False):
        block = super()._build_block_for_forward(
            layer_idx, use_quant_block=use_quant_block
        )
        _orig = block.forward

        @functools.wraps(_orig)
        def _capture(*args, **kwargs):
            out = _orig(*args, **kwargs)
            if isinstance(out, (tuple, list)) and len(out) >= 2:
                self._dsa_topk_indices = out[1]
            return out

        block.forward = _capture
        return block

