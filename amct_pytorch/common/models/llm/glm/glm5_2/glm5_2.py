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

from __future__ import annotations

import functools
import re

import torch
from compressed_tensors.utils.safetensors_load import get_weight_mappings
from loguru import logger
from transformers.models.glm_moe_dsa.configuration_glm_moe_dsa import GlmMoeDsaConfig
from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import GlmMoeDsaDecoderLayer

from amct_pytorch.common.models import MODEL_REGISTRY
from amct_pytorch.common.models.llm.common.base import BaseModel
from amct_pytorch.common.models.llm.common.deployment.deploy_selection import (
    get_structured_quant_spec,
)
from amct_pytorch.common.models.llm.common.deployment.schemes import (
    PLATFORM_QUANT_DTYPES,
)
from amct_pytorch.common.models.llm.common.quant_apply import (
    QuantGatedMLP,
    apply_quant_to_attn,
)
from amct_pytorch.common.models.llm.glm.glm5_2.quant_module import (
    QuantGlmMoeDsaAttention,
)
from amct_pytorch.common.models.llm.qwen.moe_common import (
    QuantGatedExperts,
    is_packed_experts,
    pack_gated_expert_weights,
)
from amct_pytorch.quantization.bit_policy import ensure_bit_policy
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

    MLP_LINEAR_NAMES = ("gate_proj", "down_proj", "up_proj")

    def __init__(self, args):
        super().__init__(args)
        self.config._attn_implementation = "eager"
        self.textconfig = GlmMoeDsaConfig
        self._weight_map = get_weight_mappings(self.model_path)
        self.num_layers = self.config.num_hidden_layers
        self.cls = GlmMoeDsaDecoderLayer
        self.model = self.empty_weights_model()
        self._dsa_topk_indices = []
        self.parse_quant_mode()

    @staticmethod
    def _is_indexer_layer(layer_idx, num_hidden_layers, indexer_types):
        """Check if a layer owns indexer weights (IndexShare: only "full" + MTP layers)."""
        if layer_idx >= num_hidden_layers:
            return True
        if not indexer_types or layer_idx >= len(indexer_types):
            return True
        return indexer_types[layer_idx] == "full"

    def parse_quant_mode(self):
        if (
            getattr(self.args, "granularity", None) == "tensor"
            and getattr(self.args, "deploy_selection", None) is not None
        ):
            return
        if "mlp" in self.quant_target:
            raise ValueError(
                "GLM-5.2 is a moe model and does not support quant_target='mlp'."
            )

    def _deploy_role(self, name: str) -> tuple[str, str] | None:
        """Map supported checkpoint modules to BitPolicy group/projection."""
        match = re.fullmatch(r"model\.layers\.(\d+)\.(.+)", name)
        if match is None:
            return None
        layer_idx, suffix = int(match[1]), match[2]
        if not 0 <= layer_idx < self.config.num_hidden_layers:
            return None
        if suffix in {
            "self_attn.q_a_proj",
            "self_attn.q_b_proj",
            "self_attn.kv_a_proj_with_mqa",
            "self_attn.o_proj",
        }:
            return "attn-linear", suffix.rsplit(".", 1)[1]
        if suffix == "self_attn.indexer.wq_b":
            if self._is_indexer_layer(
                layer_idx,
                self.config.num_hidden_layers,
                getattr(self.config, "indexer_types", None),
            ):
                return "attn-linear", "wq_b"
            return None
        if suffix == "self_attn.indexer.wk":
            if self._is_indexer_layer(
                layer_idx,
                self.config.num_hidden_layers,
                getattr(self.config, "indexer_types", None),
            ):
                return "attn-linear", "wk"
            return None
        dense = layer_idx < self.config.first_k_dense_replace
        if dense and suffix in {f"mlp.{p}" for p in self.MLP_LINEAR_NAMES}:
            return "mlp", suffix.rsplit(".", 1)[1]
        if not dense:
            if getattr(self.config, "n_shared_experts", 0) and suffix in {
                f"mlp.shared_experts.{p}" for p in self.MLP_LINEAR_NAMES
            }:
                return "moe.shared", suffix.rsplit(".", 1)[1]
            expert = re.fullmatch(
                r"mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)", suffix
            )
            if expert and int(expert[1]) < self.config.n_routed_experts:
                return "moe.routed", expert[2]
        return None

    def get_deploy_candidates(self, module_names: set[str]) -> set[str]:
        """INT candidates independent of legacy INT exclusions."""
        return {name for name in module_names if self._deploy_role(name) is not None}

    @staticmethod
    def get_deploy_capabilities():
        return PLATFORM_QUANT_DTYPES

    def validate_deploy_selection(self, selected: set[str]) -> None:
        selection = getattr(self.args, "deploy_selection", None)
        structured = getattr(selection, "uses_structured_rules", False)
        policy = None if structured else ensure_bit_policy(self.args)
        for name in sorted(selected):
            role = self._deploy_role(name)
            if role is None:
                raise ValueError(f"Unsupported deployment module: {name}")
            group, projection = role
            if structured:
                spec = get_structured_quant_spec(selection, name)
                bits = (spec.weight_bits, spec.activation_bits)
            else:
                bits = policy.linear_bits(name=projection, group=group)
            if bits != (8, 8):
                raise ValueError(f"deployment requires W8A8 for {name}, got {bits}")

        def require_pair(first, second):
            if (first in selected) != (second in selected):
                raise ValueError(
                    f"Fused modules must share quantization: {first}, {second}"
                )

        for idx in range(self.config.num_hidden_layers):
            prefix = f"model.layers.{idx}."
            require_pair(
                prefix + "self_attn.q_a_proj", prefix + "self_attn.kv_a_proj_with_mqa"
            )
            mlp = prefix + "mlp."
            if idx < self.config.first_k_dense_replace:
                require_pair(mlp + "gate_proj", mlp + "up_proj")
                continue
            require_pair(
                mlp + "shared_experts.gate_proj", mlp + "shared_experts.up_proj"
            )
            experts = {
                f"{mlp}experts.{expert}.{proj}"
                for expert in range(self.config.n_routed_experts)
                for proj in self.MLP_LINEAR_NAMES
            }
            if selected & experts and not experts <= selected:
                raise ValueError(
                    f"Deployment requires all experts and projections in {mlp}experts "
                    "to share quantization"
                )

    def float_model(self):
        return super().float_model()

    def empty_weights_model(self):
        return super().empty_weights_model()

    def get_layer_weight_prefix(self, layer_idx: int) -> str:
        return f"model.layers.{layer_idx}."

    def load_layer_weight(self, prefix):
        state_dict = super().load_layer_weight(prefix)
        state_dict = pack_gated_expert_weights(state_dict, expert_prefix="mlp.experts")
        return state_dict

    def load_embed_state_dict(self):
        return super().load_embed_state_dict()

    def do_embedding_forward(self, samples, dtype=torch.bfloat16, hook_name=None):
        return super().do_embedding_forward(samples, dtype=dtype, hook_name=hook_name)

    def do_block_forward(
        self,
        layer_idx,
        samples,
        hook_name=None,
        use_quant_block=False,
        enable_quant=False,
    ):
        return super().do_block_forward(
            layer_idx,
            samples,
            hook_name=hook_name,
            use_quant_block=use_quant_block,
            enable_quant=enable_quant,
        )

    def get_block_forward_kwargs(self):
        # Per-sample IndexShare state is injected by the block forward wrapper.
        return super().get_block_forward_kwargs()

    def export_eval_state(self):
        return self._dsa_topk_indices

    def restore_eval_state(self, state, completed_block_idx, inter_io):
        self._dsa_topk_indices = []
        next_layer_idx = completed_block_idx + 1
        if completed_block_idx < 0 or next_layer_idx >= self.num_layers:
            return True
        if not isinstance(state, list) or len(state) != len(inter_io):
            logger.info(
                f"Missing or mismatched indexer state before block {next_layer_idx}; "
                f"expected {len(inter_io)} samples."
            )
            return False
        for sample_idx, indices in enumerate(state):
            if not isinstance(indices, torch.Tensor):
                logger.info(
                    f"Invalid indexer state for sample {sample_idx} before "
                    f"block {next_layer_idx}."
                )
                return False
        self._dsa_topk_indices = [indices.detach().cpu() for indices in state]
        return True

    def do_head_forward(self, inps):
        return super().do_head_forward(inps)

    def build_quant_block(self, layer_idx):
        decoder_layer = self.block(layer_idx)

        if "attn-linear" in self.quant_target or "attn-cache" in self.quant_target:
            apply_quant_to_attn(self.args, decoder_layer, QuantGlmMoeDsaAttention)

        if "moe" in self.quant_target or "mlp" in self.quant_target:
            mlp = getattr(decoder_layer, "mlp", None)
            if mlp is None:
                raise ValueError(f"MLP not found on decoder layer {layer_idx}")
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

        - non-mxfp: kv_cache_scheme=None, li_cache_scheme type=int.
        - mxfp: kv_cache_scheme quantized (8-bit float, group-wise),
          li_cache_scheme type=float, plus activation_scheme='dynamic'.
        - w8a8-mxfp: quant_method='mxfp8'; w4a8-mxfp: quant_method omitted.
        """
        bit_policy = ensure_bit_policy(self.args)
        is_mxfp = self.args.quant_dtype == "mxfp"
        is_w4a8 = bit_policy["moe.routed"].default.w == 4
        li_cache_scheme = {
            "type": "float" if is_mxfp else "int",
            "num_bits": 8,
        }
        if not is_mxfp:
            # infer repo keeps kv_cache_scheme=None for non-mxfp
            return {"kv_cache_scheme": None, "li_cache_scheme": li_cache_scheme}
        scheme = {
            "kv_cache_scheme": {
                "num_bits": 8,
                "type": "float",
                "strategy": "group",
                "group_size": 128,
                "dynamic": "true",  # string, not bool, per infer repo
                "symmetric": "true",  # string, not bool, per infer repo
            },
            "li_cache_scheme": li_cache_scheme,
            "activation_scheme": "dynamic",  # mxfp-only field per infer repo
        }
        if not is_w4a8:
            # infer repo uses mxfp8 (not compressed-tensors) for w8a8-mx
            scheme["quant_method"] = "mxfp8"
        return scheme

    def bits_scheme(self):
        """Per-group (targets, w_bits, a_bits) for the deploy config groups."""
        bit_policy = ensure_bit_policy(self.args)
        routed = bit_policy["moe.routed"].default
        groups = [
            {
                "targets": ["Linear"],
                "w_bits": bit_policy.w_bits,
                "a_bits": bit_policy.a_bits,
            },
        ]
        # group_1 (MoEGMM) only for w4a8, per infer repo
        if routed.w == 4:
            groups.append(
                {"targets": ["MoEGMM"], "w_bits": routed.w, "a_bits": routed.a},
            )
        return groups

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
        previous_indices = self._dsa_topk_indices
        current_indices = []

        @functools.wraps(_orig)
        def _capture(*args, **kwargs):
            sample_idx = len(current_indices)
            if previous_indices:
                if sample_idx >= len(previous_indices):
                    raise ValueError(
                        f"Missing indexer state for sample {sample_idx} in block {layer_idx}."
                    )
                hidden_states = args[0] if args else kwargs["hidden_states"]
                kwargs["prev_topk_indices"] = previous_indices[sample_idx].to(
                    hidden_states.device
                )
            out = _orig(*args, **kwargs)
            indices = (
                out[1] if isinstance(out, (tuple, list)) and len(out) >= 2 else None
            )
            if not isinstance(indices, torch.Tensor):
                raise TypeError(f"Block {layer_idx} did not return indexer indices.")
            current_indices.append(indices.detach().cpu())
            self._dsa_topk_indices = current_indices
            return out

        block.forward = _capture
        return block
