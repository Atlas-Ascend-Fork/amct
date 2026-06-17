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
from loguru import logger
from compressed_tensors.utils.safetensors_load import get_weight_mappings
from transformers.models.glm_moe_dsa.configuration_glm_moe_dsa import GlmMoeDsaConfig
from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import GlmMoeDsaDecoderLayer

from amct_pytorch.common.models import MODEL_REGISTRY
from amct_pytorch.common.models.llm.common.base import BaseModel, PtqUnit
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

_INDEXER_KEYS = frozenset({
    "self_attn.indexer.wq_b.weight",
    "self_attn.indexer.wk.weight",
    "self_attn.indexer.k_norm.weight",
    "self_attn.indexer.k_norm.bias",
    "self_attn.indexer.weights_proj.weight",
})


@MODEL_REGISTRY.register(
    name="glm_5_2",
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
        self.textconfig = GlmMoeDsaConfig
        self._weight_map = get_weight_mappings(self.model_path)
        self.num_layers = self.config.num_hidden_layers
        self.cls = GlmMoeDsaDecoderLayer
        self.model = self.empty_weights_model()
        self._dsa_topk_indices = None
        self.parse_quant_mode()
        # Dump indexer config summary
        types = self.config.indexer_types
        full_cnt = sum(1 for t in types if t == "full")
        shared_cnt = sum(1 for t in types if t == "shared")
        logger.info(
            f"[GLM5_2] init: layers={self.num_layers}, "
            f"indexer full={full_cnt}, shared={shared_cnt}, "
            f"kv_lora={self.config.kv_lora_rank}, "
            f"qk_rope={self.config.qk_rope_head_dim}, "
            f"qk_nope={self.config.qk_nope_head_dim}"
        )
        logger.info(f"[GLM5_2]   indexer_types[:12]={types[:12]}")

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

    def block(self, layer_idx):
        decoder_layer = self.cls(self.config, layer_idx)
        state_dict = self.load_layer_weight(self.get_layer_weight_prefix(layer_idx))
        missing, unexpected = decoder_layer.load_state_dict(
            state_dict, strict=False
        )
        non_indexer_missing = [k for k in missing if k not in _INDEXER_KEYS]
        if non_indexer_missing:
            raise RuntimeError(
                f"Missing keys in state_dict for layer {layer_idx}: "
                f"{non_indexer_missing}"
            )
        indexer_missing = [k for k in missing if k in _INDEXER_KEYS]
        is_shared = self.config.indexer_types[layer_idx] == "shared"
        if indexer_missing:
            logger.info(f"[GLM5_2] layer {layer_idx} ({'shared' if is_shared else 'full'}): "
                        f"indexer keys missing={len(indexer_missing)}, "
                        f"non_indexer_missing={len(non_indexer_missing)}")
        if indexer_missing and not is_shared:
            raise RuntimeError(
                f"Layer {layer_idx} is indexer_types='full' but missing "
                f"indexer weights: {indexer_missing}"
            )
        decoder_layer.eval().bfloat16()
        return decoder_layer

    def do_embedding_forward(self, samples, dtype=torch.bfloat16, hook_name=None):
        result = super().do_embedding_forward(samples, dtype=dtype, hook_name=hook_name)
        pe_ok = self.position_embeddings is not None
        am_ok = self.attention_mask is not None
        pi_ok = self.position_ids is not None
        logger.info(f"[DEBUG] embed_forward done: pos_emb={pe_ok}, attn_mask={am_ok}, pos_ids={pi_ok}")
        if pe_ok:
            logger.info(f"[DEBUG]   pos_emb cos shape={self.position_embeddings[0].shape}, sin shape={self.position_embeddings[1].shape}")
        if am_ok:
            logger.info(f"[DEBUG]   attn_mask shape={self.attention_mask.shape}, dtype={self.attention_mask.dtype}")
        logger.info(f"[DEBUG]   inter_io samples={len(result)}, first sample shape={result[0].shape}")
        return result

    def _build_block_for_forward(self, layer_idx, use_quant_block=False):
        block = super()._build_block_for_forward(
            layer_idx, use_quant_block=use_quant_block
        )
        _orig = block.forward
        ltype = self.config.indexer_types[layer_idx]
        prev_was = self._dsa_topk_indices

        def _capture(*args, **kwargs):
            out = _orig(*args, **kwargs)
            if isinstance(out, (tuple, list)) and len(out) >= 2:
                self._dsa_topk_indices = out[1]
            return out

        block.forward = _capture
        prev_str = "None"
        if prev_was is not None:
            prev_str = f"shape={prev_was.shape}"
        logger.info(f"[GLM5_2] build_fwd layer {layer_idx:3d} ({ltype:6s}): "
                    f"prev_topk_before={prev_str}")
        return block

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
        has_topk = self._dsa_topk_indices is not None
        if has_topk:
            kwargs["prev_topk_indices"] = self._dsa_topk_indices
            topk_shape = self._dsa_topk_indices.shape
            logger.info(f"[GLM5_2] get_block_kwargs: injecting prev_topk shape={topk_shape}")
        else:
            logger.info(f"[GLM5_2] get_block_kwargs: prev_topk=None, "
                        f"pos_emb={'yes' if 'position_embeddings' in kwargs else 'no'}, "
                        f"attn_mask={'yes' if 'attention_mask' in kwargs else 'no'}")
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

    def iter_ptq_units(self, layer_idx, block):
        yield from super().iter_ptq_units(layer_idx, block)

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

    def load_unit_inputs(self, data_dir, unit: PtqUnit):
        return super().load_unit_inputs(data_dir, unit)
