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
from compressed_tensors.utils.safetensors_load import get_weight_mappings
from transformers.models.glm_moe_dsa.configuration_glm_moe_dsa import GlmMoeDsaConfig
from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import GlmMoeDsaDecoderLayer

from amct_pytorch.common.models import MODEL_REGISTRY
from amct_pytorch.common.models.llm.common.base import BaseModel
from amct_pytorch.common.models.llm.common.quant_apply import (
    PlainLinear,
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

    @staticmethod
    def _collect_quant_bits(block):
        """Extract {module_name: w_bits} from QuantLinear modules in a block.

        Also maps packed expert paths to unpacked safetensors keys:
          mlp.experts.expert_modules.{idx}.gate_proj  ->  mlp.experts.{idx}.gate_proj
        """
        bits = {}
        for name, module in block.named_modules():
            if isinstance(module, QuantLinear) and module.w_bits is not None:
                name = name.replace(".expert_modules.", ".")
                bits[name] = module.w_bits
        return bits

    @staticmethod
    def _collect_raw_linear_names(block):
        """Collect nn.Linear module names not nested inside QuantLinear/PlainLinear.

        QuantLinear wraps nn.Linear, so block.named_modules() yields both the
        QuantLinear wrapper AND its inner .linear (a raw nn.Linear). This method
        filters out inner nn.Linear modules that are children of wrapper types,
        returning only standalone nn.Linear (e.g. indexer.weights_proj).
        """
        quant_linear_prefixes = tuple(
            f"{name}."
            for name, module in block.named_modules()
            if isinstance(module, QuantLinear)
        )
        plain_linear_prefixes = tuple(
            f"{name}."
            for name, module in block.named_modules()
            if isinstance(module, PlainLinear)
        )
        names = []
        for name, module in block.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            if any(name.startswith(p) for p in quant_linear_prefixes):
                continue
            if any(name.startswith(p) for p in plain_linear_prefixes):
                continue
            names.append(name)
        return names

    @staticmethod
    def _collect_plain_linear_names(block):
        """Collect PlainLinear wrapper module names from a quant block."""
        return [
            name
            for name, module in block.named_modules()
            if isinstance(module, PlainLinear)
        ]

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
        decoder_layer.load_state_dict(state_dict, strict=True)
        decoder_layer.eval().bfloat16()
        return decoder_layer

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

    def get_kv_cache_scheme(self):
        """Align with infer repo convert_model.py: c16 returns None, mxfp returns 8-bit float."""
        if self.args.quant_dtype == "mxfp":
            return {"num_bits": 8, "type": "float", "strategy": "group",
                    "group_size": 128, "dynamic": True, "symmetric": True}
        return None

    def get_quant_weight_specs(self):
        """Return {safetensors_weight_name: w_bits} for tensor-mode quantization.

        Overrides BaseModel to:
        1. Map packed expert paths to unpacked safetensors keys.
        2. Cover MTP layers via template reuse.

        Also caches built blocks in self._cached_empty_blocks so that
        get_extra_ignore_names() can reuse them without rebuilding.
        """
        self._cached_empty_blocks = {}
        specs = {}

        # 1. find not mtp layer quant linear.
        # Regular layers: build empty block, cache it, collect QuantLinear specs.
        # We inline the loop (instead of calling super()) to cache blocks for reuse.
        for layer_idx in range(self.num_layers):
            block = self.build_empty_quant_block(layer_idx)
            self._cached_empty_blocks[layer_idx] = block
            prefix = self.get_layer_weight_prefix(layer_idx)
            for name, bits in self._collect_quant_bits(block).items():
                specs[f"{prefix}{name}.weight"] = bits


        # 2. ensure mtp layer quant linear.

        # MTP layers are beyond num_layers (= num_hidden_layers), so the loop above
        # doesn't cover them. We can't read quantization decisions from index.json
        # because which weights are QuantLinear vs PlainLinear depends on quant_dtype
        # (see quant_module.py). Instead, build ONE template block from a regular
        # full-indexer MoE layer — MTP shares the same self_attn + mlp structure —
        # and copy its {module_name: w_bits} to each MTP layer's prefix.
        num_nextn = getattr(self.config, "num_nextn_predict_layers", 0)
        if num_nextn > 0:
            template_block = self.build_empty_quant_block(self._find_template_layer_idx())
            self._cached_empty_blocks["mtp_template"] = template_block
            template_bits = self._collect_quant_bits(template_block)
            for i in range(self.config.num_hidden_layers,
                           self.config.num_hidden_layers + num_nextn):
                prefix = f"model.layers.{i}."
                for name, bits in template_bits.items():
                    specs[f"{prefix}{name}.weight"] = bits

        return specs

    def get_extra_ignore_names(self, quant_mode="block", quant_specs=None):
        """Return ignore entries not discoverable by regular block traversal.

        Args:
            quant_mode: "block" or "tensor". In block mode, MTP layers are never
                processed so all MTP modules are ignored. In tensor mode, only
                modules NOT in quant_specs are ignored.
            quant_specs: {weight_name: w_bits} dict. Required when quant_mode="tensor".
        """
        if quant_mode == "tensor":
            return self._get_extra_ignore_names_tensor(quant_specs)
        return self._get_extra_ignore_names_block()

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

        def _capture(*args, **kwargs):
            out = _orig(*args, **kwargs)
            if isinstance(out, (tuple, list)) and len(out) >= 2:
                self._dsa_topk_indices = out[1]
            return out

        block.forward = _capture
        return block

    def _is_full_indexer_layer(self, layer_idx, indexer_types):
        """Check if a layer carries a full Lightning Indexer.

        GLM-5.2 IndexShare: "shared" layers reuse the previous full layer's top-k
        and have NO indexer tensors. Only "full" layers carry their own indexer weights.
        MTP layers (layer_idx >= num_hidden_layers) always carry a full indexer.
        """
        if layer_idx >= self.config.num_hidden_layers:
            return True
        if not indexer_types:
            return True
        return indexer_types[layer_idx] == "full"

    def _find_template_layer_idx(self):
        """Find the first MoE layer with a full indexer for use as MTP template.

        MTP layers are NOT instantiated by GlmMoeDsaDecoderLayer (which only covers
        self_attn + mlp, no eh_proj/shared_head). But MTP's self_attn and mlp share
        the exact same structure as a regular full-indexer MoE layer. So we pick one
        such regular layer, build its empty quant block, and use it as the template
        to derive which weights MTP layers should quantize or ignore.
        """
        first_k_dense = self.config.first_k_dense_replace
        indexer_types = getattr(self.config, "indexer_types", None)
        for i in range(first_k_dense, self.config.num_hidden_layers):
            if self._is_full_indexer_layer(i, indexer_types):
                return i
        return first_k_dense

    def _get_cached_block(self, layer_idx, is_mtp=False):
        """Retrieve a cached empty quant block, building on demand if not cached.

        When get_quant_weight_specs() runs first (tensorwise quant path), blocks are
        already cached and this is a free dict lookup. For other paths (blockwise)
        that call get_extra_ignore_names() without get_quant_weight_specs(),
        blocks are built lazily here.
        """
        cache = getattr(self, "_cached_empty_blocks", None) or {}
        if is_mtp:
            if "mtp_template" not in cache:
                cache["mtp_template"] = self.build_empty_quant_block(
                    self._find_template_layer_idx()
                )
            return cache["mtp_template"]
        if layer_idx not in cache:
            cache[layer_idx] = self.build_empty_quant_block(layer_idx)
        return cache[layer_idx]


    def _get_extra_ignore_names_block(self):
        """Block mode: MTP layers are never exported, so ignore all MTP modules."""
        ignore = []
        num_hidden = self.config.num_hidden_layers
        num_nextn = getattr(self.config, "num_nextn_predict_layers", 0)
        first_k_dense = self.config.first_k_dense_replace
        num_layers = num_hidden + num_nextn

        for i in range(num_layers):
            prefix = f"model.layers.{i}."
            is_mtp = i >= num_hidden
            block = self._get_cached_block(i, is_mtp=is_mtp)

            for name in self._collect_raw_linear_names(block):
                ignore.append(f"{prefix}{name}")
            for name in self._collect_plain_linear_names(block):
                ignore.append(f"{prefix}{name}")

            if is_mtp:
                for mtp_name in ("eh_proj", "shared_head.head"):
                    ignore.append(f"{prefix}{mtp_name}")

            if i >= first_k_dense:
                ignore.append(f"model.layers.{i}.mlp.gate")

        ignore.append("lm_head")
        return ignore

    def _collect_tensor_ignore_for_layer(self, i, prefix, is_mtp, block, quant_specs):
        """Collect ignore entries for a single layer in tensor mode."""
        layer_ignore = []
        for name in self._collect_raw_linear_names(block):
            if f"{prefix}{name}.weight" not in quant_specs:
                layer_ignore.append(f"{prefix}{name}")
        for name in self._collect_plain_linear_names(block):
            if f"{prefix}{name}.weight" not in quant_specs:
                layer_ignore.append(f"{prefix}{name}")

        if is_mtp:
            mtp_names = ("eh_proj", "shared_head.head")
            layer_ignore.extend(
                f"{prefix}{n}" for n in mtp_names
                if f"{prefix}{n}.weight" not in quant_specs
            )

        if i >= self.config.first_k_dense_replace:
            layer_ignore.append(f"model.layers.{i}.mlp.gate")
        return layer_ignore

    def _get_extra_ignore_names_tensor(self, quant_specs):
        """Tensor mode: ignore modules NOT in quant_specs (mutual exclusion)."""
        ignore = []
        num_hidden = self.config.num_hidden_layers
        num_nextn = getattr(self.config, "num_nextn_predict_layers", 0)
        num_layers = num_hidden + num_nextn

        for i in range(num_layers):
            prefix = f"model.layers.{i}."
            is_mtp = i >= num_hidden
            block = self._get_cached_block(i, is_mtp=is_mtp)
            ignore.extend(self._collect_tensor_ignore_for_layer(
                i, prefix, is_mtp, block, quant_specs
            ))

        ignore.append("lm_head")
        return ignore
