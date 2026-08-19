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

import json
from functools import cached_property
from pathlib import Path

import torch
from safetensors import safe_open
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5DecoderLayer,
    Qwen3_5TextConfig,
)
from transformers.masking_utils import causal_mask_function, create_causal_mask

from amct_pytorch.common.models import MODEL_REGISTRY
from amct_pytorch.common.datasets.ptq_io import save_ptq_kwargs
from amct_pytorch.common.models.llm.common.base import BaseModel, PtqUnit
from amct_pytorch.common.models.llm.common.quant_apply import (
    apply_quant_to_attn,
    apply_quant_to_moe_mlp,
)
from amct_pytorch.common.models.llm.qwen.qwen3_5.quant_module import (
    QuantQwen35Attn,
    QuantQwen35LinearAttn,
    QuantQwen35MLP,
)
from amct_pytorch.quantization.bit_policy import ensure_bit_policy


_LINEAR_ROLE_TARGETS = (
    ("mlp", "gate_proj", r"re:.*\.mlp\.gate_proj$"),
    ("mlp", "up_proj", r"re:.*\.mlp\.up_proj$"),
    ("mlp", "down_proj", r"re:.*\.mlp\.down_proj$"),
    ("attn-linear", "in_proj_z", r"re:.*\.linear_attn\.in_proj_z$"),
    ("attn-linear", "in_proj_b", r"re:.*\.linear_attn\.in_proj_b$"),
    ("attn-linear", "in_proj_a", r"re:.*\.linear_attn\.in_proj_a$"),
    ("attn-linear", "in_proj_qkv", r"re:.*\.linear_attn\.in_proj_qkv$"),
    ("attn-linear", "out_proj", r"re:.*\.linear_attn\.out_proj$"),
    ("attn-linear", "q_proj", r"re:.*\.self_attn\.q_proj$"),
    ("attn-linear", "k_proj", r"re:.*\.self_attn\.k_proj$"),
    ("attn-linear", "v_proj", r"re:.*\.self_attn\.v_proj$"),
    ("attn-linear", "o_proj", r"re:.*\.self_attn\.o_proj$"),
)
_RUNTIME_FUSED_LINEAR_ROLES = (
    ("in_proj_qkv", "in_proj_z", "in_proj_qkvz"),
    ("in_proj_b", "in_proj_a", "in_proj_ba"),
)


# for reference, qwen3.5-4b ppl is 9.5375
@MODEL_REGISTRY.register(
    name="qwen3_5",
    task="llm",
    family="qwen",
    description="Qwen3.5 dense model adapter",
)
class Qwen3_5(BaseModel):
    def __init__(self, args):
        super().__init__(args)
        self.textconfig = Qwen3_5TextConfig
        self._update_config()
        self.num_layers = self.config.num_hidden_layers
        self.cls = Qwen3_5DecoderLayer
        self.model = self.empty_weights_model()
        self.linear_attention_mask = None
        self.full_attention_mask = None
        self.parse_quant_mode()

    def parse_quant_mode(self):
        if "moe" in self.quant_target:
            raise ValueError(
                "Qwen3.5 < 7B is a dense model and does not support quant_target='moe'."
            )

    @cached_property
    def _checkpoint_keys(self):
        """All tensor names in the checkpoint (header/index read, cached)."""
        model_dir = Path(self.model_path)
        index_path = model_dir / "model.safetensors.index.json"
        if index_path.exists():
            with open(index_path, encoding="utf-8") as f:
                return frozenset(json.load(f)["weight_map"])
        with safe_open(str(model_dir / "model.safetensors"), framework="pt") as f:
            return frozenset(f.keys())

    def get_scale_name(self, weight_name):
        """Return (scale_prefix, scale_inv_name) for locating scale tensors during dequantization."""
        scale_prefix = "_scale"
        base_name = weight_name
        if base_name.endswith("_scale_2"):
            base_name = base_name[: -len("_scale_2")]
        elif base_name.endswith("_scale"):
            base_name = base_name[: -len("_scale")]
        if f"{base_name}_scale_2" in self._checkpoint_keys:
            return scale_prefix, (f"{base_name}_scale", f"{base_name}_scale_2")
        return scale_prefix, f"{base_name}_scale"

    def generate_tensorwise_quant_layers(self):
        """Pure dequant (NVFP4->bf16 / HiF4->fp32) re-quantizes nothing."""
        return {}

    def generate_tensorwise_ignore_layers(self):
        return []

    def float_model(self):
        return super().float_model()

    def empty_weights_model(self):
        return super().empty_weights_model()

    def get_layer_weight_prefix(self, layer_idx: int) -> str:
        return f"model.language_model.layers.{layer_idx}."

    def load_layer_weight(self, prefix):
        return super().load_layer_weight(prefix)

    def load_embed_state_dict(self):
        return super().load_embed_state_dict()

    def block(self, layer_idx):
        return super().block(layer_idx)

    def do_embedding_forward(self, samples, dtype=torch.bfloat16, hook_name=None):
        outs = super().do_embedding_forward(samples, dtype=dtype, hook_name=hook_name)
        self.linear_attention_mask = self.attention_mask
        if not outs:
            self.full_attention_mask = None
            return outs

        # Layer 0 uses linear attention, so its captured mask is not the causal
        # mask required by the model's later full-attention layers. Materialize
        # that second mask explicitly instead of reusing the layer-0 value.
        self.full_attention_mask = create_causal_mask(
            config=self.config,
            inputs_embeds=outs[0],
            attention_mask=self.linear_attention_mask,
            past_key_values=None,
            position_ids=self.position_ids,
            and_mask_function=causal_mask_function,
        )
        if hook_name is not None:
            save_ptq_kwargs(
                self.position_ids,
                self.position_embeddings,
                self.full_attention_mask,
                self.args.data_dir,
            )
        return outs

    def do_block_forward(
        self,
        layer_idx,
        samples,
        hook_name=None,
        use_quant_block=False,
        enable_quant=False,
    ):
        previous_mask = self.attention_mask
        layer_type = self.config.layer_types[layer_idx]
        self.attention_mask = (
            self.full_attention_mask
            if layer_type == "full_attention"
            else self.linear_attention_mask
        )
        try:
            return super().do_block_forward(
                layer_idx,
                samples,
                hook_name=hook_name,
                use_quant_block=use_quant_block,
                enable_quant=enable_quant,
            )
        finally:
            self.attention_mask = previous_mask

    def do_head_forward(self, inps):
        return super().do_head_forward(inps)

    def build_quant_block(self, layer_idx):
        decoder_layer = self.block(layer_idx)

        if "attn-linear" in self.quant_target or "attn-cache" in self.quant_target:
            self.apply_quant_attn(decoder_layer)
        if "mlp" in self.quant_target:
            self.apply_quant_moe_mlp(decoder_layer)
        return decoder_layer

    def apply_quant_attn(self, decoder_layer):
        layer_type = decoder_layer.layer_type
        if layer_type == "linear_attention":
            attn_cls = QuantQwen35LinearAttn
            quant_attn = getattr(decoder_layer, "linear_attn", None)
            quant_attn.config = self.config
        else:
            attn_cls = QuantQwen35Attn
        return apply_quant_to_attn(self.args, decoder_layer, attn_cls)

    def apply_quant_moe_mlp(self, decoder_layer):
        return apply_quant_to_moe_mlp(
            self.args,
            decoder_layer,
            cls=QuantQwen35MLP,
        )

    def iter_ptq_units(self, layer_idx, block):
        yield from super().iter_ptq_units(layer_idx, block)

    def iter_deploy_bindings(self, layer_idx, block):
        yield from super().iter_deploy_bindings(layer_idx, block)

    def bits_scheme(self):
        """Return non-overlapping compressed-tensors groups for every Linear role."""
        return self._group_bits_scheme(self._source_bits_by_role())

    def deploy_bits_scheme(self):
        """Return bit metadata after reconciling vLLM's fused Linear modules."""
        return self._group_bits_scheme(self._runtime_bits_by_role())

    def deploy_runtime_fallbacks(self):
        source_bits = self._source_bits_by_role()
        runtime_bits = self._runtime_bits_by_role()
        return {
            role: {
                "source_w_bits": source_bits[role][0],
                "runtime_w_bits": runtime_bits[role][0],
                "reason": f"vLLM fuses this projection into {fused_name}",
            }
            for first, second, fused_name in _RUNTIME_FUSED_LINEAR_ROLES
            for role in (first, second)
            if source_bits[role] != runtime_bits[role]
        }

    def prepare_deploy_payload(self, weight_key, module, payload):
        role = weight_key.removesuffix(".weight").rsplit(".", 1)[-1]
        fallback = self.deploy_runtime_fallbacks().get(role)
        if fallback is None:
            return payload
        if getattr(module, "w_bits", None) != 4:
            raise ValueError(
                f"Qwen3.5 deploy fallback for '{role}' expects a W4 source module."
            )
        qweight = payload.get("qweight")
        if not isinstance(qweight, torch.Tensor) or qweight.dtype != torch.int8:
            raise ValueError(
                f"Qwen3.5 deploy fallback for '{role}' requires an INT8 qweight container."
            )
        if qweight.numel() and (qweight.min() < -7 or qweight.max() > 7):
            raise ValueError(
                f"Qwen3.5 deploy fallback for '{role}' received values outside INT4 range."
            )
        payload = dict(payload)
        payload.pop("weight_scale_second", None)
        payload.pop("weight_offset_second", None)
        return payload

    def _source_bits_by_role(self):
        bit_policy = ensure_bit_policy(self.args)
        return {
            name: tuple(bit_policy.linear_bits(name=name, group=group))
            for group, name, _ in _LINEAR_ROLE_TARGETS
        }

    def _runtime_bits_by_role(self):
        bits_by_role = self._source_bits_by_role()
        for first, second, fused_name in _RUNTIME_FUSED_LINEAR_ROLES:
            first_bits = bits_by_role[first]
            second_bits = bits_by_role[second]
            if first_bits == second_bits:
                continue
            if first_bits[1] != second_bits[1]:
                raise ValueError(
                    f"Qwen3.5 fused projection '{fused_name}' requires equal "
                    f"activation bits, got {first_bits[1]} and {second_bits[1]}."
                )
            if {first_bits[0], second_bits[0]} != {4, 8}:
                raise ValueError(
                    f"Qwen3.5 fused projection '{fused_name}' cannot reconcile "
                    f"weight bits {first_bits[0]} and {second_bits[0]}."
                )
            runtime_bits = (8, first_bits[1])
            bits_by_role[first] = runtime_bits
            bits_by_role[second] = runtime_bits
        return bits_by_role

    @staticmethod
    def _group_bits_scheme(bits_by_role):
        grouped = {}
        for _, name, target in _LINEAR_ROLE_TARGETS:
            bits = bits_by_role[name]
            grouped.setdefault(bits, []).append(target)
        if len(grouped) == 1:
            (w_bits, a_bits), _ = next(iter(grouped.items()))
            return [{"targets": ["Linear"], "w_bits": w_bits, "a_bits": a_bits}]
        return [
            {"targets": targets, "w_bits": bits[0], "a_bits": bits[1]}
            for bits, targets in sorted(grouped.items())
        ]

    def load_unit_inputs(self, data_dir, unit: PtqUnit):
        return super().load_unit_inputs(data_dir, unit)

    def get_block_attention_mask(self, block):
        if getattr(block, "layer_type", None) == "full_attention":
            return self.full_attention_mask
        return self.linear_attention_mask

    def _update_config(self):
        self.config = self.textconfig(**self.config.text_config.to_dict())

    def _embed_base_prefix(self) -> str:
        return "model.language_model."
