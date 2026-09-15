# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------
"""Quantization wrappers for Kimi-K3 MLP, MoE and MLA modules."""

import copy

import torch
import torch.nn as nn

from amct_pytorch.algorithms.quant import AlgoBuildContext
from amct_pytorch.common.models.llm.common.ptq_params import PtqParamHandler
from amct_pytorch.quantization.bit_policy import ensure_bit_policy
from amct_pytorch.quantization.dtypes.mxfp_impl import quantize_elewise
from amct_pytorch.quantization.modules.quant_base import (
    ActivationQuantizer,
    build_algorithms_by_target,
)
from amct_pytorch.quantization.modules.quant_linear import QuantLinear


class KimiFpActivationQuantizer(ActivationQuantizer):
    """FP activation fake quantization with Kimi MLA cache granularities."""

    _GRANULARITIES = ("tensor", "token_head")

    def __init__(self, quant_args, bits, granularity):
        if granularity not in self._GRANULARITIES:
            raise ValueError(
                f"Unsupported Kimi FP granularity {granularity!r}; "
                f"expected one of {self._GRANULARITIES}."
            )
        fp_args = copy.copy(quant_args)
        fp_args.quant_dtype = "fp"
        super().__init__(fp_args, bits)
        self.granularity = granularity

    def fake_quant(self, x):
        if self.bits == 16 or self.granularity == "tensor":
            return self.quant_obj(x)
        if not x.is_floating_point():
            raise ValueError("FP quantization requires a floating-point tensor.")
        if x.numel() == 0:
            return x

        values = x.float()
        amax = values.detach().abs().amax(dim=-1, keepdim=True)
        fp32 = torch.finfo(torch.float32)
        scale = (amax / self.quant_obj.max_norm).clamp_min(
            fp32.tiny * fp32.eps
        )
        scale = torch.where(amax == 0, torch.ones_like(scale), scale)
        quantized = quantize_elewise(
            values / scale,
            self.quant_obj.min_exp,
            self.quant_obj.max_norm,
            self.quant_obj.shift_val,
        )
        return (quantized * scale).to(x.dtype)


class QuantKimiLinear(QuantLinear):
    """A QuantLinear that quantizes its own input activation.

    The MLA wrapper calls each selected projection directly.  This small
    specialization keeps the per-projection activation and optional structure
    transforms next to the weight quantizer they feed.
    """

    def __init__(
        self,
        quant_args,
        linear: nn.Linear,
        name: str,
        group: str = "attn-linear",
    ):
        ensure_bit_policy(quant_args)
        self.group = group
        bits = quant_args.bit_policy[group][name]
        super().__init__(quant_args, linear, w_bits=bits.w, name=name)
        self.input_quant = ActivationQuantizer(quant_args, bits.a)
        ctx = AlgoBuildContext(matrix_size=128, dim_size=linear.in_features)
        self.input_transform = build_algorithms_by_target(quant_args, "structure", ctx)

    def forward(self, hidden_states, structure_transform=None):
        if structure_transform is not None:
            raise ValueError("QuantKimiLinear owns its structure transform")
        if self.input_transform is not None:
            hidden_states = self.input_transform(hidden_states)
        hidden_states = self.input_quant(hidden_states)
        return super().forward(hidden_states, structure_transform=self.input_transform)


def quantize_kimi_kda_linears(quant_args, attention):
    """Replace only KDA q/k/v/output projections, preserving its forward."""
    for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
        linear = getattr(attention, name)
        if not isinstance(linear, QuantKimiLinear):
            setattr(
                attention,
                name,
                QuantKimiLinear(quant_args, linear, name=name),
            )
    return attention


class QuantKimiMLAAttention(nn.Module):
    """Absorbed Kimi MLA with selected linear and latent-cache fake quant.

    q_a_proj, q_b_proj, kv_a_proj_with_mqa and o_proj participate in
    ``attn-linear``. kv_b_proj and output gate remain floating point.
    ``attn-cache`` applies per-token-per-head FP quantization
    to joined Q, and per-tensor FP quantization to joined latent KV and the
    attention probabilities. Kimi-K3 MLA uses NoPE, so neither branch receives
    rotary position embeddings.
    """

    def __init__(self, quant_args, attention):
        super().__init__()
        self.quant_args = quant_args
        policy = ensure_bit_policy(quant_args)
        self.enable_attn_linear = "attn-linear" in quant_args.quant_target
        self.enable_attn_cache = "attn-cache" in quant_args.quant_target
        self.config = attention.config
        self.layer_idx = attention.layer_idx
        self.hidden_size = attention.hidden_size
        self.num_heads = attention.num_heads
        self.num_key_value_heads = attention.num_key_value_heads
        self.num_key_value_groups = attention.num_key_value_groups
        self.q_lora_rank = attention.q_lora_rank
        self.kv_lora_rank = attention.kv_lora_rank
        self.qk_nope_head_dim = attention.qk_nope_head_dim
        self.qk_rope_head_dim = attention.qk_rope_head_dim
        self.q_head_dim = attention.q_head_dim
        self.v_head_dim = attention.v_head_dim
        self.scaling = attention.scaling
        self.attention_dropout = attention.attention_dropout
        self.use_output_gate = attention.use_output_gate

        if self.q_lora_rank is not None:
            self.q_a_proj = self._linear(quant_args, attention.q_a_proj, "q_a_proj")
            self.q_a_layernorm = attention.q_a_layernorm
            self.q_b_proj = self._linear(quant_args, attention.q_b_proj, "q_b_proj")
        else:
            self.q_proj = attention.q_proj
        self.kv_a_proj_with_mqa = self._linear(
            quant_args, attention.kv_a_proj_with_mqa, "kv_a_proj_with_mqa",
        )
        self.kv_a_layernorm = attention.kv_a_layernorm
        self.kv_b_proj = attention.kv_b_proj
        self.o_proj = self._linear(
            quant_args, attention.o_proj, "o_proj",
        )
        if self.use_output_gate:
            self.g_proj = attention.g_proj

        self.q_cache_quantizer = (
            KimiFpActivationQuantizer(
                quant_args, policy.cache_bits("q"), "token_head",
            ) if self.enable_attn_cache else nn.Identity()
        )
        self.kv_cache_quantizer = (
            KimiFpActivationQuantizer(
                quant_args, policy.cache_bits("kv"), "tensor",
            ) if self.enable_attn_cache else nn.Identity()
        )
        self.p_cache_quantizer = (
            KimiFpActivationQuantizer(
                quant_args, policy.cache_bits("p"), "tensor",
            ) if self.enable_attn_cache else nn.Identity()
        )

    def _linear(self, quant_args, linear, name):
        if not self.enable_attn_linear:
            return linear
        return QuantKimiLinear(quant_args, linear, name)

    @staticmethod
    def _as_additive_mask(mask, scores):
        mask = mask.to(device=scores.device)
        if mask.dtype == torch.bool or not mask.is_floating_point():
            additive = torch.zeros_like(mask, dtype=scores.dtype)
            return additive.masked_fill(~mask.bool(), float("-inf"))
        return mask.to(dtype=scores.dtype)

    @staticmethod
    def _attention_mask(attention_mask, scores):
        if attention_mask is None:
            query_len, key_len = scores.shape[1], scores.shape[-1]
            diagonal = key_len - query_len + 1
            return torch.full(
                (query_len, key_len), float("-inf"),
                device=scores.device, dtype=scores.dtype,
            ).triu(diagonal=diagonal).unsqueeze(0).unsqueeze(2)
        if attention_mask.ndim == 4:
            if attention_mask.shape[1] == 1:
                mask = attention_mask[:, 0, :, :scores.shape[-1]].unsqueeze(2)
            else:
                mask = attention_mask[
                    :, :, :, :scores.shape[-1]
                ].permute(0, 2, 1, 3)
            return QuantKimiMLAAttention._as_additive_mask(mask, scores)
        if attention_mask.ndim == 3:
            mask = attention_mask[:, :, :scores.shape[-1]].unsqueeze(2)
            return QuantKimiMLAAttention._as_additive_mask(mask, scores)
        if attention_mask.ndim == 2:
            if attention_mask.shape[0] != scores.shape[0]:
                mask = attention_mask[
                    :scores.shape[1], :scores.shape[-1]
                ].unsqueeze(0).unsqueeze(2)
                return QuantKimiMLAAttention._as_additive_mask(mask, scores)
            mask = attention_mask[:, None, None, :scores.shape[-1]].to(
                device=scores.device,
            )
            return QuantKimiMLAAttention._as_additive_mask(mask, scores)
        raise ValueError(
            f"Unsupported Kimi MLA attention mask rank: {attention_mask.ndim}"
        )

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        cache_position=None,
        **kwargs,
    ):
        del kwargs
        batch_size, seq_length, _ = hidden_states.shape
        if self.q_lora_rank is not None:
            q_states = self.q_b_proj(
                self.q_a_layernorm(self.q_a_proj(hidden_states))
            )
        else:
            q_states = self.q_proj(hidden_states)
        q_states = q_states.view(
            batch_size, seq_length, self.num_heads, self.q_head_dim,
        )
        q_states = self.q_cache_quantizer(q_states)
        q_nope, q_rope = torch.split(
            q_states,
            [self.qk_nope_head_dim, self.qk_rope_head_dim],
            dim=-1,
        )

        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        kv_nope, kv_rope = torch.split(
            compressed_kv,
            [self.kv_lora_rank, self.qk_rope_head_dim],
            dim=-1,
        )
        kv_nope = self.kv_a_layernorm(kv_nope)
        kv_states = self.kv_cache_quantizer(
            torch.cat((kv_nope, kv_rope), dim=-1)
        )
        kv_nope, kv_rope = torch.split(
            kv_states,
            [self.kv_lora_rank, self.qk_rope_head_dim],
            dim=-1,
        )
        kv_rope = kv_rope.unsqueeze(2)

        if past_key_values is not None:
            cache_kwargs = {"cache_position": cache_position}
            kv_nope, kv_rope = past_key_values.update(
                kv_nope.unsqueeze(1),
                kv_rope.squeeze(2).unsqueeze(1),
                self.layer_idx,
                cache_kwargs,
            )
            kv_nope = kv_nope.squeeze(1)
            kv_rope = kv_rope.transpose(1, 2)

        kv_b_weight = self.kv_b_proj.weight.view(
            self.num_heads,
            self.qk_nope_head_dim + self.v_head_dim,
            self.kv_lora_rank,
        )
        k_b_weight = kv_b_weight[:, :self.qk_nope_head_dim]
        v_b_weight = kv_b_weight[:, -self.v_head_dim:]
        q_nope = torch.einsum("bshd,hdc->bshc", q_nope, k_b_weight)
        scores = (
            torch.einsum("bshc,btc->bsht", q_nope, kv_nope)
            + torch.einsum(
                "bshr,bthr->bsht", q_rope, kv_rope,
            )
        ) * self.scaling
        scores = scores + self._attention_mask(attention_mask, scores)
        probs = scores.softmax(dim=-1, dtype=torch.float32).to(hidden_states.dtype)
        probs = self.p_cache_quantizer(probs)
        attn_output = torch.einsum("bsht,btc->bshc", probs, kv_nope)
        attn_output = torch.einsum(
            "bshc,hdc->bshd", attn_output, v_b_weight,
        ).flatten(2)
        if self.use_output_gate:
            attn_output = attn_output * self.g_proj(hidden_states).sigmoid()
        return self.o_proj(attn_output)

    def export_ptq_params(self):
        params = PtqParamHandler.export_module(self)
        if params:
            return params
        return PtqParamHandler.export_trainable_module(self)

    def load_ptq_params(self, params):
        if isinstance(params, dict) and params and all(
            isinstance(value, dict) for value in params.values()
        ):
            PtqParamHandler.load_module(self, params)
            return
        PtqParamHandler.load_trainable_module(self, params)


class QuantKimiMLP(nn.Module):
    """Quant wrapper shared by dense MLPs and explicit routed experts.

    Dense Kimi MLPs expose gate/up/down as ``gate_proj/up_proj/down_proj``;
    routed experts expose the same operations as ``w1/w3/w2``.  Attribute
    names are retained so deploy bindings continue to match checkpoint keys.
    """

    def __init__(self, quant_args, mlp_module: nn.Module, group: str = "mlp"):
        super().__init__()
        self.quant_args = quant_args
        ensure_bit_policy(quant_args)
        self.group = group
        self.config = mlp_module.config
        self.act_fn = mlp_module.act_fn
        self.is_expert = hasattr(mlp_module, "w1")
        self.hidden_size = getattr(mlp_module, "hidden_size", None)
        self.intermediate_size = getattr(mlp_module, "intermediate_size", None)
        if self.is_expert:
            self.hidden_size = mlp_module.hidden_dim
            self.intermediate_size = mlp_module.ffn_dim

        self.input_transform = None
        self.hidden_transform = None
        self._init_structure_transforms()

        bits = quant_args.bit_policy[group]
        gate_bits = bits["gate_proj"]
        up_bits = bits["up_proj"]
        down_bits = bits["down_proj"]
        if self.is_expert:
            self.w1 = QuantLinear(
                quant_args, mlp_module.w1, w_bits=gate_bits.w, name="gate_proj",
            )
            self.w3 = QuantLinear(
                quant_args, mlp_module.w3, w_bits=up_bits.w, name="up_proj",
            )
            self.w2 = QuantLinear(
                quant_args, mlp_module.w2, w_bits=down_bits.w, name="down_proj",
            )
        else:
            self.gate_proj = QuantLinear(
                quant_args, mlp_module.gate_proj, w_bits=gate_bits.w, name="gate_proj",
            )
            self.up_proj = QuantLinear(
                quant_args, mlp_module.up_proj, w_bits=up_bits.w, name="up_proj",
            )
            self.down_proj = QuantLinear(
                quant_args, mlp_module.down_proj, w_bits=down_bits.w, name="down_proj",
            )
        self.input_quant = ActivationQuantizer(quant_args, gate_bits.a)
        self.hidden_quant = ActivationQuantizer(quant_args, down_bits.a)

    def _projections(self):
        if self.is_expert:
            return self.w1, self.w3, self.w2
        return self.gate_proj, self.up_proj, self.down_proj

    def forward(self, hidden_states):
        gate_proj, up_proj, down_proj = self._projections()
        if self.input_transform is not None:
            hidden_states = self.input_transform(hidden_states)
        hidden_states = self.input_quant(hidden_states)
        gate = gate_proj(hidden_states, structure_transform=self.input_transform)
        up = up_proj(hidden_states, structure_transform=self.input_transform)

        if getattr(self.config, "hidden_act", None) == "situ":
            hidden_states = self.act_fn(torch.cat([gate, up], dim=-1))
        else:
            hidden_states = self.act_fn(gate) * up
        if self.hidden_transform is not None:
            hidden_states = self.hidden_transform(hidden_states)
        hidden_states = self.hidden_quant(hidden_states)
        return down_proj(hidden_states, structure_transform=self.hidden_transform)

    def export_ptq_params(self):
        params = PtqParamHandler.export_module(self)
        if params:
            return params
        return PtqParamHandler.export_trainable_module(self)

    def load_ptq_params(self, params):
        if isinstance(params, dict) and params and all(
            isinstance(value, dict) for value in params.values()
        ):
            PtqParamHandler.load_module(self, params)
            return
        PtqParamHandler.load_trainable_module(self, params)

    def _init_structure_transforms(self):
        ctx = AlgoBuildContext(matrix_size=128, dim_size=self.hidden_size)
        self.input_transform = build_algorithms_by_target(self.quant_args, "structure", ctx)
        ctx = AlgoBuildContext(matrix_size=128, dim_size=self.intermediate_size)
        self.hidden_transform = build_algorithms_by_target(self.quant_args, "structure", ctx)
