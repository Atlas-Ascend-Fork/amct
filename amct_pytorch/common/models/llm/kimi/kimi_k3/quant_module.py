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
"""Quantization wrappers for Kimi-K3 linear projections."""

import torch
import torch.nn as nn

from amct_pytorch.algorithms.quant import AlgoBuildContext
from amct_pytorch.common.models.llm.common.ptq_params import PtqParamHandler
from amct_pytorch.quantization.bit_policy import ensure_bit_policy
from amct_pytorch.quantization.modules.quant_base import (
    ActivationQuantizer,
    build_algorithms_by_target,
)
from amct_pytorch.quantization.modules.quant_linear import QuantLinear


_KIMI_ATTN_LINEAR_NAMES = (
    # MLA projections.
    "q_a_proj",
    "q_b_proj",
    "q_proj",
    "kv_a_proj_with_mqa",
    "kv_b_proj",
    "g_proj",
    "o_proj",
    # KDA-only projections (shared q/g/o names are listed above).
    "k_proj",
    "v_proj",
    "f_a_proj",
    "f_b_proj",
    "b_proj",
    "g_a_proj",
    "g_b_proj",
)


class QuantKimiLinear(QuantLinear):
    """A QuantLinear that quantizes its own input activation.

    Kimi attention keeps its original forward method so the NPU KDA kernels,
    convolution states and cache updates remain untouched.  This small
    specialization lets those original calls keep using ``linear(x)`` while
    still applying the activation and optional structure transforms expected
    by the AMCT quantization path.
    """

    def __init__(self, quant_args, linear: nn.Linear, name: str):
        ensure_bit_policy(quant_args)
        bits = quant_args.bit_policy["attn-linear"][name]
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


def quantize_kimi_attention(quant_args, attention: nn.Module):
    """Replace only Kimi MLA/KDA projection linears in place.

    Keeping the original attention object deliberately preserves short-conv,
    KDA operator dispatch, recurrent state and cache behavior.
    """
    for name in _KIMI_ATTN_LINEAR_NAMES:
        linear = getattr(attention, name, None)
        if isinstance(linear, QuantLinear):
            continue
        if isinstance(linear, nn.Linear):
            setattr(attention, name, QuantKimiLinear(quant_args, linear, name))
    return attention

