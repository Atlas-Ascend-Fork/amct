# Copyright 2025 Meituan and the HuggingFace Inc. team. All rights reserved.
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
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
import torch
import torch.nn as nn

from amct_pytorch.common.models.llm.common.moe_unpack import find_moe_module, GatedExpertView
from amct_pytorch.common.models.llm.common.quant_apply import QuantGatedMLP


class QuantLongcatExperts(nn.Module):
    """
    LongCat packed-expert wrapper with PTQ-friendly expert materialization.

    Runtime path keeps expert views lightweight (materialize=False).
    PTQ path can iterate per-expert modules with materialize=True so expert
    weights are registered parameters and move with module.to(device).
    """

    def __init__(
        self,
        args,
        experts_module,
        view_cls=GatedExpertView,
        quant_mlp_cls=QuantGatedMLP,
    ):
        super().__init__()
        self.args = args
        self.packed_experts = experts_module
        self.num_routed_experts = experts_module.num_routed_experts
        self.total_experts = experts_module.total_experts
        self.view_cls = view_cls
        self.quant_mlp_cls = quant_mlp_cls

        expert_modules = [
            self.quant_mlp_cls(
    self.args,
    self.view_cls(
        self.packed_experts,
        expert_idx,
        hidden_attr="hidden_size",
        intermediate_attr="intermediate_size",
         materialize=False))
            for expert_idx in range(self.num_routed_experts)
        ]
        expert_modules.extend(nn.Identity() for _ in range(self.total_experts - self.num_routed_experts))
        self.expert_modules = nn.ModuleList(expert_modules)

    def build_ptq_expert_module(self, expert_idx: int):
        if expert_idx >= self.num_routed_experts:
            raise IndexError(f"expert_idx {expert_idx} out of routed expert range {self.num_routed_experts}.")
        return self.quant_mlp_cls(
            self.args,
            self.view_cls(
    self.packed_experts,
    expert_idx,
    hidden_attr="hidden_size",
    intermediate_attr="intermediate_size",
     materialize=True),
        )

    def iter_ptq_expert_modules(self):
        for expert_idx in range(self.num_routed_experts):
            yield self.build_ptq_expert_module(expert_idx)

    def forward(self, hidden_states: torch.Tensor, top_k_index: torch.Tensor,
                top_k_weights: torch.Tensor) -> torch.Tensor:
        final_hidden_states = torch.zeros_like(hidden_states)
        if top_k_index.numel() == 0:
            return final_hidden_states

        expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.total_experts).permute(2, 1, 0)
        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero(as_tuple=False)

        for expert_idx_tensor in expert_hit:
            expert_idx = int(expert_idx_tensor.reshape(-1)[0].item())
            selection_idx, token_idx = torch.where(expert_mask[expert_idx])
            if token_idx.numel() == 0:
                continue

            current_state = hidden_states[token_idx]
            current_hidden_states = self.expert_modules[expert_idx](current_state)
            current_hidden_states = current_hidden_states * top_k_weights[token_idx, selection_idx, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))

        return final_hidden_states
