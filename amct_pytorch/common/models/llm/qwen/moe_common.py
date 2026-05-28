# Copyright 2025 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
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
import re

import torch
import torch.nn as nn

from amct_pytorch.common.models.llm.common.moe_unpack import ExpertLinearView, GatedExpertView, find_moe_module
from amct_pytorch.common.models.llm.common.quant_apply import QuantGatedMLP


class QuantGatedExperts(nn.Module):
    def __init__(
        self,
        args,
        experts_module,
        view_cls=GatedExpertView,
        group: str = "moe.routed",
    ):
        super().__init__()
        self.args = args
        self.packed_experts = experts_module
        self.num_experts = experts_module.num_experts
        self.view_cls = view_cls
        self.group = group
        self.expert_modules = nn.ModuleList(
            [
                QuantGatedMLP(
    args,
    self.view_cls(
        self.packed_experts,
        expert_idx,
        hidden_attr="hidden_dim",
        intermediate_attr="intermediate_dim",
        materialize=False),
         group=group)
                for expert_idx in range(self.num_experts)
            ]
        )

    def build_ptq_expert_module(self, expert_idx: int):
        return QuantGatedMLP(self.args, self.view_cls(self.packed_experts, expert_idx, hidden_attr="hidden_dim",
                             intermediate_attr="intermediate_dim", materialize=True), group=self.group)

    def iter_ptq_expert_modules(self):
        for expert_idx in range(self.num_experts):
            yield self.build_ptq_expert_module(expert_idx)

    def forward(self, hidden_states: torch.Tensor, top_k_index: torch.Tensor,
                top_k_weights: torch.Tensor) -> torch.Tensor:
        final_hidden_states = torch.zeros_like(hidden_states)
        if top_k_index.numel() == 0:
            return final_hidden_states

        expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts).permute(2, 1, 0)
        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero(as_tuple=False)

        for expert_idx_tensor in expert_hit:
            expert_idx = int(expert_idx_tensor.reshape(-1)[0].item())
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]
            current_hidden_states = self.expert_modules[expert_idx](current_state)
            current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))

        return final_hidden_states


def pack_gated_expert_weights(state_dict, expert_prefix: str = "mlp.experts"):
    gate_proj = {}
    up_proj = {}
    down_proj = {}
    packed_state = {}
    pattern = re.compile(
        rf"{re.escape(expert_prefix)}\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$"
    )

    for key, tensor in state_dict.items():
        match = pattern.match(key)
        if match is None:
            packed_state[key] = tensor
            continue
        expert_idx = int(match.group(1))
        proj_name = match.group(2)
        if proj_name == "gate_proj":
            gate_proj[expert_idx] = tensor
        elif proj_name == "up_proj":
            up_proj[expert_idx] = tensor
        else:
            down_proj[expert_idx] = tensor

    if not gate_proj and not up_proj and not down_proj:
        return state_dict

    expert_indices = sorted(set(gate_proj) | set(up_proj) | set(down_proj))
    if set(gate_proj) != set(up_proj) or set(gate_proj) != set(down_proj):
        raise KeyError(f"Inconsistent expert weights while packing {expert_prefix}.")

    packed_state[f"{expert_prefix}.gate_up_proj"] = torch.stack(
        [torch.cat([gate_proj[idx], up_proj[idx]], dim=0) for idx in expert_indices],
        dim=0,
    )
    packed_state[f"{expert_prefix}.down_proj"] = torch.stack(
        [down_proj[idx] for idx in expert_indices],
        dim=0,
    )
    return packed_state


def is_packed_experts(experts):
    return hasattr(experts, "gate_up_proj") and hasattr(experts, "down_proj")