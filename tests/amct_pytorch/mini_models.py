#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
"""
HuggingFace-style mini model definitions
- CNN, MoE, MLP models
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MiniMLPConfig:
    def __init__(
        self,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=4,
        num_attention_heads=4,
        vocab_size=1000,
        max_position_embeddings=128,
        hidden_dropout_prob=0.1,
    ):
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_dropout_prob = hidden_dropout_prob


class MiniMLPLayer(nn.Module):
    """Dense MLP Layer (BERT FFN style)"""

    def __init__(self, config):
        super().__init__()
        self.dense1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.dense2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states):
        hidden_states = self.dense1(hidden_states)
        hidden_states = F.gelu(hidden_states)
        hidden_states = self.dense2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class MiniMLPModel(nn.Module):
    """Mini MLP Model for Testing"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [MiniMLPLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.classifier = nn.Linear(config.hidden_size, config.vocab_size)

    def forward(self, input_ids):
        hidden_states = self.embeddings(input_ids)
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        logits = self.classifier(hidden_states)
        return logits


class MiniCNNConfig:
    def __init__(
        self,
        num_channels=None,
        num_classes=10,
        input_channels=3,
    ):
        self.num_channels = num_channels if num_channels is not None else [64, 128, 256]
        self.num_classes = num_classes
        self.input_channels = input_channels


class MiniCNNBlock(nn.Module):
    """Basic CNN Block with BN"""

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class MiniCNNModel(nn.Module):
    """Mini CNN Model for Testing - Fixed for pruning compatibility"""

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.conv1 = nn.Conv2d(
            config.input_channels, 64, kernel_size=3, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(256)

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, config.num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)

        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class MiniMoEConfig:
    def __init__(
        self,
        hidden_size=256,
        intermediate_size=512,
        num_experts=8,
        num_experts_per_tok=2,
        num_hidden_layers=4,
        vocab_size=1000,
    ):
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.num_hidden_layers = num_hidden_layers
        self.vocab_size = vocab_size


class MiniExpert(nn.Module):
    """Single Expert (FFN)"""

    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class MiniMoELayer(nn.Module):
    """MoE Layer with Top-K Routing"""

    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.hidden_size = config.hidden_size

        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)

        self.experts = nn.ModuleList(
            [MiniExpert(config) for _ in range(config.num_experts)]
        )

    def forward(self, hidden_states):
        batch_size, seq_len, hidden_dim = hidden_states.shape
        hidden_states_flat = hidden_states.view(-1, hidden_dim)

        router_logits = self.gate(hidden_states_flat)
        router_probs = F.softmax(router_logits, dim=-1)

        topk_weights, topk_indices = torch.topk(
            router_probs, self.num_experts_per_tok, dim=-1
        )
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

        final_output = torch.zeros_like(hidden_states_flat)
        for i in range(self.num_experts_per_tok):
            expert_idx = topk_indices[:, i]
            i_end = i + 1
            expert_weight = topk_weights[:, i:i_end]

            for expert_id in range(self.num_experts):
                mask = expert_idx == expert_id
                if mask.any():
                    expert_input = hidden_states_flat[mask]
                    expert_output = self.experts[expert_id](expert_input)
                    final_output[mask] += expert_weight[mask] * expert_output

        return final_output.view(batch_size, seq_len, hidden_dim)


class MiniMoEModel(nn.Module):
    """Mini MoE Model for Testing"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [MiniMoELayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.classifier = nn.Linear(config.hidden_size, config.vocab_size)

    def forward(self, input_ids):
        hidden_states = self.embeddings(input_ids)
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        logits = self.classifier(hidden_states)
        return logits


def create_mini_mlp():
    """Create a mini MLP model"""
    config = MiniMLPConfig()
    return MiniMLPModel(config), config


def create_mini_cnn():
    """Create a mini CNN model"""
    config = MiniCNNConfig()
    return MiniCNNModel(config), config


def create_mini_moe():
    """Create a mini MoE model"""
    config = MiniMoEConfig()
    return MiniMoEModel(config), config


if __name__ == "__main__":
    print("Testing Mini Models...")

    print("\n1. Mini MLP:")
    mlp, mlp_config = create_mini_mlp()
    print(f"   Params: {sum(p.numel() for p in mlp.parameters()):,}")
    test_input = torch.randint(0, 1000, (2, 10))
    output = mlp(test_input)
    print(f"   Output shape: {output.shape}")

    print("\n2. Mini CNN:")
    cnn, cnn_config = create_mini_cnn()
    print(f"   Params: {sum(p.numel() for p in cnn.parameters()):,}")
    test_input = torch.randn(2, 3, 32, 32)
    output = cnn(test_input)
    print(f"   Output shape: {output.shape}")

    print("\n3. Mini MoE:")
    moe, moe_config = create_mini_moe()
    print(f"   Params: {sum(p.numel() for p in moe.parameters()):,}")
    test_input = torch.randint(0, 1000, (2, 10))
    output = moe(test_input)
    print(f"   Output shape: {output.shape}")

    print("\nAll models created successfully!")
