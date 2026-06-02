from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ToyConfig:
    def __init__(self, hidden_size: int, intermediate_size: int, vocab_size: int) -> None:
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.vocab_size = vocab_size


class TinyCNN(nn.Module):
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(32, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


class TinyTransformerMLP(nn.Module):
    def __init__(self, hidden_size: int = 32, intermediate_size: int = 64) -> None:
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(intermediate_size, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class TinyTransformerModel(nn.Module):
    def __init__(self, hidden_size: int = 32, intermediate_size: int = 64, vocab_size: int = 50) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.mlp = TinyTransformerMLP(hidden_size=hidden_size, intermediate_size=intermediate_size)
        self.norm = nn.LayerNorm(hidden_size)
        self.head = nn.Linear(hidden_size, vocab_size)
        self.config = ToyConfig(hidden_size=hidden_size, intermediate_size=intermediate_size, vocab_size=vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(input_ids)
        x = self.mlp(x)
        x = self.norm(x)
        return self.head(x)


class TinyExpert(nn.Module):
    def __init__(self, hidden_size: int = 16, expert_hidden: int = 32) -> None:
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, expert_hidden)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(expert_hidden, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class TinyMoE(nn.Module):
    def __init__(self, hidden_size: int = 16, expert_hidden: int = 32, num_experts: int = 4, top_k: int = 2) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)
        self.experts = nn.ModuleList([TinyExpert(hidden_size, expert_hidden) for _ in range(num_experts)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        router_logits = self.gate(x)
        probs = torch.softmax(router_logits, dim=-1)
        topk_weights, topk_indices = torch.topk(probs, k=min(self.top_k, len(self.experts)), dim=-1)
        outputs = torch.zeros_like(x)
        for rank in range(topk_indices.size(-1)):
            idx = topk_indices[..., rank]
            weight = topk_weights[..., rank].unsqueeze(-1)
            expert_out = torch.zeros_like(x)
            for expert_id, expert in enumerate(self.experts):
                mask = idx == expert_id
                if mask.any():
                    expert_out[mask] = expert(x[mask])
            outputs = outputs + weight * expert_out
        return outputs
