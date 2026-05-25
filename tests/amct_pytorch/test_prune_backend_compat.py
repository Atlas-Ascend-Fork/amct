from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from mock_torch_npu import *  # noqa: F401,F403

import torch
import torch.nn as nn

from amct_pytorch.pruning import AutoPruner, MethodSpec, PruneConfig
from amct_pytorch.pruning.compat import detect_backend


class FakeHFMLP(nn.Module):
    __module__ = "transformers.models.fake.modeling_fake"

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(16, 32)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(32, 16)
        self.config = type("Cfg", (), {"intermediate_size": 32})()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))

    def save_pretrained(self, save_dir: str) -> None:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), save_path / "pytorch_model.bin")
        with open(save_path / "config.json", "w", encoding="utf-8") as f:
            json.dump({"intermediate_size": self.config.intermediate_size}, f)


class FakeModelScopeMoE(nn.Module):
    __module__ = "modelscope.models.fake"

    def __init__(self) -> None:
        super().__init__()
        self.num_experts = 4
        self.top_k = 2
        self.gate = nn.Linear(8, 4, bias=False)
        self.experts = nn.ModuleList([nn.Linear(8, 8) for _ in range(4)])
        self.config = type("Cfg", (), {"num_experts": 4})()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.gate(x)
        probs = torch.softmax(logits, dim=-1)
        idx = probs.argmax(dim=-1)
        out = torch.zeros_like(x)
        for eid, expert in enumerate(self.experts):
            mask = idx == eid
            if mask.any():
                out[mask] = expert(x[mask])
        return out

    def save_pretrained(self, save_dir: str) -> None:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), save_path / "modelscope_model.bin")
        with open(save_path / "configuration.json", "w", encoding="utf-8") as f:
            json.dump({"num_experts": self.config.num_experts}, f)


class BackendCompatTest(unittest.TestCase):
    def test_detect_backend_hf(self) -> None:
        model = FakeHFMLP()
        backend = detect_backend(model)
        self.assertEqual(backend.name, "huggingface")
        self.assertTrue(backend.has_save_pretrained)

    def test_detect_backend_modelscope(self) -> None:
        model = FakeModelScopeMoE()
        backend = detect_backend(model)
        self.assertEqual(backend.name, "modelscope")
        self.assertTrue(backend.has_save_pretrained)

    def test_save_pretrained_is_used(self) -> None:
        torch.manual_seed(0)
        model = FakeHFMLP()
        dense_epoch_data = [torch.randn(4, 16) for _ in range(6)]
        pruner = AutoPruner(
            PruneConfig(
                methods={
                    "cnn": MethodSpec("tp_variance_channel", {"prune_ratio": 0.0}),
                    "dense": MethodSpec("low_variance", {"prune_ratio": 0.5}),
                    "moe": MethodSpec("activation_count", {"prune_ratio": 0.0}),
                }
            )
        )
        pruned = pruner(model, data=dense_epoch_data)
        with tempfile.TemporaryDirectory() as tmpdir:
            out = pruner.save(pruned, tmpdir)
            self.assertTrue((Path(out) / "pytorch_model.bin").exists())
            self.assertTrue((Path(out) / "config.json").exists())
            self.assertTrue((Path(out) / "ascend_prune_meta.json").exists())


if __name__ == "__main__":
    unittest.main()
