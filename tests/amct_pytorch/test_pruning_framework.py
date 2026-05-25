from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Any, List

from mock_torch_npu import *  # noqa: F401,F403

import torch
import torch.nn as nn

from amct_pytorch.pruning import AutoPruner, BasePruningMethod, MethodSpec, PruneConfig, create_default_registry
from amct_pytorch.pruning.report import PruneReport
from amct_pytorch.pruning.toy_models import TinyCNN, TinyMoE, TinyTransformerModel
from amct_pytorch.pruning.utils import count_parameters


class NoOpCNNMethod(BasePruningMethod):
    domain = "cnn"
    name = "noop_custom"
    requires_data = False

    def apply(self, model, domain, targets, context, report, config, spec) -> None:  # type: ignore[override]
        _ = model
        _ = domain
        _ = targets
        _ = context
        _ = config
        _ = spec
        report.add("cnn", self.name, "<none>", "custom plugin executed")


class PruningFrameworkTest(unittest.TestCase):
    def test_cnn_tp_variance_reduces_parameters_and_preserves_shape(self) -> None:
        torch.manual_seed(0)
        model = TinyCNN(num_classes=10)
        x = torch.randn(2, 3, 8, 8)
        cnn_cal_data = [torch.randn(2, 3, 8, 8) for _ in range(8)]
        y = model(x)

        pruner = AutoPruner(
            PruneConfig(
                methods={
                    "cnn": MethodSpec("tp_variance_channel", {"prune_ratio": 0.30}),
                    "dense": MethodSpec("low_variance", {"prune_ratio": 0.0}),
                    "moe": MethodSpec("activation_count", {"prune_ratio": 0.0}),
                }
            )
        )
        pruned = pruner(model, data=cnn_cal_data)
        y2 = pruned(x)

        self.assertEqual(tuple(y.shape), tuple(y2.shape))
        self.assertLess(count_parameters(pruned), count_parameters(model))

    def test_dense_low_variance_plugin_reduces_mlp_hidden(self) -> None:
        torch.manual_seed(0)
        model = TinyTransformerModel(hidden_size=32, intermediate_size=64, vocab_size=50)
        input_ids = torch.randint(0, 50, (2, 5))
        dense_epoch_data = [torch.randint(0, 50, (2, 5)) for _ in range(8)]
        y = model(input_ids)

        pruner = AutoPruner(
            PruneConfig(
                methods={
                    "cnn": MethodSpec("tp_variance_channel", {"prune_ratio": 0.0}),
                    "dense": MethodSpec("low_variance", {"prune_ratio": 0.50}),
                    "moe": MethodSpec("activation_count", {"prune_ratio": 0.0}),
                }
            )
        )
        pruned = pruner(model, data=dense_epoch_data)
        y2 = pruned(input_ids)

        self.assertEqual(tuple(y.shape), tuple(y2.shape))
        self.assertLess(count_parameters(pruned), count_parameters(model))
        self.assertEqual(pruned.mlp.fc1.out_features, 32)
        self.assertEqual(pruned.mlp.fc2.in_features, 32)
        self.assertEqual(getattr(pruned.config, "intermediate_size"), 32)

    def test_moe_mass_reduces_experts(self) -> None:
        torch.manual_seed(0)
        model = TinyMoE(hidden_size=16, expert_hidden=32, num_experts=4, top_k=2)
        x = torch.randn(8, 16)
        moe_epoch_data = [torch.randn(4, 16) for _ in range(10)]
        y = model(x)

        pruner = AutoPruner(
            PruneConfig(
                methods={
                    "cnn": MethodSpec("tp_variance_channel", {"prune_ratio": 0.0}),
                    "dense": MethodSpec("low_variance", {"prune_ratio": 0.0}),
                    "moe": MethodSpec("activation_count", {"prune_ratio": 0.50}),
                }
            )
        )
        pruned = pruner(model, data=moe_epoch_data)
        y2 = pruned(x)

        self.assertEqual(tuple(y.shape), tuple(y2.shape))
        self.assertLess(count_parameters(pruned), count_parameters(model))
        self.assertEqual(len(pruned.experts), 2)
        self.assertEqual(pruned.gate.out_features, 2)
        self.assertEqual(pruned.top_k, 2)

    def test_moe_mass_variance_reduces_experts(self) -> None:
        torch.manual_seed(0)
        model = TinyMoE(hidden_size=16, expert_hidden=32, num_experts=4, top_k=2)
        x = torch.randn(8, 16)
        moe_epoch_data = [torch.randn(4, 16) for _ in range(10)]
        y = model(x)

        pruner = AutoPruner(
            PruneConfig(
                methods={
                    "cnn": MethodSpec("tp_variance_channel", {"prune_ratio": 0.0}),
                    "dense": MethodSpec("low_variance", {"prune_ratio": 0.0}),
                    "moe": MethodSpec("mass_variance", {"prune_ratio": 0.50, "top_k": 2, "boundary": 0}),
                }
            )
        )
        pruned = pruner(model, data=moe_epoch_data)
        y2 = pruned(x)

        self.assertEqual(tuple(y.shape), tuple(y2.shape))
        self.assertLess(count_parameters(pruned), count_parameters(model))
        self.assertEqual(len(pruned.experts), 2)
        self.assertEqual(pruned.gate.out_features, 2)
        self.assertEqual(pruned.top_k, 2)

    def test_custom_plugin_registration(self) -> None:
        model = TinyCNN(num_classes=10)
        registry = create_default_registry()
        registry.register_method(NoOpCNNMethod())
        pruner = AutoPruner(
            PruneConfig(
                methods={
                    "cnn": MethodSpec("noop_custom", {}),
                    "dense": MethodSpec("low_variance", {"prune_ratio": 0.0}),
                    "moe": MethodSpec("activation_count", {"prune_ratio": 0.0}),
                }
            ),
            registry=registry,
        )
        pruned = pruner(model)
        self.assertEqual(count_parameters(pruned), count_parameters(model))
        self.assertIsNotNone(pruner.last_report)
        assert pruner.last_report is not None
        self.assertTrue(any(event.method == "noop_custom" for event in pruner.last_report.events))

    def test_save_outputs_exist(self) -> None:
        torch.manual_seed(0)
        model = TinyCNN()
        pruner = AutoPruner(
            PruneConfig(
                methods={
                    "cnn": MethodSpec("tp_variance_channel", {"prune_ratio": 0.10, "seed": 11}),
                    "dense": MethodSpec("low_variance", {"prune_ratio": 0.0}),
                    "moe": MethodSpec("activation_count", {"prune_ratio": 0.0}),
                }
            )
        )
        pruned = pruner(model)
        with tempfile.TemporaryDirectory() as tmpdir:
            out = pruner.save(pruned, tmpdir)
            self.assertTrue((Path(out) / "model.pt").exists())
            self.assertTrue((Path(out) / "model_full.pt").exists())
            self.assertTrue((Path(out) / "ascend_prune_meta.json").exists())


if __name__ == "__main__":
    unittest.main()
