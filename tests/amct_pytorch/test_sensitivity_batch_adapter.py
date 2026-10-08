# Copyright (c) 2026 gcw_Z57idahf.
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

"""Custom calibration batches work throughout sensitivity allocation."""

from dataclasses import dataclass
import unittest

import torch
from torch import nn

from amct_pytorch.pruning import prune
from amct_pytorch.pruning.report import PruneReport


class LayeredModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(4, 6), nn.ReLU(), nn.Linear(6, 4))
                for _ in range(2)
            ]
        )

    def forward(self, features):
        for layer in self.layers:
            features = layer(features)
        return features


@dataclass
class CalibrationBatch:
    features: torch.Tensor


def config(allocation=True):
    result = {
        "methods": {"dense": {"name": "low_variance", "kwargs": {"prune_ratio": 0.5}}},
        "min_neurons": 1,
    }
    if allocation:
        result["allocation"] = {"strategy": "sensitivity", "guard": "none"}
    return result


class TestSensitivityBatchAdapter(unittest.TestCase):
    def check_pruning(self, data, adapter=None, allocation=True):
        torch.manual_seed(17)
        model = LayeredModel().eval()
        report = PruneReport()
        prune(
            model, config(allocation), data=data, batch_adapter=adapter, report=report
        )
        self.assertLess(report.params_after, report.params_before)
        self.assertTrue(all(layer[0].out_features < 6 for layer in model.layers))
        self.assertTrue(
            all(layer[0].out_features == layer[2].in_features for layer in model.layers)
        )
        self.assertFalse(
            any(
                "sensitivity probe did not complete" in message
                for message in report.warnings
            )
        )
        with torch.no_grad():
            output = model(torch.arange(24, dtype=torch.float32).reshape(6, 4) / 10)
        self.assertEqual(tuple(output.shape), (6, 4))
        self.assertTrue(torch.isfinite(output).all())
        if allocation:
            self.assertEqual(report.allocation_choice, "sensitivity")

    def test_custom_object_batches(self):
        values = torch.arange(24, dtype=torch.float32).reshape(6, 4) / 10
        self.check_pruning(
            [CalibrationBatch(values)], lambda batch: ((batch.features,), {})
        )

    def test_dictionary_metadata_is_removed_by_adapter(self):
        values = torch.arange(24, dtype=torch.float32).reshape(6, 4) / 10
        self.check_pruning(
            [{"payload": values, "source": "calibration"}],
            lambda batch: ((), {"features": batch["payload"]}),
        )

    def test_adapter_is_used_for_every_calibration_batch(self):
        values = torch.arange(24, dtype=torch.float32).reshape(6, 4) / 10
        calls = []

        def adapter(batch):
            calls.append(batch)
            return ((batch.features,), {})

        batches = [CalibrationBatch(values[:3]), CalibrationBatch(values[3:])]
        self.check_pruning(batches, adapter)
        self.assertGreaterEqual(len(calls), 6)
        self.assertTrue(all(any(item is batch for item in calls) for batch in batches))

    def test_plain_tensor_batches_remain_supported(self):
        self.check_pruning([torch.arange(24, dtype=torch.float32).reshape(6, 4) / 10])

    def test_default_dictionary_batches_remain_supported(self):
        self.check_pruning(
            [{"features": torch.arange(24, dtype=torch.float32).reshape(6, 4) / 10}]
        )

    def test_custom_object_already_works_without_allocation(self):
        values = torch.arange(24, dtype=torch.float32).reshape(6, 4) / 10
        self.check_pruning(
            [CalibrationBatch(values)],
            lambda batch: ((batch.features,), {}),
            allocation=False,
        )


if __name__ == "__main__":
    unittest.main()
