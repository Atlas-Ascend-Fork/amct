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
"""Custom calibration batches work throughout sensitivity allocation."""

from dataclasses import dataclass
import copy
import logging
import unittest
from unittest import mock

import torch
from torch import nn

from amct_pytorch.pruning import prune
from amct_pytorch.pruning import allocation as allocation_module
from amct_pytorch.pruning.config import PruneConfig
from amct_pytorch.pruning.report import PruneReport
from amct_pytorch.pruning.calib import calib_nll


class _RecordingHandler(logging.Handler):
    def __init__(self, sink):
        super().__init__(level=logging.WARNING)
        self._sink = sink

    def emit(self, record):
        self._sink.append(record.getMessage())


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
        messages = []
        handler = _RecordingHandler(messages)
        logger = logging.getLogger("Log")
        logger.addHandler(handler)
        try:
            prune(
                model,
                config(allocation),
                data=data,
                batch_adapter=adapter,
                report=report,
            )
        finally:
            logger.removeHandler(handler)
        self.assertLess(report.params_after, report.params_before)
        self.assertTrue(all(layer[0].out_features < 6 for layer in model.layers))
        self.assertTrue(
            all(layer[0].out_features == layer[2].in_features for layer in model.layers)
        )
        self.assertFalse(
            any("sensitivity probe did not complete" in message for message in messages)
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

    def test_sensitivity_probes_forward_adapter_to_each_pruned_layer(self):
        torch.manual_seed(17)
        model = LayeredModel().eval()
        values = torch.arange(24, dtype=torch.float32).reshape(6, 4) / 10
        batches = [CalibrationBatch(values)]

        def adapter(batch):
            return ((batch.features,), {})

        with mock.patch.object(
            allocation_module,
            "_prune_single_layer",
            wraps=allocation_module._prune_single_layer,
        ) as prune_layer:
            allocation_module.measure_layer_sensitivity(
                model, batches, PruneConfig(**config(False)), batch_adapter=adapter
            )
        self.assertEqual(prune_layer.call_count, len(model.layers))
        self.assertEqual(
            [call.args[3] for call in prune_layer.call_args_list],
            ["layers.0", "layers.1"],
        )
        self.assertTrue(
            all(call.args[6] is adapter for call in prune_layer.call_args_list)
        )

    def test_probe_failure_warning_is_observed_from_real_logger(self):
        model = LayeredModel().eval()
        values = torch.arange(24, dtype=torch.float32).reshape(6, 4) / 10
        with (
            mock.patch.object(
                allocation_module,
                "_prune_single_layer",
                side_effect=RuntimeError("controlled probe failure"),
            ),
            self.assertLogs("Log", level="WARNING") as captured,
        ):
            sensitivity = allocation_module.measure_layer_sensitivity(
                model, [values], PruneConfig(**config(False))
            )
        self.assertEqual(sensitivity, {"layers.0": 1.0, "layers.1": 1.0})
        self.assertEqual(
            sum(
                "sensitivity probe did not complete" in record
                for record in captured.output
            ),
            len(model.layers),
        )


class TokenModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embeddings = nn.Embedding(17, 4)
        self.layers = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(4, 6), nn.GELU(), nn.Linear(6, 4))
                for _ in range(2)
            ]
        )
        self.classifier = nn.Linear(4, 17)

    def forward(self, input_ids, attention_mask=None):
        features = self.embeddings(input_ids)
        for layer in self.layers:
            features = layer(features)
        if attention_mask is not None:
            features = features * attention_mask.unsqueeze(-1)
        return self.classifier(features)


class TestDefaultGuardBatchAdapter(unittest.TestCase):
    def test_default_guard_matches_plain_token_batches(self):
        for keyword in (False, True):
            with self.subTest(keyword=keyword):
                torch.manual_seed(23)
                reference = TokenModel().eval()
                adapted = copy.deepcopy(reference)
                ids = torch.arange(12).reshape(2, 6)
                cfg = config()
                del cfg["allocation"]["guard"]
                direct_report, adapted_report = PruneReport(), PruneReport()
                prune(reference, cfg, data=[ids], report=direct_report)
                calls = []

                def adapter(batch):
                    calls.append(batch)
                    return (
                        ((), {"input_ids": batch.features})
                        if keyword
                        else ((batch.features,), {})
                    )

                messages = []
                handler = _RecordingHandler(messages)
                logger = logging.getLogger("Log")
                logger.addHandler(handler)
                try:
                    prune(
                        adapted,
                        cfg,
                        data=[CalibrationBatch(ids)],
                        batch_adapter=adapter,
                        report=adapted_report,
                    )
                finally:
                    logger.removeHandler(handler)
                self.assertFalse(
                    any(
                        "calib NLL guard unavailable" in message for message in messages
                    )
                )
                self.assertEqual(
                    adapted_report.allocation_choice, direct_report.allocation_choice
                )
                self.assertEqual(
                    adapted_report.params_after, direct_report.params_after
                )
                self.assertLess(
                    adapted_report.params_after, adapted_report.params_before
                )
                self.assertGreater(len(calls), 2)
                for key, value in reference.state_dict().items():
                    torch.testing.assert_close(adapted.state_dict()[key], value)
                torch.testing.assert_close(adapted(ids), reference(ids))

    def test_calib_nll_adapter_preserves_keyword_inputs(self):
        torch.manual_seed(23)
        model = TokenModel().eval()
        ids = torch.arange(12).reshape(2, 6)
        expected = calib_nll(model, [ids])
        value = calib_nll(
            model,
            [{"tokens": ids, "source": "calibration"}],
            batch_adapter=lambda batch: (
                (),
                {
                    "input_ids": batch["tokens"],
                    "attention_mask": torch.ones_like(batch["tokens"]),
                },
            ),
        )
        self.assertAlmostEqual(value, expected)

    def test_calib_nll_rejects_non_token_adapter_output(self):
        model = TokenModel().eval()
        with self.assertRaisesRegex(ValueError, "token-id"):
            calib_nll(
                model,
                [CalibrationBatch(torch.ones(2, 6))],
                batch_adapter=lambda batch: ((batch.features,), {}),
            )


if __name__ == "__main__":
    unittest.main()
