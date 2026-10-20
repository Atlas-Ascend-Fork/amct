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
"""Regression tests for issue #245: a Mapping passed directly as ``data`` / ``eval_data``
must be consumed as ONE batch, while list / tuple / DataLoader / generator stay a
sequence of batches.

Before the fix every consumer iterated the mapping directly, so ``prune(model, cfg,
data={"input_ids": ids})`` walked the *keys* (``"input_ids"``) and raised a misleading
``TypeError`` from ``default_batch_adapter``, and a classification search with
``eval_data={"input_ids": ids, "labels": labels}`` was rejected as missing labels.
The tests below assert both the normalisation itself and the ``data=D`` / ``data=[D]``
equivalence across the fixed, tolerance, size-budget, menu, allocation, finetune and
diagnose entry points.
"""

import copy
import os
import sys
import unittest
import warnings
from collections import OrderedDict, UserDict
from collections.abc import Mapping

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(__file__))

from mini_models import create_mini_mlp
from amct_pytorch.pruning import (
    DENSE_LOWVAR_PRUNE_CFG,
    DENSE_RECOVERY_MENU_CFG,
    SENSITIVITY_ALLOC_PRUNE_CFG,
    PruneReport,
    prune,
    prune_diagnose,
    prune_finetune,
)
from amct_pytorch.pruning.accuracy_based_auto_prune import (
    _make_classification_quality,
    _prepare_search_inputs,
)
from amct_pytorch.pruning.allocation import measure_layer_sensitivity
from amct_pytorch.pruning.context import (
    PruneContext,
    as_batch_sequence,
    default_batch_adapter,
)

try:
    from transformers.tokenization_utils_base import BatchEncoding

    _HAS_TF = True
except Exception:  # pragma: no cover - transformers is an optional test dependency
    _HAS_TF = False


def _params(model):
    return sum(p.numel() for p in model.parameters())


def _fresh_mlp():
    """A freshly seeded, unpruned mini MLP -- used as the parameter-count baseline."""
    torch.manual_seed(0)
    model, _ = create_mini_mlp()
    model.eval()
    return model


def _token_batch(seed=0, n_batch=4, n_seq=20):
    """One deterministic integer token batch, wrapped in a mapping."""
    torch.manual_seed(seed)
    return {"input_ids": torch.randint(0, 1000, (n_batch, n_seq))}


class _PlainMapping(Mapping):
    """A Mapping that is *not* a dict subclass -- the shape UserDict/BatchEncoding have."""

    def __init__(self, payload):
        self._payload = dict(payload)

    def __getitem__(self, key):
        return self._payload[key]

    def __iter__(self):
        return iter(self._payload)

    def __len__(self):
        return len(self._payload)


def _mapping_variants(batch):
    """Same batch in every Mapping flavour a user can realistically hand over.

    ``UserDict`` is the base class of HF ``BatchEncoding``, i.e. what ``tokenizer(...)``
    returns, so it is the case issue #245 is actually about -- none of these are
    ``dict`` subclasses except the first two.
    """
    payload = dict(batch)
    variants = {
        "dict": dict(payload),
        "OrderedDict": OrderedDict(payload),
        "UserDict": UserDict(dict(payload)),
        "plain Mapping": _PlainMapping(payload),
    }
    if _HAS_TF:
        variants["BatchEncoding"] = BatchEncoding(dict(payload))
    return variants


class _ToyDenseClassifier(nn.Module):
    """Two-layer dense MLP emitting [batch, classes] logits -- a dense-domain pruning target."""

    def __init__(self, in_dim=16, hidden=32, classes=4):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(hidden, classes)

    def forward(self, input_ids, labels=None):
        _ = labels
        return self.fc2(self.act(self.fc1(input_ids.float())))


class TestAsBatchSequence(unittest.TestCase):
    """The shared normaliser: mapping -> one batch, None -> empty, iterable -> itself."""

    def test_mapping_is_a_single_batch(self):
        batch = {"input_ids": torch.zeros(2, 3)}
        out = list(as_batch_sequence(batch))
        self.assertEqual(len(out), 1)
        self.assertIs(out[0], batch)

    def test_ordered_dict_is_a_single_batch(self):
        batch = OrderedDict(input_ids=torch.zeros(2, 3))
        self.assertEqual(list(as_batch_sequence(batch)), [batch])

    def test_none_yields_no_batch(self):
        self.assertEqual(list(as_batch_sequence(None)), [])

    def test_list_and_tuple_stay_multi_batch(self):
        batches = [{"input_ids": torch.zeros(2, 3)} for _ in range(3)]
        self.assertEqual(len(list(as_batch_sequence(batches))), 3)
        self.assertEqual(len(list(as_batch_sequence(tuple(batches)))), 3)

    def test_generator_is_not_consumed_eagerly(self):
        def gen():
            yield {"input_ids": torch.zeros(2, 3)}
            yield {"input_ids": torch.ones(2, 3)}

        seq = as_batch_sequence(gen())
        self.assertEqual(next(iter(seq))["input_ids"].sum().item(), 0.0)
        self.assertEqual(next(iter(seq))["input_ids"].sum().item(), 6.0)

    def test_tensor_keeps_row_wise_iteration(self):
        data = torch.zeros(3, 4)
        self.assertEqual(len(list(as_batch_sequence(data))), 3)


class TestPruneContextMapping(unittest.TestCase):
    def test_mapping_data_yields_one_batch(self):
        batch = _token_batch()
        out = list(PruneContext(data=batch).iter_model_inputs())
        self.assertEqual(len(out), 1)
        args, kwargs = out[0]
        self.assertEqual(args, ())
        self.assertTrue(torch.equal(kwargs["input_ids"], batch["input_ids"]))

    def test_list_of_mapping_yields_every_batch(self):
        batches = [_token_batch(seed=i) for i in range(3)]
        out = list(PruneContext(data=batches).iter_model_inputs())
        self.assertEqual(len(out), 3)

    def test_none_data_yields_no_batch(self):
        self.assertEqual(list(PruneContext(data=None).iter_model_inputs()), [])


class TestFixedPruneMapping(unittest.TestCase):
    """``prune(model, cfg, data=D)`` must equal ``prune(model, cfg, data=[D])``."""

    def _run(self, data):
        torch.manual_seed(0)
        model, _ = create_mini_mlp()
        model.eval()
        report = PruneReport()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            prune(model, DENSE_LOWVAR_PRUNE_CFG, data=data, report=report)
        return _params(model), report.params_after

    def test_mapping_equals_single_element_list(self):
        batch = _token_batch()
        self.assertEqual(self._run(batch), self._run([copy.deepcopy(batch)]))

    def test_mapping_actually_prunes(self):
        torch.manual_seed(0)
        model, _ = create_mini_mlp()
        model.eval()
        before = _params(model)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            prune(model, DENSE_LOWVAR_PRUNE_CFG, data=_token_batch())
        self.assertLess(_params(model), before)


class TestSearchModeMapping(unittest.TestCase):
    """Tolerance / size-budget / menu searches must accept a mapping as one batch."""

    def test_prepare_search_inputs_mapping_eval_data(self):
        model, _ = create_mini_mlp()
        model.eval()
        batch = _token_batch()
        _, _, _, from_mapping = _prepare_search_inputs(
            model, DENSE_LOWVAR_PRUNE_CFG, True, [0.25], None, batch
        )
        _, _, _, from_list = _prepare_search_inputs(
            model, DENSE_LOWVAR_PRUNE_CFG, True, [0.25], None, [batch]
        )
        self.assertEqual(len(from_mapping), 1)
        self.assertEqual(len(from_list), 1)

    def test_classification_eval_data_mapping(self):
        """The issue's second repro: eval_data={"input_ids": ..., "labels": ...}."""
        torch.manual_seed(0)
        model = _ToyDenseClassifier().eval()
        feats = torch.randn(6, 16)
        with torch.no_grad():
            labels = model(feats).argmax(-1)
        eval_data = {"input_ids": feats, "labels": labels}
        q_mapping = _make_classification_quality(model, [eval_data], None)
        q_list = _make_classification_quality(model, [eval_data], None)
        # One mapping batch, measured against its own predictions: top-1 == 1.0.
        self.assertAlmostEqual(q_mapping(model), 1.0, places=6)
        self.assertAlmostEqual(q_mapping(model), q_list(model), places=9)

    def test_classification_quality_rejects_keys_as_batches(self):
        """Guard: iterating the mapping (old behaviour) would hand strings to the metric."""
        torch.manual_seed(0)
        model = _ToyDenseClassifier().eval()
        eval_data = {
            "input_ids": torch.randn(6, 16),
            "labels": torch.tensor([0, 1, 2, 3, 0, 1]),
        }
        with self.assertRaisesRegex(ValueError, "int64 class label per sample"):
            _make_classification_quality(model, list(eval_data), None)

    def test_classification_search_through_prune_entry(self):
        """End-to-end: evaluator="classification" with a mapping eval_data must run."""
        cfg = {
            "methods": {
                "dense": {"name": "low_variance", "kwargs": {"prune_ratio": 0.25}}
            }
        }
        chosen = []
        for eval_data_is_mapping in (True, False):
            torch.manual_seed(0)
            model = _ToyDenseClassifier().eval()
            feats = torch.randn(6, 16)
            with torch.no_grad():
                labels = model(feats).argmax(-1)
            eval_data = {"input_ids": feats, "labels": labels}
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                prune(
                    model,
                    cfg,
                    data=[feats],
                    evaluator="classification",
                    eval_data=eval_data if eval_data_is_mapping else [eval_data],
                    tolerance=0.05,
                )
            chosen.append(_params(model))
        self.assertEqual(chosen[0], chosen[1])

    def test_tolerance_mode_mapping_data(self):
        results = []
        for data in (_token_batch(), [_token_batch()]):
            torch.manual_seed(0)
            model, _ = create_mini_mlp()
            model.eval()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                prune(model, DENSE_LOWVAR_PRUNE_CFG, data=data, tolerance=0.05)
            results.append(_params(model))
        self.assertEqual(results[0], results[1])

    def test_size_budget_mode_mapping_data(self):
        results = []
        for data in (_token_batch(), [_token_batch()]):
            torch.manual_seed(0)
            model, _ = create_mini_mlp()
            model.eval()
            before = _params(model)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                prune(model, DENSE_LOWVAR_PRUNE_CFG, data=data, size_budget=0.85)
            results.append(_params(model))
            self.assertLessEqual(results[-1], int(round(before * 0.85)))
        self.assertEqual(results[0], results[1])

    def test_menu_mode_mapping_eval_data(self):
        results = []
        calib, val = _token_batch(), _token_batch(seed=1)
        for eval_data in (val, [val]):
            torch.manual_seed(0)
            model, _ = create_mini_mlp()
            model.eval()
            before = _params(model)
            report = PruneReport()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                prune(
                    model,
                    DENSE_RECOVERY_MENU_CFG,
                    data=[calib],
                    eval_data=eval_data,
                    report=report,
                )
            results.append((_params(model), report.params_after))
            self.assertLessEqual(results[-1][0], before)
        self.assertEqual(results[0], results[1])


class TestAllocationMapping(unittest.TestCase):
    def test_measure_layer_sensitivity_mapping(self):
        torch.manual_seed(0)
        model, _ = create_mini_mlp()
        model.eval()
        batch = _token_batch()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from_mapping = measure_layer_sensitivity(model, batch, ref_ratio=0.5)
            from_list = measure_layer_sensitivity(model, [batch], ref_ratio=0.5)
        self.assertEqual(sorted(from_mapping), sorted(from_list))
        for key in from_mapping:
            self.assertAlmostEqual(from_mapping[key], from_list[key], places=9)

    def test_sensitivity_allocation_mapping_data(self):
        results = []
        batch = _token_batch()
        for data in (batch, [copy.deepcopy(batch)]):
            torch.manual_seed(0)
            model, _ = create_mini_mlp()
            model.eval()
            report = PruneReport()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                prune(model, SENSITIVITY_ALLOC_PRUNE_CFG, data=data, report=report)
            results.append((_params(model), report.allocation_choice))
        self.assertEqual(results[0], results[1])
        self.assertLess(results[0][0], _params(create_mini_mlp()[0].eval()))


class TestFinetuneMapping(unittest.TestCase):
    def test_mapping_data_finetunes(self):
        def loss_fn(model, batch):
            return model(**batch).float().pow(2).mean()

        results = []
        batch = _token_batch()
        for data in (batch, [copy.deepcopy(batch)]):
            torch.manual_seed(0)
            model, _ = create_mini_mlp()
            history = prune_finetune(model, data, loss_fn=loss_fn, steps=2, lr=1e-4)
            results.append((history["steps"], history["loss_history"]))
        self.assertEqual(results[0][0], 2)
        self.assertEqual(results[0], results[1])

    def test_empty_mapping_still_raises(self):
        with self.assertRaises(ValueError):
            prune_finetune(_ToyDenseClassifier(), {}, steps=1)


class TestDiagnoseMapping(unittest.TestCase):
    def test_mapping_data_diagnoses(self):
        torch.manual_seed(0)
        model, _ = create_mini_mlp()
        model.eval()
        batch = _token_batch()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from_mapping = prune_diagnose(model, data=batch)
            from_list = prune_diagnose(model, data=[copy.deepcopy(batch)])
        self.assertTrue(from_mapping.prune_works)
        self.assertEqual(from_mapping.targets, from_list.targets)
        self.assertAlmostEqual(
            from_mapping.prune_reduction, from_list.prune_reduction, places=9
        )


class TestMappingSubclassBatching(unittest.TestCase):
    """Every Mapping is one batch -- and the adapter must agree, or it dies one layer down.

    Regression guard: ``as_batch_sequence`` widens to ``Mapping`` while
    ``default_batch_adapter`` used to gate on ``isinstance(batch, dict)``, so a
    ``UserDict`` / HF ``BatchEncoding`` was normalised into one batch and then rejected
    with the very same misleading ``TypeError`` the fix was meant to remove.
    """

    def test_every_mapping_flavour_is_a_single_batch(self):
        for name, batch in _mapping_variants(_token_batch()).items():
            with self.subTest(mapping=name):
                out = list(as_batch_sequence(batch))
                self.assertEqual(len(out), 1)
                self.assertIs(out[0], batch)

    def test_default_adapter_unwraps_every_mapping_flavour(self):
        for name, batch in _mapping_variants(_token_batch()).items():
            with self.subTest(mapping=name):
                args, kwargs = default_batch_adapter(batch)
                self.assertEqual(args, ())
                self.assertEqual(list(kwargs), ["input_ids"])
                self.assertIsInstance(kwargs, dict)

    def test_prune_context_yields_one_unwrapped_batch_per_flavour(self):
        batch = _token_batch()
        for name, variant in _mapping_variants(batch).items():
            with self.subTest(mapping=name):
                out = list(PruneContext(data=variant).iter_model_inputs())
                self.assertEqual(len(out), 1)
                args, kwargs = out[0]
                self.assertEqual(args, ())
                self.assertTrue(torch.equal(kwargs["input_ids"], batch["input_ids"]))


class TestMappingSubclassEndToEnd(unittest.TestCase):
    """The public entry points must accept a non-dict Mapping, not just a plain dict."""

    def _fixed_prune(self, data):
        torch.manual_seed(0)
        model, _ = create_mini_mlp()
        model.eval()
        report = PruneReport()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            prune(model, DENSE_LOWVAR_PRUNE_CFG, data=data, report=report)
        return _params(model), report.params_after

    def test_fixed_prune_accepts_every_mapping_flavour(self):
        reference = self._fixed_prune(_token_batch())
        for name, variant in _mapping_variants(_token_batch()).items():
            with self.subTest(mapping=name):
                self.assertEqual(self._fixed_prune(variant), reference)

    def test_fixed_prune_mapping_equals_single_element_list_per_flavour(self):
        for name, variant in _mapping_variants(_token_batch()).items():
            with self.subTest(mapping=name):
                direct = self._fixed_prune(variant)
                wrapped = self._fixed_prune([copy.deepcopy(variant)])
                self.assertEqual(direct, wrapped)
                self.assertLess(direct[0], _params(_fresh_mlp()))

    def test_finetune_accepts_every_mapping_flavour(self):
        def loss_fn(model, batch):
            return model(**batch).float().pow(2).mean()

        reference = None
        for name, variant in _mapping_variants(_token_batch()).items():
            with self.subTest(mapping=name):
                torch.manual_seed(0)
                model, _ = create_mini_mlp()
                history = prune_finetune(
                    model, variant, loss_fn=loss_fn, steps=2, lr=1e-4
                )
                self.assertEqual(history["steps"], 2)
                if reference is None:
                    reference = history["loss_history"]
                else:
                    self.assertEqual(history["loss_history"], reference)

    def test_diagnose_reports_pruning_as_available_per_flavour(self):
        """Regression guard: a Mapping must not be silently degraded into "cannot prune".

        Before the adapter gate was widened, ``prune_diagnose`` swallowed the
        ``TypeError`` and returned ``prune_works=False`` / ``prune_forward_ok=None``,
        i.e. it diagnosed a working configuration as unprunable.
        """
        for name, variant in _mapping_variants(_token_batch()).items():
            with self.subTest(mapping=name):
                torch.manual_seed(0)
                model, _ = create_mini_mlp()
                model.eval()
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    report = prune_diagnose(model, data=variant)
                self.assertTrue(report.prune_works)
                self.assertTrue(report.prune_forward_ok)
                self.assertTrue(report.search_works)
                self.assertNotIn("Unsupported batch type", " ".join(report.notes))


class TestDataLoaderStaysMultiBatch(unittest.TestCase):
    """docs/zh/api/prune.md promises DataLoader stays a batch *sequence*."""

    def _loader(self, n_samples=8, n_feat=20, batch_size=2):
        torch.manual_seed(0)
        feats = torch.randint(0, 1000, (n_samples, n_feat))
        return DataLoader(TensorDataset(feats), batch_size=batch_size)

    def test_dataloader_is_not_collapsed_into_one_batch(self):
        loader = self._loader()
        seq = as_batch_sequence(loader)
        self.assertEqual(len(list(seq)), 4)

    def test_dataloader_and_equivalent_list_agree(self):
        loader = self._loader()
        materialised = list(loader)

        def run(data):
            torch.manual_seed(0)
            model, _ = create_mini_mlp()
            model.eval()
            report = PruneReport()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                prune(model, DENSE_LOWVAR_PRUNE_CFG, data=data, report=report)
            return _params(model), report.params_after

        self.assertEqual(run(loader), run(materialised))

    def test_diagnose_does_not_consume_the_loader_while_probing(self):
        loader = self._loader()
        torch.manual_seed(0)
        model, _ = create_mini_mlp()
        model.eval()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            prune_diagnose(model, data=loader)
        self.assertEqual(len(list(loader)), 4)


class TestUnsupportedContainerFailsFast(unittest.TestCase):
    """An unusable container must raise, not be reported as "pruning unavailable"."""

    def test_diagnose_raises_on_unsupported_batch_object(self):
        torch.manual_seed(0)
        model, _ = create_mini_mlp()
        model.eval()
        with self.assertRaises(TypeError) as ctx:
            prune_diagnose(model, data=[object()])
        self.assertIn("Unsupported batch type", str(ctx.exception))

    def test_diagnose_raises_on_set_of_batches(self):
        torch.manual_seed(0)
        model, _ = create_mini_mlp()
        model.eval()
        with self.assertRaises(TypeError):
            prune_diagnose(model, data={object()})

    def test_generator_is_not_probed_eagerly(self):
        """A one-shot iterator cannot be inspected without eating a batch."""
        seen = []

        def gen():
            for i in range(3):
                batch = _token_batch(seed=i)
                seen.append(i)
                yield batch

        torch.manual_seed(0)
        model, _ = create_mini_mlp()
        model.eval()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            prune_diagnose(model, data=gen())
        self.assertEqual(seen[:1], [0])


if __name__ == "__main__":
    unittest.main()
