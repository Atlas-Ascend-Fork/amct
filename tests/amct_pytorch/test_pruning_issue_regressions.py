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
"""Regressions for pruning issues #234-#237 and #240-#243."""

from collections import UserDict

import pytest
import torch
from torch import nn

from amct_pytorch.pruning.calib import calib_nll, nll_from_logits
from amct_pytorch.pruning.config import MethodSpec, PruneConfig
from amct_pytorch.pruning.context import PruneContext
from amct_pytorch.pruning.finetune import _move_to_device, prune_finetune
from amct_pytorch.pruning.prune_op.cnn_reconstruct import (
    _conv_bias_fold_delta,
    _reconstruct_conv_consumer,
)
from amct_pytorch.pruning.prune_op.dense_reconstruct import _collect_consumer_acts
from amct_pytorch.pruning.prune_op.moe_output_merge import _collect_block_inputs
from amct_pytorch.pruning.pruner import PruningStage
from amct_pytorch.pruning.report import PruneReport
from amct_pytorch.pruning.utils import (
    RunningVariance,
    ridge_solve,
    run_calibration_epoch,
)


class TinyLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(8, 4)
        self.head = nn.Linear(4, 8)

    def forward(self, ids):
        return self.head(self.embed(ids))


def test_int32_calibration_targets_keep_model_input_dtype():
    model = TinyLM()
    ids = torch.tensor([[1, 2, 3]], dtype=torch.int32)
    assert calib_nll(model, [ids]) == pytest.approx(calib_nll(model, [ids.long()]))
    with torch.no_grad():
        logits = model(ids)
        assert nll_from_logits(logits, ids) == pytest.approx(
            nll_from_logits(logits, ids.long())
        )
    assert ids.dtype == torch.int32


@pytest.mark.parametrize("shape", [(1, 1), (0, 3), (1, 0)])
def test_calibration_without_next_token_targets_raises(shape):
    with pytest.raises(ValueError, match="next-token"):
        calib_nll(TinyLM(), [torch.zeros(shape, dtype=torch.long)])


def test_calibration_ignores_empty_batch_if_others_have_targets():
    model = TinyLM()
    valid = torch.tensor([[1, 2, 3]])
    assert calib_nll(model, [valid[:, :1], valid]) == pytest.approx(
        calib_nll(model, [valid])
    )


def test_running_variance_preserves_large_offset_and_batch_splits():
    values = torch.tensor([[9998.0, -1.0], [10002.0, 1.0]], dtype=torch.float64)
    for chunks in ([values], [values[:1], values[1:]]):
        stats = RunningVariance()
        for chunk in chunks:
            stats.update(chunk)
        torch.testing.assert_close(
            stats.variance(), torch.tensor([4.0, 1.0], dtype=torch.float64)
        )


@pytest.mark.parametrize("mode", ["reflect", "replicate", "circular"])
def test_cnn_reconstruction_matches_nonzero_padding(mode):
    torch.manual_seed(4)
    conv = nn.Conv2d(2, 1, 3, padding=1, padding_mode=mode, bias=False)
    x = torch.randn(8, 2, 7, 7)
    restored = _reconstruct_conv_consumer(conv, x, conv(x), ridge=1e-8)
    torch.testing.assert_close(restored, conv.weight, rtol=2e-3, atol=2e-3)
    delta = _conv_bias_fold_delta(conv, x, [0], 2)
    removed = x.clone()
    removed[:, 0] = 0
    torch.testing.assert_close(delta, conv(removed).mean((0, 2, 3)))


def test_cnn_reconstruction_supports_same_padding():
    torch.manual_seed(5)
    conv = nn.Conv2d(1, 1, 2, padding="same", bias=False)
    x = torch.randn(8, 1, 7, 7)
    restored = _reconstruct_conv_consumer(conv, x, conv(x), ridge=1e-8)
    torch.testing.assert_close(restored, conv.weight, rtol=2e-3, atol=2e-3)


def test_nested_mapping_batch_moves_all_tensor_leaves():
    batch = {
        "ids": torch.ones(2),
        "cache": {"states": [torch.zeros(2)]},
        "meta": "kept",
    }
    moved = _move_to_device(batch, torch.device("meta"))
    assert moved["ids"].device.type == "meta"
    assert moved["cache"]["states"][0].device.type == "meta"
    assert batch["cache"]["states"][0].device.type == "cpu"
    assert moved["meta"] == "kept"


def test_user_mapping_batch_keeps_recursive_device_transfer():
    batch = UserDict(
        {"ids": torch.ones(2), "nested": UserDict({"states": [torch.zeros(2)]})}
    )
    moved = _move_to_device(batch, torch.device("meta"))
    assert moved["ids"].device.type == "meta"
    assert moved["nested"]["states"][0].device.type == "meta"
    assert batch["ids"].device.type == "cpu"


def test_failed_stage_discards_only_its_report_changes():
    class Domain:
        def find_targets(self, model, config):
            return [model]

    class Method:
        name = "partial"
        requires_targets = True
        requires_data = False

        def apply(self, model, domain, targets, context, report, config, spec):
            report.add("dense", self.name, "layer", "cut")
            report.record_layer_sparsity("layer", 4, 2)
            raise RuntimeError("later target failed")

    model = nn.Linear(4, 4)
    report = PruneReport()
    report.add("earlier", "ok", "before", "retained")
    stage = PruningStage("dense", Domain(), Method(), MethodSpec("partial"))
    result = stage.apply(
        model, PruneContext(), report, PruneConfig(stage_error_policy="warn_skip")
    )
    assert result is model
    assert [event.module for event in report.events] == ["before"]
    assert report.per_layer_sparsity == {}
    assert len(report.warnings) == 1


def test_successful_stage_publishes_its_report_changes():
    class Domain:
        def find_targets(self, model, config):
            return [model]

    class Method:
        name = "complete"
        requires_targets = True
        requires_data = False

        def apply(self, model, domain, targets, context, report, config, spec):
            report.add("dense", self.name, "layer", "cut")
            report.record_layer_sparsity("layer", 4, 2)

    report = PruneReport()
    stage = PruningStage("dense", Domain(), Method(), MethodSpec("complete"))
    stage.apply(
        nn.Linear(4, 4),
        PruneContext(),
        report,
        PruneConfig(stage_error_policy="warn_skip"),
    )
    assert [event.module for event in report.events] == ["layer"]
    assert report.per_layer_sparsity == {"layer": 0.5}


@pytest.mark.parametrize("operation", ["calibration", "finetune"])
def test_temporary_modes_restore_each_submodule(operation):
    model = nn.Sequential(nn.Linear(4, 4), nn.Dropout(0.2), nn.Linear(4, 2))
    model.train()
    model[1].eval()
    before = [module.training for module in model.modules()]
    if operation == "calibration":
        run_calibration_epoch(model, PruneContext(data=[torch.randn(2, 4)]))
    else:
        prune_finetune(
            model, [(torch.randn(2, 4), torch.zeros(2, dtype=torch.long))], steps=1
        )
    assert [module.training for module in model.modules()] == before


@pytest.mark.parametrize("operation", ["calibration", "finetune"])
def test_temporary_modes_restore_each_submodule_on_error(operation):
    class FailingModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(4, 4)
            self.dropout = nn.Dropout(0.2)

        def forward(self, inputs):
            raise RuntimeError("forward failed")

    model = FailingModel()
    model.train()
    model.dropout.eval()
    before = [module.training for module in model.modules()]
    with pytest.raises(RuntimeError, match="forward failed"):
        if operation == "calibration":
            run_calibration_epoch(model, PruneContext(data=[torch.randn(2, 4)]))
        else:
            prune_finetune(
                model,
                [torch.randn(2, 4)],
                loss_fn=lambda current, batch: current(batch).sum(),
                steps=1,
            )
    assert [module.training for module in model.modules()] == before


@pytest.mark.parametrize("collector", ["dense", "moe"])
def test_other_pruning_collectors_restore_mixed_module_modes(collector):
    model = nn.Sequential(nn.Linear(4, 4), nn.Dropout(0.2))
    model.train()
    model[1].eval()
    before = [module.training for module in model.modules()]
    context = PruneContext(data=[torch.randn(2, 4)])
    if collector == "dense":
        _collect_consumer_acts(model, [], context, torch.device("cpu"))
    else:
        _collect_block_inputs(model, [model[0]], context)
    assert [module.training for module in model.modules()] == before


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_positive_ridge_solves_zero_gram(dtype):
    gram = torch.zeros(2, 2, dtype=dtype)
    result = ridge_solve(gram, torch.zeros(2, 1, dtype=dtype), ridge=1e-4)
    torch.testing.assert_close(result, torch.zeros_like(result))


def test_ridge_keeps_relative_scale_for_nonzero_gram():
    gram = torch.diag(torch.tensor([2.0, 4.0]))
    ridge_solve(gram, torch.zeros(2, 1), ridge=0.1)
    torch.testing.assert_close(gram.diagonal(), torch.tensor([2.3, 4.3]))
