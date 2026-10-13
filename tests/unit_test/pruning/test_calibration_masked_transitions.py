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
"""Real CPU calibration scores must not depend on masked padding tokens."""

import importlib
from pathlib import Path
import sys
import types

import pytest
import torch
from torch import nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def calibration_api():
    # Bypass only the unrelated hardware-dependent root initializer. The
    # complete real calibration, PruneContext, and CPU model forward are used.
    previous = {
        name: module
        for name, module in sys.modules.items()
        if name == "amct_pytorch" or name.startswith("amct_pytorch.")
    }
    for name in previous:
        sys.modules.pop(name, None)
    package = types.ModuleType("amct_pytorch")
    package.__path__ = [str(ROOT / "amct_pytorch")]
    sys.modules.update({"amct_pytorch": package})
    try:
        yield importlib.import_module("amct_pytorch.pruning.calib")
    finally:
        names = [
            name
            for name in sys.modules
            if name == "amct_pytorch" or name.startswith("amct_pytorch.")
        ]
        for name in names:
            sys.modules.pop(name, None)
        sys.modules.update(previous)


class TinyLM(nn.Module):
    def __init__(self, weights=None):
        super().__init__()
        if weights is None:
            weights = [[0, 5, 0], [0, 0, 2], [0, 5, 0]]
        self.output = nn.Embedding.from_pretrained(
            torch.tensor(weights, dtype=torch.float64)
        )

    def forward(self, input_ids, attention_mask=None):
        return self.output(input_ids)


@pytest.mark.parametrize(
    "ids, mask",
    [([[1, 2, 0, 0]], [[1, 1, 0, 0]]), ([[0, 0, 1, 2]], [[0, 0, 1, 1]])],
)
def test_padding_keeps_the_same_valid_sequence_score(calibration_api, ids, mask):
    model = TinyLM()
    reference = calibration_api.calib_nll(model, [torch.tensor([[1, 2]])])
    expected = F.cross_entropy(
        torch.tensor([[0.0, 0.0, 2.0]], dtype=torch.float64), torch.tensor([2])
    ).item()
    batch = {"input_ids": torch.tensor(ids), "attention_mask": torch.tensor(mask)}
    assert calibration_api.calib_nll(model, [batch]) == pytest.approx(expected)
    assert reference == pytest.approx(expected)


def test_padding_does_not_reverse_candidate_quality(calibration_api):
    better = TinyLM()
    worse = TinyLM([[5, 0, 0], [0, 0, 1], [5, 0, 0]])
    plain = [torch.tensor([[1, 2]])]
    padded = [
        {
            "input_ids": torch.tensor([[1, 2, 0, 0]]),
            "attention_mask": torch.tensor([[1, 1, 0, 0]]),
        }
    ]
    assert calibration_api.calib_nll(better, plain) < calibration_api.calib_nll(
        worse, plain
    )
    assert calibration_api.calib_nll(better, padded) < calibration_api.calib_nll(
        worse, padded
    )


def test_unequal_padded_sequences_use_valid_token_count(calibration_api):
    model = TinyLM()
    ids = torch.tensor([[1, 2, 0, 0], [1, 2, 1, 2]])
    mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 1]], dtype=torch.bool)
    expected_logits = torch.tensor(
        [[0, 0, 2], [0, 0, 2], [0, 5, 0], [0, 0, 2]], dtype=torch.float64
    )
    expected = F.cross_entropy(
        expected_logits, torch.tensor([2, 2, 1, 2]), reduction="sum"
    ).item()
    assert calibration_api.calib_nll(
        model, [{"input_ids": ids, "attention_mask": mask}]
    ) == pytest.approx(expected / 4)


def test_no_mask_keeps_all_existing_targets(calibration_api):
    model = TinyLM()
    ids = torch.tensor([[1, 2, 0, 0]])
    expected = F.cross_entropy(
        model(ids)[:, :-1].reshape(-1, 3), ids[:, 1:].reshape(-1)
    ).item()
    assert calibration_api.calib_nll(model, [ids]) == pytest.approx(expected)


def test_all_valid_mask_matches_existing_unmasked_result(calibration_api):
    model = TinyLM()
    ids = torch.tensor([[1, 2, 1, 2]])
    reference = calibration_api.calib_nll(model, [ids])
    batch = {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
    assert calibration_api.calib_nll(model, [batch]) == pytest.approx(reference)


def test_all_masked_batch_is_not_a_zero_loss_candidate(calibration_api):
    ids = torch.tensor([[1, 2, 0, 0]])
    batch = {"input_ids": ids, "attention_mask": torch.zeros_like(ids)}
    with pytest.raises(ValueError, match="needs at least one next-token target"):
        calibration_api.calib_nll(TinyLM(), [batch])


def test_invalid_mask_shape_is_reported(calibration_api):
    batch = {"input_ids": torch.tensor([[1, 2]]), "attention_mask": torch.tensor([[1]])}
    with pytest.raises(ValueError, match="attention_mask must match input_ids shape"):
        calibration_api.calib_nll(TinyLM(), [batch])


@pytest.fixture
def moe_calibration_api(calibration_api):
    # Execute the second real consumer too; no pruning/model function is replaced.
    consumer = importlib.import_module("amct_pytorch.pruning.prune_op.moe_output_merge")
    nll_function = vars(consumer).get("_calib_nll")
    assert callable(nll_function)
    context_class = importlib.import_module("amct_pytorch.pruning.context").PruneContext
    return nll_function, context_class


@pytest.mark.parametrize(
    "ids, mask",
    [([[1, 2, 0, 0]], [[1, 1, 0, 0]]), ([[0, 0, 1, 2]], [[0, 0, 1, 1]])],
)
def test_moe_selector_uses_the_same_valid_sequence_score(
    moe_calibration_api, ids, mask
):
    nll_function, context_class = moe_calibration_api
    model = TinyLM()
    reference = nll_function(model, context_class(data=[torch.tensor([[1, 2]])]))
    context = context_class(
        data=[{"input_ids": torch.tensor(ids), "attention_mask": torch.tensor(mask)}]
    )
    assert nll_function(model, context) == pytest.approx(reference)


def test_moe_selector_reports_no_valid_targets(moe_calibration_api):
    nll_function, context_class = moe_calibration_api
    ids = torch.tensor([[1, 2, 0, 0]])
    context = context_class(
        data=[{"input_ids": ids, "attention_mask": torch.zeros_like(ids)}]
    )
    assert nll_function(TinyLM(), context) is None


def test_moe_selector_keeps_unmasked_score(moe_calibration_api, calibration_api):
    nll_function, context_class = moe_calibration_api
    model = TinyLM()
    ids = torch.tensor([[1, 2, 0, 0]])
    assert nll_function(model, context_class(data=[ids])) == pytest.approx(
        calibration_api.calib_nll(model, [ids])
    )
