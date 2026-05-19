#!/usr/bin/env python3
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

from types import SimpleNamespace

import pytest
import torch

from amct_pytorch.common.datasets.ptq_provider import (
    BlockPtqBatch,
    LlmPtqDataProvider,
)
from amct_pytorch.common.models.llm.common.ptq_units import make_ptq_unit

MLP = 'mlp'


@pytest.fixture(autouse=True)

def _stub_npu_empty_cache(monkeypatch):
    fake = type("F", (), {"empty_cache": staticmethod(lambda: None)})()
    monkeypatch.setattr(torch, "npu", fake, raising=False)


def _make_provider(cali_bsz=2, device="cpu", data_dir="/tmp/ptq", pipeline=None):
    args = SimpleNamespace(
        cali_bsz=cali_bsz,
        device=device,
        data_dir=data_dir,
        start_block_idx=2,
        end_block_idx=5,
    )
    return LlmPtqDataProvider(args, pipeline=pipeline)


# ---- load_unit_inputs / get_model_data ----------------------------------


def test_load_unit_inputs_delegates_to_pipeline():
    captured = {}

    class _Pipe:
        @staticmethod
        def load_unit_inputs(data_dir, unit):
            captured["data_dir"] = data_dir
            captured["unit"] = unit
            return ("inps", "kwargs")

    provider = _make_provider(data_dir="/d", pipeline=_Pipe())
    unit = make_ptq_unit(MLP, MLP, 1, module=None)
    out = provider.load_unit_inputs(unit)
    assert out == ("inps", "kwargs")
    assert captured == {"data_dir": "/d", "unit": unit}


def test_get_model_data_packs_args():
    provider = _make_provider(data_dir="/d")
    assert provider.get_model_data() == {
        "data_dir": "/d",
        "start_block_idx": 2,
        "end_block_idx": 5,
    }


# ---- build_unit_batch ----------------------------------------------------


def test_build_unit_batch_without_gts():
    provider = _make_provider(cali_bsz=2)
    inps = torch.arange(8).reshape(4, 2)
    unit = make_ptq_unit(MLP, MLP, layer_idx=3, module=None, metadata={"k": "v"})

    batch = provider.build_unit_batch(unit, inps, kwargs={"a": 1})

    assert isinstance(batch, BlockPtqBatch)
    assert batch.layer_idx == 3
    assert batch.unit_name == MLP
    assert batch.kwargs == {"a": 1}
    assert batch.has_gts is False
    assert batch.metadata == {"k": "v"}
    assert batch.num_samples == 4

    # DataLoader yields cali_bsz at a time, single tensor (no gts).
    batches = list(batch.data_loader)
    assert len(batches) == 2
    assert all(len(b) == 1 for b in batches)
    assert batches[0][0].shape == (2, 2)


def test_build_unit_batch_with_gts_pairs_inputs_and_targets():
    provider = _make_provider(cali_bsz=2)
    inps = torch.arange(4).reshape(2, 2)
    gts = torch.arange(4, 8).reshape(2, 2)
    unit = make_ptq_unit(MLP, MLP, 0, module=None)

    batch = provider.build_unit_batch(unit, inps, kwargs=None, gts=gts)
    assert batch.has_gts is True
    assert batch.kwargs is None
    (x, y), = list(batch.data_loader)
    assert torch.equal(x, inps)
    assert torch.equal(y, gts)


def test_build_unit_batch_metadata_is_none_when_unit_has_empty_dict():
    provider = _make_provider()
    # make_ptq_unit defaults metadata to {}; provider downgrades to None.
    unit = make_ptq_unit(MLP, MLP, 0, module=None)
    batch = provider.build_unit_batch(unit, torch.zeros(2, 2), kwargs=None)
    assert batch.metadata is None


# ---- materialize_gt -------------------------------------------------------


class _ScalingModule(torch.nn.Module):
    def __init__(self, factor):
        super().__init__()
        self.factor = factor

    def forward(self, x, **kwargs):
        return x * self.factor


def test_materialize_gt_runs_module_and_concatenates():
    provider = _make_provider(cali_bsz=2)
    inps = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    out = provider.materialize_gt(inps, _ScalingModule(2.0))
    assert out.shape == inps.shape
    assert torch.equal(out, inps * 2.0)


def test_materialize_gt_passes_forward_kwargs():
    seen = []

    class _M(torch.nn.Module):
        def forward(self, x, scale=1.0):
            seen.append(scale)
            return x * scale

    provider = _make_provider(cali_bsz=2)
    inps = torch.ones(4, 2)
    out = provider.materialize_gt(inps, _M(), kwargs={"scale": 3.0})
    assert seen == [3.0, 3.0]   # 4 samples / cali_bsz=2 -> 2 batches
    assert torch.equal(out, inps * 3.0)


def test_materialize_gt_takes_first_when_module_returns_tuple():
    class _M(torch.nn.Module):
        def forward(self, x, **kw):
            return x + 1, "aux"

    provider = _make_provider(cali_bsz=2)
    inps = torch.zeros(2, 3)
    out = provider.materialize_gt(inps, _M())
    assert torch.equal(out, torch.ones(2, 3))


def test_materialize_gt_non_floating_point_input():
    class _EmbeddingModule(torch.nn.Module):
        def forward(self, x, **kwargs):
            return x.float() * 2.0

    provider = _make_provider(cali_bsz=2)
    inps = torch.randint(0, 10, (4,)).to(torch.int64)
    out = provider.materialize_gt(inps, _EmbeddingModule())
    assert out.shape == (4,)
    assert out.dtype == torch.float32

