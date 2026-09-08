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
"""Unit tests for seed_everything (mirrors examples flatquant utils)."""

import random
from types import SimpleNamespace

import numpy as np
import torch

from amct_pytorch.common.utils import seed_utils


def _mock_global_rngs(monkeypatch, captured):
    monkeypatch.setattr(random, 'seed', lambda s: captured.__setitem__('random', s))
    monkeypatch.setattr(np.random, 'seed', lambda s: captured.__setitem__('numpy', s))
    monkeypatch.setattr(
        torch, 'manual_seed', lambda s: captured.__setitem__('torch', s)
    )
    monkeypatch.setattr(
        seed_utils, 'set_seed', lambda s: captured.__setitem__('transformers', s)
    )


def _mock_npu(monkeypatch, calls):
    fake_npu = SimpleNamespace(
        manual_seed=lambda s: calls.append(('manual_seed', s)),
        manual_seed_all=lambda s: calls.append(('manual_seed_all', s)),
    )
    monkeypatch.setattr(torch, 'npu', fake_npu)


def test_seed_everything_seeds_random_numpy_torch_and_transformers(monkeypatch):
    captured = {}
    calls = []
    _mock_global_rngs(monkeypatch, captured)
    _mock_npu(monkeypatch, calls)

    seed_utils.seed_everything(7)

    assert captured == {'random': 7, 'numpy': 7, 'torch': 7, 'transformers': 7}
    assert calls == [('manual_seed', 7), ('manual_seed_all', 7)]


def test_seed_everything_makes_rngs_deterministic():
    seed_utils.seed_everything(3)
    py_a, np_a, torch_a = random.random(), np.random.rand(), torch.randn(2)
    seed_utils.seed_everything(3)
    py_b, np_b, torch_b = random.random(), np.random.rand(), torch.randn(2)
    assert py_a == py_b
    assert np_a == np_b
    assert torch.equal(torch_a, torch_b)
