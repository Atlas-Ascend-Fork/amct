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
"""Large activation offsets must not reverse variance-based neuron selection."""

import pytest
import torch

from amct_pytorch.pruning.utils import RunningVariance, topk_keep_indices


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("chunks", [1, 2])
def test_offset_variance_retains_neuron_ranking(dtype, chunks):
    values = torch.tensor([[9998.0, -1.0], [10002.0, 1.0]], dtype=dtype)
    stats = RunningVariance()
    for chunk in values.chunk(chunks):
        stats.update(chunk)
    expected = values.double().var(dim=0, correction=0)
    torch.testing.assert_close(stats.variance().double(), expected)
    assert topk_keep_indices(stats.variance(), prune_ratio=0.5, min_keep=1) == [0]


def test_fp64_activation_values_are_not_rounded_to_fp32():
    values = torch.tensor([[1e9 - 2], [1e9 + 2]], dtype=torch.float64)
    stats = RunningVariance()
    stats.update(values[:1])
    stats.update(values[1:])
    torch.testing.assert_close(
        stats.variance().double(), torch.tensor([4.0], dtype=torch.float64)
    )


@pytest.mark.parametrize("chunks", [1, 2, 4])
def test_population_variance_and_batch_partition_agree(chunks):
    values = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
    stats = RunningVariance()
    for chunk in values.chunk(chunks):
        stats.update(chunk)
    assert stats.count == 4
    torch.testing.assert_close(
        stats.variance().double(), values.double().var(0, correction=0)
    )


def test_no_samples_still_raises():
    with pytest.raises(RuntimeError, match="no samples"):
        RunningVariance().variance()
