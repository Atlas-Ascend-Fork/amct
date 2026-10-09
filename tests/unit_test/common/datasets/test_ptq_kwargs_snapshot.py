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


"""Repeated extraction must replace the complete optional kwargs snapshot."""

import pytest
import torch

from amct_pytorch.common.datasets import ptq_io


@pytest.mark.parametrize(
    "keep", [None, "position_ids", "position_embeddings", "attention_mask"]
)
def test_second_extraction_does_not_load_absent_previous_kwargs(tmp_path, keep):
    old = {
        "position_ids": torch.tensor([[0, 1, 2]]),
        "position_embeddings": (torch.ones(1, 3), torch.zeros(1, 3)),
        "attention_mask": torch.ones(1, 3),
    }
    ptq_io.save_ptq_kwargs(**old, data_dir=str(tmp_path))
    current = {key: value if key == keep else None for key, value in old.items()}
    if keep is not None:
        current[keep] = (
            (torch.ones(1, 2), torch.zeros(1, 2))
            if keep == "position_embeddings"
            else torch.full((1, 2), 7)
        )
    inputs = torch.ones(1, 2, 4)
    torch.save(inputs, tmp_path / "block_0_attn_in.pkl")
    sentinel = tmp_path / "unrelated.txt"
    sentinel.write_text("preserve")

    ptq_io.save_ptq_kwargs(**current, data_dir=str(tmp_path))
    cached, kwargs = ptq_io.load_ptq_inps(str(tmp_path), "attn", 0)

    assert set(kwargs) == ({keep} if keep else set())
    assert torch.equal(cached, inputs)
    assert sentinel.read_text() == "preserve"
    if keep:
        actual, expected = kwargs[keep], current[keep]
        if isinstance(expected, tuple):
            assert all(torch.equal(a, b) for a, b in zip(actual, expected))
        else:
            assert torch.equal(actual, expected)
