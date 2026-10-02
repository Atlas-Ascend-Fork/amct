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

"""PPL failure must close a retained streaming producer immediately."""

import pytest
import torch

from amct_pytorch.common.evaluate import eval_ppl


@pytest.mark.parametrize("failure", ["loss", "extra", "device"])
def test_failed_ppl_closes_retained_generator(tmp_path, failure):
    artifact = tmp_path / "producer-open.txt"
    artifact.write_text("open")
    closed = []

    def producer():
        try:
            yield torch.zeros(1, 3, 8)
            yield torch.zeros(1, 3, 8)
        finally:
            closed.append(True)
            artifact.unlink()

    iterator = producer()
    samples = [torch.zeros(1, 4, dtype=torch.long)]
    if failure == "loss":
        samples[0][:, 1:] = 8  # Out-of-range labels exercise real cross entropy.
    device = "cuda:0" if failure == "device" else "cpu"
    error = IndexError if failure == "loss" else ValueError
    try:
        with pytest.raises(error):
            eval_ppl.wikitext2_ppl(iterator, samples, device=device, seq_len=4)
        assert closed == [True]
        assert not artifact.exists()
    finally:
        # Also clean up on the unfixed baseline; retain the reference for repro.
        iterator.close()


def test_successful_ppl_keeps_result_and_closes_producer():
    closed = []

    def producer():
        try:
            yield torch.zeros(1, 3, 8)
        finally:
            closed.append(True)

    iterator = producer()
    samples = [torch.zeros(1, 4, dtype=torch.long)]
    assert eval_ppl.wikitext2_ppl(
        iterator, samples, device="cpu", seq_len=4
    ) == pytest.approx(8.0)
    assert closed == [True]
