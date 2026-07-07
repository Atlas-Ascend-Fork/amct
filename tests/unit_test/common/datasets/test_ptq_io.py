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

import os

import pytest
import torch

from amct_pytorch.common.datasets import ptq_io

# ---- save_ptq_kwargs ------------------------------------------------------


def test_save_ptq_kwargs_creates_dir_and_writes_all_provided(tmp_path):
    target_dir = tmp_path / "kw"
    pos = torch.tensor([[0, 1, 2]])
    pe = (torch.ones(1, 3, 4), torch.zeros(1, 3, 4))
    mask = torch.tensor([[1, 1, 1]])

    ptq_io.save_ptq_kwargs(pos, pe, mask, str(target_dir))

    assert target_dir.is_dir()
    assert torch.equal(torch.load(target_dir / "position_ids.pkl"), pos)
    pe_loaded = torch.load(target_dir / "position_embeddings.pkl")
    assert torch.equal(pe_loaded[0], pe[0]) and torch.equal(pe_loaded[1], pe[1])
    assert torch.equal(torch.load(target_dir / "attention_mask.pkl"), mask)


def test_save_ptq_kwargs_skips_none_arguments(tmp_path):
    ptq_io.save_ptq_kwargs(None, None, None, str(tmp_path))
    assert not (tmp_path / "position_ids.pkl").exists()
    assert not (tmp_path / "position_embeddings.pkl").exists()
    assert not (tmp_path / "attention_mask.pkl").exists()


def test_save_ptq_kwargs_writes_only_non_none(tmp_path):
    ptq_io.save_ptq_kwargs(
        torch.tensor([[1]]), None, torch.tensor([[1]]), str(tmp_path)
    )
    assert (tmp_path / "position_ids.pkl").exists()
    assert not (tmp_path / "position_embeddings.pkl").exists()
    assert (tmp_path / "attention_mask.pkl").exists()


# ---- save_ptq_inps --------------------------------------------------------


def test_save_ptq_inps_concatenates_outs_along_dim0(tmp_path):
    act_stat = {
        "hookA_out": [torch.zeros(2, 4), torch.ones(3, 4)],
    }
    ptq_io.save_ptq_inps(
        act_stat, hook_name="hookA", quant_target="mlp", layer_idx=7, data_dir=str(tmp_path)
    )
    saved = torch.load(tmp_path / "block_7_mlp_in.pkl")
    assert saved.shape == (5, 4)
    # Concat is in order: zeros first, then ones.
    assert torch.equal(saved[:2], torch.zeros(2, 4))
    assert torch.equal(saved[2:], torch.ones(3, 4))


# ---- load_ptq_inps --------------------------------------------------------


def test_load_ptq_inps_reads_kwargs_and_block_for_attn(tmp_path):
    pos = torch.tensor([[0, 1]])
    pe = (torch.ones(1, 2), torch.zeros(1, 2))
    mask = torch.tensor([[1, 1]])
    inps = torch.randn(2, 4)
    torch.save(pos, tmp_path / "position_ids.pkl")
    torch.save(pe, tmp_path / "position_embeddings.pkl")
    torch.save(mask, tmp_path / "attention_mask.pkl")
    torch.save(inps, tmp_path / "block_3_attn_in.pkl")

    cached, kwargs = ptq_io.load_ptq_inps(str(tmp_path), "attn", layer_idx=3)
    assert torch.equal(cached, inps)
    assert torch.equal(kwargs["position_ids"], pos)
    assert torch.equal(kwargs["attention_mask"], mask)
    pe_loaded = kwargs["position_embeddings"]
    assert torch.equal(pe_loaded[0], pe[0]) and torch.equal(pe_loaded[1], pe[1])


def test_load_ptq_inps_returns_empty_kwargs_for_non_attn_target(tmp_path):
    inps = torch.randn(2, 4)
    torch.save(inps, tmp_path / "block_0_mlp_in.pkl")
    cached, kwargs = ptq_io.load_ptq_inps(str(tmp_path), "mlp", layer_idx=0)
    assert torch.equal(cached, inps)
    assert not kwargs


def test_load_ptq_inps_returns_none_when_block_file_missing(tmp_path):
    cached, kwargs = ptq_io.load_ptq_inps(str(tmp_path), "mlp", layer_idx=99)
    assert cached is None
    assert not kwargs


def test_load_ptq_inps_skips_missing_optional_kwargs_files(tmp_path):
    # attn target but only mask exists; position_ids/position_embeddings missing.
    torch.save(torch.tensor([[1, 1]]), tmp_path / "attention_mask.pkl")
    torch.save(torch.zeros(1, 4), tmp_path / "block_0_attn_in.pkl")
    cached, kwargs = ptq_io.load_ptq_inps(str(tmp_path), "attn", layer_idx=0)
    assert "attention_mask" in kwargs
    assert "position_ids" not in kwargs
    assert "position_embeddings" not in kwargs
    assert cached is not None
