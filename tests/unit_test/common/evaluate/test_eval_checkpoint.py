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
from unittest.mock import patch

import pytest
import torch

from amct_pytorch.common.evaluate.eval_checkpoint import EvalCheckpointManager


def _manager(tmp_path, **overrides):
    args = dict(
        model_name="qwen3",
        quant_dtype="int",
        eval_mode="quant",
        output_dir=str(tmp_path),
    )
    args.update(overrides)
    return EvalCheckpointManager(SimpleNamespace(**args), 20)


def test_latest_output_replaces_previous_and_resumes(tmp_path):
    manager = _manager(tmp_path)
    manager.save(2, [torch.tensor([2.0])])
    manager.save(12, [torch.tensor([12.0])])
    assert [p.name for p in manager.run_dir.iterdir()] == ["qwen3_int_12.pt"]
    block_idx, outputs, state = _manager(tmp_path, algos=["lac"], seq_len=1024).load()
    assert state is None
    assert block_idx == 12
    assert torch.equal(outputs[0], torch.tensor([12.0]))


def test_load_selects_largest_numeric_block(tmp_path):
    manager = _manager(tmp_path)
    manager.save(2, [torch.tensor([2.0])])
    torch.save(
        {"completed_block_idx": 12, "inter_io": [torch.tensor([12.0])]},
        manager.run_dir / "qwen3_int_12.pt",
    )
    assert manager.load()[0] == 12


def test_cleanup_preserves_other_models_and_dtypes(tmp_path):
    manager = _manager(tmp_path)
    other = _manager(tmp_path, model_name="qwen3_5")
    bf16 = _manager(tmp_path, eval_mode="bf16")
    for item in (manager, other, bf16):
        item.save(0, [torch.tensor([1.0])])
    manager.cleanup()
    assert other.load()[0] == 0
    assert bf16.load()[0] == 0
    assert manager.load() is None


def test_corrupt_latest_falls_back_and_removes_incomplete_write(tmp_path):
    manager = _manager(tmp_path)
    manager.save(2, [torch.tensor([2.0])])
    corrupt = manager.run_dir / "qwen3_int_12.pt"
    corrupt.write_bytes(b"invalid")
    temporary = manager.run_dir / "qwen3_int_13.pt.tmp"
    temporary.write_bytes(b"incomplete")
    assert manager.load()[0] == 2
    assert not corrupt.exists()
    assert not temporary.exists()


def test_failed_save_keeps_previous_checkpoint(tmp_path):
    manager = _manager(tmp_path)
    manager.save(2, [torch.tensor([2.0])])
    with patch(
        "amct_pytorch.common.evaluate.eval_checkpoint.torch.save",
        side_effect=OSError("disk full"),
    ):
        with pytest.raises(OSError, match="disk full"):
            manager.save(3, [torch.tensor([3.0])])
    assert manager.load()[0] == 2


@pytest.mark.parametrize(
    "payload",
    [
        None,
        [],
        {},
        {"inter_io": ["invalid"]},
        {"inter_io": torch.tensor([1.0])},
        {"inter_io": []},
    ],
)
def test_invalid_checkpoint_restarts(tmp_path, payload):
    manager = _manager(tmp_path)
    manager.run_dir.mkdir(parents=True)
    torch.save(payload, manager.run_dir / "qwen3_int_0.pt")
    assert manager.load() is None


@pytest.mark.parametrize("legacy_fields", [{}, {"completed_block_idx": "unused"}])
def test_load_uses_filename_block_index(tmp_path, legacy_fields):
    manager = _manager(tmp_path)
    manager.run_dir.mkdir(parents=True)
    torch.save(
        {"inter_io": [torch.tensor([2.0])], **legacy_fields},
        manager.run_dir / "qwen3_int_2.pt",
    )
    assert manager.load()[0] == 2


def test_out_of_range_checkpoint_restarts(tmp_path):
    manager = _manager(tmp_path)
    manager.run_dir.mkdir(parents=True)
    path = manager.run_dir / "qwen3_int_20.pt"
    torch.save({"inter_io": [torch.tensor([1.0])]}, path)
    assert manager.load() is None
    assert not path.exists()


def test_missing_checkpoint_creates_directory(tmp_path):
    manager = _manager(tmp_path)
    assert manager.load() is None
    assert manager.run_dir.is_dir()


def test_model_state_round_trip(tmp_path):
    manager = _manager(tmp_path, model_name="glm5_2")
    indices = torch.tensor([[[0, 1]]], dtype=torch.int32)
    manager.save(
        2,
        [torch.zeros(1, 1, 4)],
        model_state=[{"prev_topk_indices": indices, "use_cache": False}],
    )
    block_idx, outputs, state = manager.load()
    assert block_idx == 2
    assert outputs[0].shape == (1, 1, 4)
    assert torch.equal(state[0]["prev_topk_indices"], indices)
    assert state[0]["use_cache"] is False
