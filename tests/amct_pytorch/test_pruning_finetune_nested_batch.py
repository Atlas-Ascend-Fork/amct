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
"""Fine-tuning recursively transfers every Tensor leaf in Mapping batches."""

from collections import UserDict
from types import MappingProxyType

import pytest
import torch
from torch import nn

from amct_pytorch.pruning import prune_finetune
from amct_pytorch.pruning.finetune import _move_to_device
from amct_pytorch.pruning.utils import move_batch_to_device


MAPPINGS = [dict, UserDict, MappingProxyType]


@pytest.mark.parametrize("container", [dict, list, tuple, UserDict, MappingProxyType])
@pytest.mark.parametrize("nested_mapping", MAPPINGS)
def test_nested_tensor_leaves_are_moved_without_accelerator(container, nested_mapping):
    tensor = torch.ones(2)
    nested = nested_mapping(
        {
            "states": [tensor, nested_mapping({"cache": (tensor, None)})],
            "name": "sample",
        }
    )
    batch = container(nested) if container in MAPPINGS else container([nested])
    moved = _move_to_device(batch, torch.device("meta"))
    item = moved if container in MAPPINGS else moved[0]
    assert isinstance(moved, dict if container in MAPPINGS else container)
    assert item["states"][0].device.type == "meta"
    assert item["states"][1]["cache"][0].device.type == "meta"
    assert isinstance(item["states"][1], dict)
    assert item["states"][1]["cache"][1] is None
    assert item["name"] == "sample"
    assert tensor.device.type == "cpu"
    assert nested["states"][0] is tensor


def test_plain_tensor_and_nontensor_metadata_remain_supported():
    tensor = torch.ones(2)
    moved = _move_to_device(tensor, torch.device("meta"))
    assert moved.device.type == "meta"
    metadata = {"name": "sample", "count": 2, "optional": None}
    assert _move_to_device(metadata, torch.device("cpu")) == metadata


@pytest.mark.parametrize("mapping", MAPPINGS)
def test_shared_transfer_handles_mapping_in_args_and_kwargs(mapping):
    tensor = torch.ones(2)
    batch = mapping({"value": [mapping({"tensor": tensor})], "tag": "sample"})
    args, kwargs = move_batch_to_device(
        (batch,), {"batch": batch}, torch.device("meta")
    )
    for moved in (args[0], kwargs["batch"]):
        assert isinstance(moved, dict)
        assert moved["value"][0]["tensor"].device.type == "meta"
        assert moved["tag"] == "sample"
    assert tensor.device.type == "cpu"


class BatchInspected(RuntimeError):
    """Stop after inspecting a genuine meta transfer, before numeric training."""


@pytest.mark.parametrize("mapping", MAPPINGS)
def test_public_finetune_transfers_mapping_before_calling_loss(mapping):
    model = nn.Linear(2, 1)
    model.eval()
    tensor = torch.ones(2, 2)
    batch = mapping({"inputs": [UserDict({"tensor": tensor})], "name": "sample"})

    def inspect_loss(_model, moved):
        assert isinstance(moved, dict)
        assert moved["inputs"][0]["tensor"].device.type == "meta"
        assert moved["name"] == "sample"
        raise BatchInspected

    with pytest.raises(BatchInspected):
        prune_finetune(model, [batch], steps=1, device="meta", loss_fn=inspect_loss)
    assert not model.training
    assert batch["inputs"][0]["tensor"] is tensor
    assert tensor.device.type == "cpu"


@pytest.mark.parametrize("mapping", MAPPINGS)
def test_public_finetune_trains_with_nested_mapping_on_cpu(mapping):
    torch.manual_seed(7)
    model = nn.Linear(2, 1)
    before = model.weight.detach().clone()
    batch = mapping({"inputs": UserDict({"tensor": torch.ones(2, 2)})})

    def loss_fn(current, moved):
        assert isinstance(moved, dict)
        return current(moved["inputs"]["tensor"]).square().mean()

    result = prune_finetune(model, [batch], steps=2, device="cpu", loss_fn=loss_fn)
    assert result["steps"] == 2
    assert all(torch.isfinite(torch.tensor(result["loss_history"])))
    assert not torch.equal(before, model.weight)
