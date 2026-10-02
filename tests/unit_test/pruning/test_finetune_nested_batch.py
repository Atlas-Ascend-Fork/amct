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
"""Nested fine-tuning batches must transfer every tensor leaf."""

import pytest
import torch

from amct_pytorch.pruning.finetune import _move_to_device


@pytest.mark.parametrize("container", [dict, list, tuple])
def test_nested_tensor_leaves_are_moved_without_accelerator(container):
    tensor = torch.ones(2)
    nested = {"states": [tensor, {"cache": (tensor, None)}], "name": "sample"}
    batch = nested if container is dict else container([nested])
    moved = _move_to_device(batch, torch.device("meta"))
    item = moved if container is dict else moved[0]
    assert isinstance(moved, container)
    assert item["states"][0].device.type == "meta"
    assert item["states"][1]["cache"][0].device.type == "meta"
    assert item["states"][1]["cache"][1] is None
    assert item["name"] == "sample"
    assert tensor.device.type == "cpu"


def test_plain_tensor_and_nontensor_metadata_remain_supported():
    tensor = torch.ones(2)
    moved = _move_to_device(tensor, torch.device("meta"))
    assert moved.device.type == "meta"
    metadata = {"name": "sample", "count": 2, "optional": None}
    assert _move_to_device(metadata, torch.device("cpu")) == metadata
