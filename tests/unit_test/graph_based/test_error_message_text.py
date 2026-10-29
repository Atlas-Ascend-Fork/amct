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

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------

import pytest
import torch

from amct_pytorch.classic.graph_based.amct_pytorch.custom_op.arq.arq import (
    weight_cali_tensor,
)
from amct_pytorch.classic.graph_based.amct_pytorch.custom_op.recorder.recorder import (
    Recorder,
)
from amct_pytorch.classic.graph_based.amct_pytorch.utils import output_shape


def test_arq_weight_cali_tensor_rejects_unknown_wts_algo():
    weight = torch.randn(2, 2)
    with pytest.raises(RuntimeError, match="Unsupported wts_algo nonexistent"):
        weight_cali_tensor(weight, {"wts_algo": "nonexistent"})


def test_recorder_forward_rejects_unknown_factors_type(tmp_path):
    recorder = Recorder(str(tmp_path / "record.txt"))
    with pytest.raises(ValueError, match="unsupported factors_type not_a_type"):
        recorder.forward("layer_x", "not_a_type", {"scale_d": 1.0, "offset_d": 0})


def test_output_shape_rejects_unknown_module_type():
    with pytest.raises(TypeError, match="42 is unknown type"):
        output_shape.OutputShape(42)


def test_output_shape_reports_isinstance_dispatch_failure(monkeypatch):
    # isinstance(x, None) 本身抛 TypeError，触发 except 分支拼接的报错文案。
    monkeypatch.setattr(
        output_shape, "COMPUTE_OP_TYPES", [(None, lambda module, shape: None)]
    )
    sentinel = object()
    with pytest.raises(TypeError, match="is unknown type"):
        output_shape.OutputShape(sentinel)
