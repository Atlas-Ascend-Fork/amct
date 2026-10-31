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
"""Recorder.forward 在非法 factors_type 时的入参校验单测。"""

import pytest

from amct_pytorch.classic.graph_based.amct_pytorch.custom_op.recorder.recorder import (
    Recorder,
)


@pytest.mark.parametrize("bad_type", ["unknown", "", "ifmr ", "IFMR", "kv-cache"])
def test_forward_rejects_unknown_factors_type(tmp_path, bad_type):
    recorder = Recorder(str(tmp_path / "record.txt"))
    with pytest.raises(ValueError, match="unsupported factors_type"):
        recorder.forward("layer.0", bad_type, {"scale_d": 1.0, "offset_d": 0})
