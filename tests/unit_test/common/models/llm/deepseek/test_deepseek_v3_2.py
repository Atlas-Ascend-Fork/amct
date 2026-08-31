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

from types import SimpleNamespace

import pytest

from amct_pytorch.common.models.llm.deepseek.deepseek_v3_2.deepseekv3_2 import (
    DeepseekV32,
)


def test_init_requires_trust_remote_code():
    args = SimpleNamespace(
        model_name="deepseek_v3_2",
        trust_remote_code=False,
    )

    with pytest.raises(
        ValueError,
        match="deepseek_v3_2 requires --trust_remote_code",
    ):
        DeepseekV32(args)
