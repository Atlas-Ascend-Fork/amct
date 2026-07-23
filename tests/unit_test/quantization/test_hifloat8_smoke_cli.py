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
import subprocess
import sys
from pathlib import Path


SMOKE_SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "amct_ops"
    / "run_hifloat8_dtype_smoke.py"
)


def test_hifloat8_smoke_help_lists_backend_modes():
    result = subprocess.run(
        [sys.executable, str(SMOKE_SCRIPT), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "--backend" in result.stdout
    assert "native" in result.stdout
    assert "amct_ops" in result.stdout
    assert "auto" in result.stdout
