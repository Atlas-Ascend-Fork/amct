#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
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
"""回归：pytest 会话下 TORCH_DEVICE_BACKEND_AUTOLOAD 必须为 "0"。

原先该变量依赖 pyproject [tool.pytest.ini_options] 的 env 选项设置，但那是
pytest-env 插件的选项、本仓未声明该插件，配置被静默忽略，导致 tests/conftest.py
的 `import torch` 在装有 torch_npu 的机器上会触发真实后端自动加载。现由
tests/conftest.py 在 import torch 之前 os.environ.setdefault 兜底，本用例锁定
该保证不被回退。
"""
import os


def test_torch_backend_autoload_disabled():
    assert os.environ.get("TORCH_DEVICE_BACKEND_AUTOLOAD") == "0", (
        "tests/conftest.py 应在 import torch 之前将 "
        "TORCH_DEVICE_BACKEND_AUTOLOAD 设为 '0'"
    )
