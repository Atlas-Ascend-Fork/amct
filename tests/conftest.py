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
import os
import sys
from unittest.mock import MagicMock

# torch 在 import 时会依据 TORCH_DEVICE_BACKEND_AUTOLOAD 决定是否自动加载
# torch_npu 后端。测试用 MagicMock 顶替 torch_npu，必须在 import torch 之前
# 关闭自动加载，否则装有 torch_npu 的机器上 import torch 会先触发真实后端加载
# （可能在收集阶段崩溃）。原先依赖 pyproject [tool.pytest.ini_options] 的 env
# 设置该变量，但那是 pytest-env 插件的选项、本仓未声明该插件，配置被静默忽略
# （Unknown config option: env）。这里在进程内兜底设置，与 build.sh /
# run_smoke.sh 中的显式 export 保持一致，且不依赖任何插件。
os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
import torch

_mock_npu = MagicMock()
_mock_npu.__spec__ = MagicMock()
sys.modules.setdefault("torch_npu", _mock_npu)

if not hasattr(torch, "npu"):
    torch.npu = MagicMock()
