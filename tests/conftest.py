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
import glob
import importlib
import os
import sys
from unittest.mock import MagicMock

# Disable torch_npu backend autoload before torch is imported during collection.
os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"

import torch  # noqa: E402

# Importing the real torch_npu registers the NPU dispatch backend and other
# torch integrations, so it must only happen when an Ascend NPU is actually
# attached; on no-device runners (even with torch_npu installed) those side
# effects break tests at aclInit (#310). Ascend NPUs are exposed as
# /dev/davinciN device nodes on hosts and containers.
if glob.glob("/dev/davinci[0-9]*"):
    try:
        importlib.import_module("torch_npu")
    except Exception:
        pass

if "torch_npu" not in sys.modules:
    _mock_npu = MagicMock()
    _mock_npu.__spec__ = MagicMock()
    sys.modules["torch_npu"] = _mock_npu

    if not hasattr(torch, "npu"):
        torch.npu = MagicMock()
