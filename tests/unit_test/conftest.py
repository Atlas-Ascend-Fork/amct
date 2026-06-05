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
import random
import shutil
import sys
import types
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
import torch

# ---------------------------------------------------------------------------
# Stub out optional heavy dependencies that are not installed in the CPU CI
# environment.  These stubs must be inserted before any amct_pytorch imports
# are collected so that the package-level __init__.py can resolve them.
# ---------------------------------------------------------------------------

def _make_transformers_stub():
    """Return a minimal sys.modules stub for `transformers` and the
    `transformers.cache_utils` sub-module that deepseekv3_attention_quant_module
    imports at the top level."""
    transformers = types.ModuleType("transformers")
    cache_utils = types.ModuleType("transformers.cache_utils")
    cache_utils.DynamicCache = mock.MagicMock(name="DynamicCache")
    transformers.cache_utils = cache_utils
    sys.modules.setdefault("transformers", transformers)
    sys.modules.setdefault("transformers.cache_utils", cache_utils)


_make_transformers_stub()

SAFETENSORS_TMP_DIR = Path(__file__).resolve().parent / "tmp"


@pytest.fixture(autouse=True, scope="session")
def _safetensors_tmp_cleanup():
    """Create and tear down the shared safetensors mock directory."""
    SAFETENSORS_TMP_DIR.mkdir(parents=True, exist_ok=True)
    yield
    if SAFETENSORS_TMP_DIR.exists():
        shutil.rmtree(SAFETENSORS_TMP_DIR)


@pytest.fixture(autouse=True)
def _deterministic():
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    yield


@pytest.fixture
def cpu_device():
    return torch.device("cpu")
