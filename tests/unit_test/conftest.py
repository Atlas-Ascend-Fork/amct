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
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

# torchao is an optional runtime dependency (MX FP8 tensorwise path).
# Stub it out so unit tests that don't exercise _run_tensorwise can still import
# llm_deploy and mxfp_impl without a full torchao installation.
# Python 3.11+ find_spec raises ValueError when __spec__ is None or missing,
# so we must attach a real ModuleSpec to each stub entry.
if "torchao" not in sys.modules:
    import importlib.machinery

    def _make_stub(name):
        mod = types.ModuleType(name)
        mod.__spec__ = importlib.machinery.ModuleSpec(name, loader=None, origin="<stub>")
        mod.__path__ = []
        return mod

    _torchao_stub = _make_stub("torchao")
    _torchao_proto_stub = _make_stub("torchao.prototype")
    _torchao_utils_stub = _make_stub("torchao.prototype.custom_fp_utils")
    # Provide the symbols that mxfp_impl.py imports from this sub-module.
    for _attr in ("_f32_to_floatx_unpacked", "_floatx_unpacked_to_f32",
                  "fp8_blockwise_act_quant", "fp8_blockwise_weight_quant",
                  "fp4_blockwise_act_quant", "triton_mm_fp8"):
        setattr(_torchao_utils_stub, _attr, MagicMock())
    _torchao_stub.prototype = _torchao_proto_stub
    _torchao_proto_stub.custom_fp_utils = _torchao_utils_stub
    sys.modules.setdefault("torchao", _torchao_stub)
    sys.modules.setdefault("torchao.prototype", _torchao_proto_stub)
    sys.modules.setdefault("torchao.prototype.custom_fp_utils", _torchao_utils_stub)

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
