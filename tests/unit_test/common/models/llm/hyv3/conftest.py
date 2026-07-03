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
"""Shared fixtures for hyv3 unit tests.

``transformers.models.hy_v3`` is only available since transformers 5.12.1.
When the installed transformers is older, inject a lightweight mock into
``sys.modules`` so that collection of the whole test suite does not break.
Tests that require the real ``HYV3Attention`` / ``HYV3MoE`` modules skip
themselves via the ``HY_V3_MOCKED`` flag set here.
"""
import importlib
import os
import sys
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn


def _build_hy_v3_mock():
    mod = MagicMock()
    mod.HYV3Config = type("HYV3Config", (), {})
    mod.HYV3DecoderLayer = type("HYV3DecoderLayer", (), {})

    def _hyv3_attn_init(self, config, layer_idx):
        nn.Module.__init__(self)
        self.head_dim = config.head_dim
        self.attention_dropout = getattr(config, "attention_dropout", 0.0)
        self.scaling = getattr(config, "scaling", 1.0)
        nq = getattr(config, "num_attention_heads", 1)
        nkv = getattr(config, "num_key_value_heads", nq)
        self.num_key_value_groups = nq // nkv

    mod.HYV3Attention = type(
        "HYV3Attention", (nn.Module,), {"__init__": _hyv3_attn_init}
    )
    mod.HYV3MoE = type(
        "HYV3MoE",
        (nn.Module,),
        {"__init__": lambda self, *a, **kw: nn.Module.__init__(self)},
    )
    mod.apply_rotary_pos_emb = lambda *a, **kw: a[:2]
    return mod


def pytest_configure(config):
    try:
        importlib.import_module("transformers.models.hy_v3.modeling_hy_v3")
        return
    except ImportError:
        pass
    mod = _build_hy_v3_mock()
    sys.modules.setdefault("transformers.models.hy_v3", mod)
    sys.modules.setdefault("transformers.models.hy_v3.modeling_hy_v3", mod)
    os.environ["HY_V3_MOCKED"] = "1"


def pytest_collection_modifyitems(config, items):
    """Skip tests that require a real hy_v3 module when it is mocked."""
    if os.environ.get("HY_V3_MOCKED") != "1":
        return
    skip_marker = pytest.mark.skip(
        reason="transformers lacks real hy_v3 module; needs >=5.12.1"
    )
    for item in items:
        if "test_hyv3_quant_module" in item.nodeid or "TestHyV3Mocked" in item.nodeid:
            item.add_marker(skip_marker)
