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

from amct_pytorch.common.utils import safe_load


def test_safe_torch_load_defaults_to_weights_only(monkeypatch):
    captured_kwargs = {}

    def fake_load(path, **kwargs):
        captured_kwargs.update(kwargs)
        return {"path": path}

    monkeypatch.setattr(safe_load.torch, "load", fake_load)

    assert safe_load.safe_torch_load("params.pth") == {"path": "params.pth"}
    assert captured_kwargs["weights_only"] is True


def test_safe_torch_load_preserves_explicit_weights_only(monkeypatch):
    captured_kwargs = {}

    def fake_load(path, **kwargs):
        captured_kwargs.update(kwargs)
        return {"path": path}

    monkeypatch.setattr(safe_load.torch, "load", fake_load)

    safe_load.safe_torch_load("params.pth", weights_only=False)
    assert captured_kwargs["weights_only"] is False


def test_safe_torch_load_rejects_unsupported_torch_version(monkeypatch):
    monkeypatch.setattr(safe_load.torch, "__version__", "1.12.1")

    with pytest.raises(RuntimeError, match="requires PyTorch 1.13.0 or later"):
        safe_load.safe_torch_load("params.pth")
