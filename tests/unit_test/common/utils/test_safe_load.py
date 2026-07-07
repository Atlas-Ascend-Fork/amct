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
import logging

from amct_pytorch.common.utils import safe_load


def test_safe_torch_load_defaults_to_weights_only(monkeypatch):
    captured_kwargs = {}

    def fake_load(path, *, weights_only=None, **kwargs):
        captured_kwargs["weights_only"] = weights_only
        captured_kwargs.update(kwargs)
        return {"path": path}

    monkeypatch.setattr(safe_load.torch, "load", fake_load)

    assert safe_load.safe_torch_load("params.pth") == {"path": "params.pth"}
    assert captured_kwargs["weights_only"] is True


def test_safe_torch_load_preserves_explicit_weights_only(monkeypatch):
    captured_kwargs = {}

    def fake_load(path, *, weights_only=None, **kwargs):
        captured_kwargs["weights_only"] = weights_only
        captured_kwargs.update(kwargs)
        return {"path": path}

    monkeypatch.setattr(safe_load.torch, "load", fake_load)

    safe_load.safe_torch_load("params.pth", weights_only=False)
    assert captured_kwargs["weights_only"] is False


def test_safe_torch_load_falls_back_when_weights_only_unsupported(monkeypatch, caplog):
    captured = {}

    def fake_load_without_weights_only(path, **kwargs):
        captured["path"] = path
        captured["kwargs"] = kwargs
        return {"path": path}

    monkeypatch.setattr(safe_load.torch, "load", fake_load_without_weights_only)

    with caplog.at_level(logging.WARNING):
        result = safe_load.safe_torch_load("params.pth", weights_only=True)

    assert result == {"path": "params.pth"}
    # weights_only must be stripped so the legacy torch.load does not raise.
    assert "weights_only" not in captured["kwargs"]
    assert any("weights_only" in record.message for record in caplog.records)
