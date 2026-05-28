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
from types import SimpleNamespace

import pytest


def test_ensure_log_dir_uses_output_dir_when_log_dir_empty(tmp_path, monkeypatch):
    monkeypatch.setattr("os.makedirs", lambda p, exist_ok: None)
    out_dir = str(tmp_path / "output")
    args = SimpleNamespace(output_dir=out_dir, log_dir="")
    from amct_pytorch.common.utils.run_logging import ensure_log_dir
    log_dir = ensure_log_dir(args)
    assert "logs" in log_dir
    assert out_dir in log_dir


def test_ensure_log_dir_uses_explicit_log_dir(tmp_path, monkeypatch):
    monkeypatch.setattr("os.makedirs", lambda p, exist_ok: None)
    log_dir_path = str(tmp_path / "custom_logs")
    args = SimpleNamespace(output_dir="/tmp/fallback", log_dir=log_dir_path)
    from amct_pytorch.common.utils.run_logging import ensure_log_dir
    result = ensure_log_dir(args)
    assert result == log_dir_path
    assert args.log_dir == log_dir_path


def test_ensure_log_dir_creates_directory(tmp_path):
    out_dir = tmp_path / "output"
    out_dir.mkdir()
    args = SimpleNamespace(output_dir=str(out_dir), log_dir="")
    from amct_pytorch.common.utils.run_logging import ensure_log_dir
    log_dir = ensure_log_dir(args)
    import os
    assert os.path.isdir(log_dir)


def test_setup_run_logging_returns_sink_id_and_log_path(monkeypatch, tmp_path):
    monkeypatch.setattr("os.makedirs", lambda p, exist_ok: None)
    monkeypatch.setattr("amct_pytorch.common.utils.run_logging.logger.add",
                        lambda *a, **k: 12345)
    args = SimpleNamespace(output_dir=str(tmp_path / "output"), log_dir="")
    from amct_pytorch.common.utils.run_logging import setup_run_logging
    sink_id, log_path = setup_run_logging(args, "ptq")
    assert sink_id == 12345
    assert log_path.endswith("ptq.log")
