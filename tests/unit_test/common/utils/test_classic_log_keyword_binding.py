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
"""Both classic import paths preserve keyword messages in real file handlers."""

import importlib.util
from pathlib import Path
import sys
from types import ModuleType

import pytest


@pytest.fixture(params=["common", "amct_pytorch/common"])
def classic_logger(request, tmp_path, monkeypatch):
    relative = Path("amct_pytorch/classic/graph_based")
    root = next(
        parent
        for parent in Path(__file__).resolve().parents
        if (parent / relative / "common/utils/log_base.py").is_file()
    )
    graph = root / relative
    common = graph / request.param
    if common.is_file():
        # Windows Git checkouts represent tracked directory symlinks as text.
        assert common.read_text(encoding="utf-8").strip() == "../common"
        common = (common.parent / "../common").resolve()
    package_name = "_classic_keyword_test"
    package = ModuleType(package_name)
    package.__path__ = [str(common / "utils")]
    monkeypatch.setitem(sys.modules, package_name, package)
    name = package_name + ".log_base"
    spec = importlib.util.spec_from_file_location(name, common / "utils/log_base.py")
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    logger = module.LoggerBase(str(tmp_path), "classic.log")
    logger.set_debug_level("debug", "debug")
    yield logger, tmp_path / "classic.log"
    for handler in (logger.console_handler, logger.file_handler):
        logger.logger.removeHandler(handler)
        handler.close()
    monkeypatch.delitem(sys.modules, package_name + ".util", raising=False)


def messages(logger, path):
    logger.file_handler.flush()
    return [line.split("[AMCT]:", 1)[1] for line in path.read_text().splitlines()]


@pytest.mark.parametrize(
    "method,message_name",
    [
        ("logd", "debug_message"),
        ("logi", "info_message"),
        ("logw", "warning_message"),
        ("loge", "error_message"),
    ],
)
@pytest.mark.parametrize("module_first", [False, True])
def test_classic_keyword_message_order(
    classic_logger, method, message_name, module_first
):
    logger, path = classic_logger
    items = [(message_name, "hello world"), ("module_name", "CLI")]
    if module_first:
        items.reverse()
    getattr(logger, method)(**dict(items))
    assert messages(logger, path) == ["[CLI]: hello world"]


@pytest.mark.parametrize("module_first", [False, True])
def test_classic_long_keyword_message_chunks(classic_logger, module_first):
    logger, path = classic_logger
    items = [("info_message", "a" * 550), ("module_name", "CLI")]
    if module_first:
        items.reverse()
    logger.logi(**dict(items))
    assert messages(logger, path) == ["[CLI]: " + "a" * 500, "[CLI]: " + "a" * 50]


@pytest.mark.parametrize("method", ["logd", "logi", "logw", "loge"])
def test_classic_positional_messages_remain_supported(classic_logger, method):
    logger, path = classic_logger
    getattr(logger, method)("hello world", "CLI")
    assert messages(logger, path) == ["[CLI]: hello world"]
