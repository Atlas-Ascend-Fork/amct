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

import pytest

from amct_pytorch.common.utils.log import (
    LOG_FILE_SET_ENV,
    LOG_SET_ENV,
    LOGGER,
    Logger,
    LoggerBase,
    check_level,
    log_split_deco,
)


def test_check_level_accepts_valid_levels():
    for level in ("DEBUG", "INFO", "WARNING", "ERROR", "debug", "info"):
        check_level(level, "test_level")


def test_check_level_rejects_invalid_level():
    with pytest.raises(ValueError, match="is invalid"):
        check_level("INVALID", "bad_level")


def test_log_split_deco_splits_long_message_via_kwargs():
    @log_split_deco(length=260)
    def fake_log(self, message="default", module_name="AMCT"):
        return message

    msg = fake_log(None, message="a" * 300)
    assert isinstance(msg, list)
    assert len(msg) == 2


def test_log_split_deco_shorter_than_length_returns_single_element():
    @log_split_deco(length=260)
    def fake_log(self, message="default", module_name="AMCT"):
        return message

    msg = fake_log(None, message="short")
    assert isinstance(msg, list)
    assert len(msg) == 1
    assert msg[0] == "short"


def test_log_split_deco_plain_function_preserves_module_and_extra_arguments():
    @log_split_deco(length=5)
    def plain(message, module_name="AMCT", *extra):
        return message, module_name, extra

    assert plain("a" * 11, "custom", 7) == (["aaaaa", "aaaaa", "a"], "custom", (7,))
    assert plain(module_name="custom", message="short") == (["short"], "custom", ())


def test_log_split_deco_single_message_parameter():
    @log_split_deco(length=5)
    def plain(message):
        return message

    assert plain("hello") == ["hello"]


def test_log_split_deco_static_and_class_methods():
    class Messages:
        @staticmethod
        @log_split_deco(length=5)
        def static(message, module_name="AMCT"):
            return message, module_name

        @classmethod
        @log_split_deco(length=5)
        def class_method(cls, message, module_name="AMCT"):
            return cls, message, module_name

    assert Messages.static("hello", module_name="custom") == (["hello"], "custom")
    assert Messages.class_method(message="hello") == (Messages, ["hello"], "AMCT")


def test_log_split_deco_keyword_only_message():
    @log_split_deco(length=5)
    def plain(*, message="hello", module_name="AMCT"):
        return message, module_name

    assert plain(module_name="custom") == (["hello"], "custom")


@pytest.mark.parametrize(
    "func",
    [
        lambda: None,
        lambda self: None,
        lambda *args: None,
        lambda **kwargs: None,
        lambda self, *args: None,
    ],
)
def test_log_split_deco_rejects_missing_or_variadic_message_at_decoration(func):
    with pytest.raises(ValueError, match="named message parameter"):
        log_split_deco()(func)


def test_logger_base_is_file_debug_level_true_when_file_handler_at_debug():
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        lb = LoggerBase(tmpdir, "test.log")
        lb.file_handler.setLevel(logging.DEBUG)
        assert lb.is_file_debug_level() is True


def test_logger_base_is_file_debug_level_false_when_file_handler_at_info():
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        lb = LoggerBase(tmpdir, "test.log")
        lb.file_handler.setLevel(logging.INFO)
        assert lb.is_file_debug_level() is False


def test_logger_set_debug_level_valid():
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        lb = LoggerBase(tmpdir, "test.log")
        lb.set_debug_level("warning", "error")
        assert lb.console_handler.level == logging.WARNING
        assert lb.file_handler.level == logging.ERROR


def test_logger_set_debug_level_rejects_bad_print_level():
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        lb = LoggerBase(tmpdir, "test.log")
        with pytest.raises(ValueError, match="is invalid"):
            lb.set_debug_level("bad", "info")


def test_logger_set_debug_level_rejects_bad_save_level():
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        lb = LoggerBase(tmpdir, "test.log")
        with pytest.raises(ValueError, match="is invalid"):
            lb.set_debug_level("info", "very_bad")


def test_logger_logi_writes_info_message():
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        lb = LoggerBase(tmpdir, "test.log")
        lb.logi("hello info")


def test_logger_loge_writes_error_message():
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        lb = LoggerBase(tmpdir, "test.log")
        lb.loge("fatal error")


def test_logger_init_uses_env_print_level(monkeypatch):
    import tempfile

    monkeypatch.setenv(LOG_SET_ENV, "DEBUG")
    monkeypatch.setenv(LOG_FILE_SET_ENV, "ERROR")
    with tempfile.TemporaryDirectory() as tmpdir:
        log = Logger(tmpdir, "test_env.log")
        assert log.console_handler.level == logging.DEBUG
        assert log.file_handler.level == logging.ERROR


def test_logger_init_rejects_bad_env_print_level(monkeypatch):
    import tempfile

    monkeypatch.setenv(LOG_SET_ENV, "GARBAGE")
    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(ValueError, match="is invalid"):
            Logger(tmpdir, "test_bad_env.log")


def test_logger_init_rejects_bad_env_file_level(monkeypatch):
    import tempfile

    monkeypatch.setenv(LOG_SET_ENV, "INFO")
    monkeypatch.setenv(LOG_FILE_SET_ENV, "GARBAGE")
    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(ValueError, match="is invalid"):
            Logger(tmpdir, "test_bad_file_env.log")


def test_logger_init_defaults_when_no_env(monkeypatch):
    import tempfile

    monkeypatch.delenv(LOG_SET_ENV, raising=False)
    monkeypatch.delenv(LOG_FILE_SET_ENV, raising=False)
    with tempfile.TemporaryDirectory() as tmpdir:
        log = Logger(tmpdir, "test_no_env.log")
        assert log.console_handler.level == logging.INFO
        assert log.file_handler.level == logging.INFO


def test_logger_logw_writes_warning_message():
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        lb = LoggerBase(tmpdir, "test.log")
        lb.logw("this is a warning")


def test_logger_is_instantiated():
    assert LOGGER is not None
    assert isinstance(LOGGER, Logger)


@pytest.fixture
def file_logger(tmp_path):
    """Own and close handlers before temporary log-directory cleanup."""
    logger = LoggerBase(str(tmp_path), "keyword-message.log")
    logger.set_debug_level("error", "debug")
    yield logger, tmp_path / "keyword-message.log"
    for handler in (logger.console_handler, logger.file_handler):
        logger.logger.removeHandler(handler)
        handler.close()


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
def test_logger_keyword_order_preserves_complete_message(
    file_logger, method, message_name, module_first
):
    logger, path = file_logger
    items = [(message_name, "hello world"), ("module_name", "CLI")]
    if module_first:
        items.reverse()
    getattr(logger, method)(**dict(items))
    logger.file_handler.flush()
    messages = [line.split("[AMCT]:", 1)[1] for line in path.read_text().splitlines()]
    assert messages == ["[CLI]: hello world"]


@pytest.mark.parametrize("module_first", [False, True])
def test_logger_keyword_order_preserves_long_message_chunks(file_logger, module_first):
    logger, path = file_logger
    items = [("info_message", "a" * 550), ("module_name", "CLI")]
    if module_first:
        items.reverse()
    logger.logi(**dict(items))
    logger.file_handler.flush()
    messages = [line.split("[AMCT]:", 1)[1] for line in path.read_text().splitlines()]
    assert messages == ["[CLI]: " + "a" * 500, "[CLI]: " + "a" * 50]


@pytest.mark.parametrize("method", ["logd", "logi", "logw", "loge"])
def test_logger_positional_message_and_module_compatibility(file_logger, method):
    logger, path = file_logger
    getattr(logger, method)("hello world", "CLI")
    logger.file_handler.flush()
    messages = [line.split("[AMCT]:", 1)[1] for line in path.read_text().splitlines()]
    assert messages == ["[CLI]: hello world"]
