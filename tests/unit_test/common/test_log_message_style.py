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
import ast
import re
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
FORBIDDEN_FAILURE_WORDS = re.compile(
    r"\b(fail(?:ed|ure|s|ing)?|error(?:s)?|exception(?:s)?|abnormal)\b", re.I
)
KNOWN_LOG_TYPOS = re.compile(
    r"\b(fot|fuison|pune|paramaters|otputs|reesult|unknow)\b"
    r"|\bnot find self_attn\.rotary_emb\.inv_freq\b"
    r"|\bhas not clip_factor_[aw]_max\b"
    r"|`use_cache=False`transformers\b"
)


def _tracked_python_files():
    result = subprocess.run(
        ["git", "ls-files", "*.py"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return [REPO_ROOT / file_path for file_path in result.stdout.splitlines()]


def _string_expr_text(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        return "".join(
            part.value
            if isinstance(part, ast.Constant) and isinstance(part.value, str)
            else "{}"
            for part in node.values
        )
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _string_expr_text(node.left)
        right = _string_expr_text(node.right)
        if left is not None and right is not None:
            return left + right
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        if node.func.attr == "format":
            return _string_expr_text(node.func.value)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mod):
        return _string_expr_text(node.left)
    return None


def _iter_calls():
    for file_path in _tracked_python_files():
        tree = ast.parse(file_path.read_text(encoding="utf-8"), filename=str(file_path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                yield file_path, node


def _format_violation(file_path, node, message):
    return f"{file_path.relative_to(REPO_ROOT)}:{node.lineno}: {message}"


def test_logger_messages_use_failure_words_only_for_errors():
    violations = []
    for file_path, node in _iter_calls():
        func = node.func
        if not (
            isinstance(func, ast.Attribute)
            and isinstance(func.value, ast.Name)
            and func.value.id == "LOGGER"
        ):
            continue
        if func.attr == "loge" or not node.args:
            continue

        message = _string_expr_text(node.args[0])
        if message is not None and FORBIDDEN_FAILURE_WORDS.search(message):
            violations.append(
                _format_violation(file_path, node, f"LOGGER.{func.attr}: {message}")
            )

    assert not violations, "\n".join(violations)


def test_log_messages_have_no_known_typos():
    violations = []
    for file_path, node in _iter_calls():
        func = node.func
        is_logger = (
            isinstance(func, ast.Attribute)
            and isinstance(func.value, ast.Name)
            and func.value.id in {"LOGGER", "logger", "logging"}
        )
        is_runtime_error = isinstance(func, ast.Name) and func.id == "RuntimeError"
        if not (is_logger or is_runtime_error) or not node.args:
            continue

        message = _string_expr_text(node.args[0])
        if message is not None and KNOWN_LOG_TYPOS.search(message):
            violations.append(_format_violation(file_path, node, message))

    assert not violations, "\n".join(violations)


def test_log_message_format_calls_are_spelled_correctly():
    violations = []
    for file_path, node in _iter_calls():
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr == "foramt":
            violations.append(
                _format_violation(file_path, node, "use .format(...) instead")
            )

    assert not violations, "\n".join(violations)
