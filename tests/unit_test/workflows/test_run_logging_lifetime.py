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
"""Use real Loguru file sinks to exercise public workflow run lifetimes."""

from types import SimpleNamespace
import importlib

from loguru import logger
import pytest

from amct_pytorch.common.utils.run_logging import setup_run_logging
from amct_pytorch.workflows.llm_extract_ptq_data import LlmExtractPtqDataWorkflow
from amct_pytorch.workflows.llm_ptq import LlmPtqWorkflow

# Import the workflow modules at collection time, before any fixture creates a
# loguru sink: their transitive imports (e.g. compressed_tensors) reconfigure
# the root logger with clear_loggers=True and would silently drop that sink.
pytestmark = pytest.mark.cpu


WORKFLOWS = [
    ("llm_ptq", LlmPtqWorkflow),
    ("llm_extract_ptq_data", LlmExtractPtqDataWorkflow),
]


@pytest.fixture
def tracked_sink(tmp_path):
    args = SimpleNamespace(output_dir=str(tmp_path))
    sink_id, path = setup_run_logging(args, "lifetime")
    try:
        yield sink_id, path
    finally:
        # An unfixed implementation still owns the sink after the assertion.
        try:
            logger.remove(sink_id)
        except ValueError:
            pass


def make_workflow(monkeypatch, entry, sink_id, operation):
    module_name, cls = entry
    module = importlib.import_module("amct_pytorch.workflows." + module_name)
    workflow = cls.__new__(cls)
    workflow.granularity = "block"
    workflow.model_name = "test_model"
    workflow.quant_target = ["mlp"]
    workflow.quant_dtype = "int"
    workflow.bit_policy = SimpleNamespace(summary=lambda: "test policy")
    monkeypatch.setattr(workflow, "setup", lambda: sink_id)
    monkeypatch.setattr(workflow, "_run_blockwise", operation)
    if module_name == "llm_ptq":
        monkeypatch.setattr(
            module, "SOLVER_REGISTRY", SimpleNamespace(get=lambda key: object)
        )
    return module, workflow


@pytest.mark.parametrize("entry", WORKFLOWS)
@pytest.mark.parametrize("error_type", [RuntimeError, KeyboardInterrupt])
def test_run_removes_file_sink_on_failure(monkeypatch, tracked_sink, entry, error_type):
    sink_id, path = tracked_sink
    original = error_type("injected workflow failure")

    def fail(*args):
        logger.info("during failing workflow")
        raise original

    _, workflow = make_workflow(monkeypatch, entry, sink_id, fail)
    with pytest.raises(error_type) as caught:
        workflow.run()
    assert caught.value is original
    logger.info("after workflow exit marker")
    with open(path, encoding="utf-8") as log:
        contents = log.read()
    assert "during failing workflow" in contents
    assert "after workflow exit marker" not in contents


@pytest.mark.parametrize("entry", WORKFLOWS)
def test_success_preserves_return_and_other_sinks(monkeypatch, tracked_sink, entry):
    sink_id, path = tracked_sink
    expected = object()
    _, workflow = make_workflow(monkeypatch, entry, sink_id, lambda *args: expected)
    messages = []
    other_sink = logger.add(lambda message: messages.append(str(message)))
    try:
        actual = workflow.run()
        assert actual is expected
        logger.info("after successful workflow marker")
        assert any(
            "after successful workflow marker" in message for message in messages
        )
        with open(path, encoding="utf-8") as log:
            assert "after successful workflow marker" not in log.read()
    finally:
        logger.remove(other_sink)


def test_registry_failure_removes_file_sink(monkeypatch, tracked_sink):
    sink_id, path = tracked_sink
    original = KeyError("missing solver")
    module, workflow = make_workflow(
        monkeypatch, WORKFLOWS[0], sink_id, lambda *args: None
    )

    def fail(key):
        logger.info("during registry failure")
        raise original

    monkeypatch.setattr(module, "SOLVER_REGISTRY", SimpleNamespace(get=fail))
    with pytest.raises(KeyError) as caught:
        workflow.run()
    assert caught.value is original
    logger.info("after registry failure marker")
    with open(path, encoding="utf-8") as log:
        contents = log.read()
    assert "during registry failure" in contents
    assert "after registry failure marker" not in contents
