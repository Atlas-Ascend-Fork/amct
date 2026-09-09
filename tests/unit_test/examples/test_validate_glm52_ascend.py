# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

import json
import runpy
from pathlib import Path

import pytest


@pytest.fixture
def client():
    return runpy.run_path(
        str(Path(__file__).resolve().parents[3] / "examples/validate_glm5_2_ascend.py")
    )


def response(count=1):
    return {
        "choices": [
            {"index": i, "text": "Hello", "finish_reason": "length"}
            for i in range(count)
        ],
        "usage": {
            "prompt_tokens": 5,
            "completion_tokens": count,
            "total_tokens": 5 + count,
        },
    }


@pytest.mark.parametrize(
    "fault", ["empty", "choices", "index", "usage", "total", "bool"]
)
def test_response_validation_rejects_bad_responses(client, fault):
    data = response()
    if fault == "empty":
        data["choices"][0]["text"] = " "
    elif fault == "choices":
        data["choices"] = []
    elif fault == "index":
        data["choices"][0]["index"] = 2
    elif fault == "usage":
        del data["usage"]
    elif fault == "total":
        data["usage"]["total_tokens"] = 2
    else:
        data["usage"]["completion_tokens"] = True
    with pytest.raises(ValueError):
        client["validate_response"](data, 1)


def test_requests_and_evidence_use_single_and_batch(client, tmp_path, monkeypatch):
    import urllib.request

    calls = []

    class Reply:
        status = 200

        def __init__(self, data):
            self.data = data

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self):
            return json.dumps(self.data).encode()

    def urlopen(request, timeout):
        data = json.loads(request.data)
        calls.append(data)
        assert request.full_url == "http://localhost:8000/v1/completions"
        assert timeout > 0
        return Reply(response(2 if isinstance(data["prompt"], list) else 1))

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    assert (
        client["main"](
            [
                "--base-url",
                "http://localhost:8000/v1",
                "--mtp-mode",
                "on",
                "--output-dir",
                str(tmp_path),
            ]
        )
        == 0
    )
    assert len(calls) == 2
    assert isinstance(calls[0]["prompt"], str) and len(calls[1]["prompt"]) == 2
    assert all(c["max_tokens"] == 32 and c["temperature"] == 0 for c in calls)
    output = tmp_path / "mtp_on"
    report = json.loads((output / "acceptance.json").read_text())
    assert report["http_status"] == "passed"
    assert report["deployment_status"] == "pending_log_review"
    assert report["mtp_mode_declared"] == "on"
    assert (output / "batch.response.json").exists()
    assert (output / "deployment_acceptance.md").exists()
    with pytest.raises(ValueError, match="empty"):
        client["main"](["--mtp-mode", "on", "--output-dir", str(tmp_path)])


def test_failed_request_keeps_failed_evidence(client, tmp_path, monkeypatch):
    import urllib.error
    import urllib.request

    def fail(*args, **kwargs):
        raise urllib.error.URLError("offline")

    monkeypatch.setattr(urllib.request, "urlopen", fail)
    assert client["main"](["--mtp-mode", "off", "--output-dir", str(tmp_path)]) == 1
    report = json.loads((tmp_path / "mtp_off/acceptance.json").read_text())
    assert report["http_status"] == "failed"
    assert report["deployment_status"] != "passed"


def test_export_shell_forwards_paths_and_cpu_options(tmp_path):
    import os
    import subprocess
    import sys

    capture = tmp_path / "arguments.json"
    launcher = tmp_path / "python launcher"
    launcher.write_text(
        f"#!{sys.executable}\nimport json, os, sys\n"
        "open(os.environ['CAPTURE_ARGS'], 'w').write(json.dumps(sys.argv[1:]))\n"
    )
    launcher.chmod(0o755)
    script = Path(__file__).resolve().parents[3] / "examples/deploy_glm5_2_ascend.sh"
    env = dict(
        os.environ,
        PYTHON_BIN=str(launcher),
        CAPTURE_ARGS=str(capture),
        GLM52_BF16_DIR=str(tmp_path / "BF16 model"),
        GLM52_ASCEND_DIR=str(tmp_path / "INT8 model"),
    )
    subprocess.run(["bash", str(script)], cwd=tmp_path, env=env, check=True)
    args = json.loads(capture.read_text())
    assert args[args.index("--model") + 1] == env["GLM52_BF16_DIR"]
    assert args[args.index("--output_dir") + 1] == env["GLM52_ASCEND_DIR"]
    assert args[args.index("--device") + 1] == "cpu"
    assert args[args.index("--deploy_format") + 1] == "ascend"
    assert Path(args[args.index("--quant_layers_config") + 1]).is_file()
