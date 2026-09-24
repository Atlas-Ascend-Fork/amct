#!/usr/bin/env python3
# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
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

"""Capture HTTP smoke-test evidence from an already running vLLM service."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import urllib.error
import urllib.request


PROMPTS = ["The capital of France is", "One plus one equals"]


def validate_response(data: dict, count: int) -> None:
    """Require one nonempty completion per prompt and consistent token usage."""
    if not isinstance(data, dict):
        raise ValueError("Response must be a JSON object")
    choices = data.get("choices")
    if not isinstance(choices, list) or len(choices) != count:
        raise ValueError("Unexpected completion choices count")
    indexes = []
    for choice in choices:
        if (
            not isinstance(choice, dict)
            or not isinstance(choice.get("text"), str)
            or not choice["text"].strip()
        ):
            raise ValueError("Completion text must be nonempty")
        if type(choice.get("index")) is not int:
            raise ValueError("Choice index must be an integer")
        indexes.append(choice["index"])
    if sorted(indexes) != list(range(count)):
        raise ValueError("Missing or duplicate choice indexes")
    usage = data.get("usage")
    if not isinstance(usage, dict) or any(
        type(usage.get(k)) is not int or usage[k] < 0
        for k in ("prompt_tokens", "completion_tokens", "total_tokens")
    ):
        raise ValueError("Invalid token usage")
    if (
        usage["completion_tokens"] == 0
        or usage["total_tokens"] != usage["prompt_tokens"] + usage["completion_tokens"]
    ):
        raise ValueError("Inconsistent token usage")


def write_json(path: Path, data: dict) -> None:
    temporary = path.with_name("." + path.name + ".tmp")
    temporary.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000/v1",
        help="OpenAI API base URL including /v1",
    )
    parser.add_argument("--model", default="glm52-amct")
    parser.add_argument(
        "--mtp-mode",
        required=True,
        choices=("off", "on"),
        help="Declared server mode; must be verified separately in server logs",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--timeout", type=float, default=120)
    args = parser.parse_args(argv)
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    directory = args.output_dir / ("mtp_" + args.mtp_mode)
    if directory.exists() and (not directory.is_dir() or any(directory.iterdir())):
        raise ValueError("Evidence directory must be new or empty")
    directory.mkdir(parents=True, exist_ok=True)
    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "base_url": args.base_url,
        "model": args.model,
        "mtp_mode_declared": args.mtp_mode,
        "http_status": "failed",
        "deployment_status": "pending_log_review",
        "cases": [],
    }
    endpoint = args.base_url.rstrip("/") + "/completions"
    headers = {"Content-Type": "application/json"}
    if os.environ.get("OPENAI_API_KEY"):
        headers["Authorization"] = "Bearer " + os.environ["OPENAI_API_KEY"]
    try:
        for name, prompt, count in (("single", PROMPTS[0], 1), ("batch", PROMPTS, 2)):
            payload = {
                "model": args.model,
                "prompt": prompt,
                "temperature": 0,
                "max_tokens": 32,
                "stream": False,
            }
            write_json(directory / f"{name}.request.json", payload)
            request = urllib.request.Request(
                endpoint,
                data=json.dumps(payload).encode(),
                headers=headers,
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=args.timeout) as response:
                raw = response.read()
                (directory / f"{name}.response.json").write_bytes(raw)
                if not 200 <= response.status < 300:
                    raise ValueError(f"HTTP status {response.status}")
                data = json.loads(raw)
            validate_response(data, count)
            report["cases"].append(
                {"name": name, "status": "passed", "usage": data["usage"]}
            )
        report["http_status"] = "passed"
    except (urllib.error.URLError, OSError, ValueError) as exc:
        report["error"] = str(exc)
        report["deployment_status"] = "not_accepted"
        if isinstance(exc, urllib.error.HTTPError):
            (directory / "http_error.response.txt").write_bytes(exc.read())
    write_json(directory / "acceptance.json", report)
    (directory / "deployment_acceptance.md").write_text(
        "# GLM-5.2 deployment evidence\n\n"
        f"- Declared MTP mode: {args.mtp_mode}\n"
        f"- HTTP smoke test: {report['http_status']}\n"
        f"- Deployment acceptance: {report['deployment_status']}\n\n"
        "The mode is supplied by the operator, not detected through HTTP.\n\n"
        "- [ ] Record hardware, CANN, torch, vLLM and vllm-ascend versions.\n"
        "- [ ] Attach exact startup command and server loading logs.\n"
        "- [ ] Check dynamic Linear/MoE implementation and any MLAPO fallback.\n"
        "- [ ] Verify declared MTP mode in logs; for on, confirm FLOAT MTP loading and speculative execution.\n"
        "- [ ] Confirm no missing weights or rotation/shape errors.\n"
        "- [ ] Review both MTP off/on evidence before accepting deployment.\n",
        encoding="utf-8",
    )
    print(f"HTTP smoke test: {report['http_status']}; evidence: {directory}")
    return 0 if report["http_status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
