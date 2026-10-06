# -*- coding: UTF-8 -*-
"""Lightweight self checks for quant-benchmark-eval scripts.

This does not load model weights, allocate NPU memory, download datasets, or run
benchmarks. It verifies script wiring and local config assumptions.
"""
import os
import subprocess
import sys
import tempfile

SKILL_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SKILL_DIR)

from config import DEFAULT_BIT_CONFIG, local_dataset_map, load_dataset_registry  # noqa: E402
from utils import (  # noqa: E402
    ensure_supported_quant_target,
    local_dataset_files,
    read_dataset_file,
    validate_lcb_docs,
)

EXPECTED_TASKS = {"mmlu_pro", "math500", "livecodebench", "livecodebench_release_v6"}
CLI_SCRIPTS = [
    "prefetch_data.py",
    "assemble_and_eval.py",
    "livecodebench_eval.py",
    "check_env.py",
    "compare_results.py",
]


def _check_registry():
    reg = load_dataset_registry()
    missing = sorted(EXPECTED_TASKS.difference(reg))
    if missing:
        raise AssertionError(f"datasets.json missing tasks: {missing}")
    mapping = local_dataset_map()
    for task in sorted(EXPECTED_TASKS):
        files = local_dataset_files(task, reg, mapping, existing_only=False)
        if not files:
            raise AssertionError(f"task {task} resolved to no files")
    if not DEFAULT_BIT_CONFIG.endswith(os.path.join("amct_pytorch", "configs", "w8a8.yaml")):
        raise AssertionError(f"unexpected DEFAULT_BIT_CONFIG: {DEFAULT_BIT_CONFIG}")


def _check_jsonl_reader():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "sample.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            f.write('{"question_id":"q1","question_content":"Solve it"}\n')
        df = read_dataset_file(path)
        if df.to_dict("records") != [{"question_id": "q1", "question_content": "Solve it"}]:
            raise AssertionError("jsonl reader returned unexpected rows")


def _check_lcb_validation():
    validate_lcb_docs([{"question_id": "q1", "question_content": "Solve it"}])
    try:
        validate_lcb_docs([{"question_id": ""}])
    except ValueError:
        return
    raise AssertionError("validate_lcb_docs accepted an invalid row")


def _check_quant_target_guard():
    ensure_supported_quant_target("mlp")
    try:
        ensure_supported_quant_target("attn-linear")
    except RuntimeError:
        pass
    else:
        raise AssertionError("attn-linear was accepted without --allow_attn_quant")
    ensure_supported_quant_target("attn-linear", allow_attn_quant=True)


def _check_cli_help():
    for script in CLI_SCRIPTS:
        path = os.path.join(SKILL_DIR, script)
        proc = subprocess.run(
            [sys.executable, path, "--help"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if proc.returncode != 0:
            raise AssertionError(
                f"{script} --help failed with {proc.returncode}: {proc.stderr}")


def main():
    _check_registry()
    _check_jsonl_reader()
    _check_lcb_validation()
    _check_quant_target_guard()
    _check_cli_help()
    print("[self_check] ok")


if __name__ == "__main__":
    main()
