# -*- coding: UTF-8 -*-
"""Lightweight self checks for quant-benchmark-eval scripts.

This does not load model weights, allocate NPU memory, download datasets, or run
benchmarks. It verifies script wiring and local config assumptions.
"""

import os
import re
import subprocess
import sys
import tempfile

SKILL_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SKILL_DIR)

from config import DEFAULT_BIT_CONFIG, local_dataset_map, load_dataset_registry  # noqa: E402
from utils import (  # noqa: E402
    NON_LM_EVAL_TASKS,
    TASK_ALIASES,
    ensure_supported_quant_target,
    local_dataset_files,
    read_dataset_file,
    resolve_task_metric_keys,
    validate_lcb_docs,
)

EXPECTED_TASKS = {
    "mmlu_pro",
    "math500",
    "livecodebench",
    "livecodebench_release_v6",
    "arc_challenge",
    "arc_easy",
    "boolq",
    "copa",
    "piqa",
    "sciq",
}
SKILL_MD = os.path.join(os.path.dirname(SKILL_DIR), "SKILL.md")
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
        # Iterate the task's own declared splits: boolq / copa / piqa have no
        # "test" split (lm-eval scores them off "validation"), so a hardcoded
        # split name would fail on valid registry entries.
        for split in reg[task]["splits"]:
            files = local_dataset_files(
                task, reg, mapping, split=split, existing_only=False
            )
            if not files:
                raise AssertionError(f"task {task} split {split} resolved to no files")
    if not DEFAULT_BIT_CONFIG.endswith(
        os.path.join("amct_pytorch", "configs", "w8a8.yaml")
    ):
        raise AssertionError(f"unexpected DEFAULT_BIT_CONFIG: {DEFAULT_BIT_CONFIG}")


def _check_jsonl_reader():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "sample.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            f.write('{"question_id":"q1","question_content":"Solve it"}\n')
        df = read_dataset_file(path)
        if df.to_dict("records") != [
            {"question_id": "q1", "question_content": "Solve it"}
        ]:
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


def _parse_metric_table(path):
    """Read (task, metric) pairs out of SKILL.md's cost/metric table.

    Rows look like `| 低 | ` + "`arc_challenge`" + ` | ` + "`acc_norm,none`" + ` | ... |`,
    i.e. column 2 is the task and column 3 the default metric, both backticked.
    A column may list several backticked names (e.g. two livecodebench tasks).
    """
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.startswith("|"):
                continue
            cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
            if len(cells) < 4 or not cells[0].startswith(("低", "高")):
                continue
            tasks = re.findall(r"`([^`]+)`", cells[1])
            metrics = re.findall(r"`([^`]+)`", cells[2])
            if not tasks or not metrics:
                raise AssertionError(f"unparsable metric table row: {line.strip()}")
            rows.extend((task, metric) for task in tasks for metric in metrics)
    return rows


def _check_skill_metric_keys():
    """Cross-check SKILL.md's metric column against lm-eval's installed task index.

    lm-eval keys every result as "<metric>,<filter>" (mmlu_pro uses the
    "custom-extract" filter, most tasks use "none"). A doc that drifts from the
    index makes compare_results.py raise a KeyError at the last step of the
    workflow, so verify it here instead. Only the task index / YAML configs are
    read - no dataset download, no NPU. Skipped when lm-eval is absent.
    """
    rows = _parse_metric_table(SKILL_MD)
    if not rows:
        raise AssertionError(f"no metric table rows parsed from {SKILL_MD}")
    task_manager = None
    for task, metric in rows:
        if task in NON_LM_EVAL_TASKS:
            continue
        lm_task = TASK_ALIASES.get(task, task)
        try:
            from lm_eval.tasks import TaskManager
        except ImportError:
            print("[self_check] skip metric-key check: lm-eval not installed")
            return
        if task_manager is None:
            task_manager = TaskManager()
        keys = resolve_task_metric_keys(lm_task, task_manager)
        if metric not in keys:
            raise AssertionError(
                f"SKILL.md documents metric '{metric}' for task '{task}', "
                f"but lm-eval emits {keys} for '{lm_task}'"
            )
    print(f"[self_check] metric keys ok ({len(rows)} rows)")


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
                f"{script} --help failed with {proc.returncode}: {proc.stderr}"
            )


def main():
    _check_registry()
    _check_jsonl_reader()
    _check_lcb_validation()
    _check_quant_target_guard()
    _check_skill_metric_keys()
    _check_cli_help()
    print("[self_check] ok")


if __name__ == "__main__":
    main()
