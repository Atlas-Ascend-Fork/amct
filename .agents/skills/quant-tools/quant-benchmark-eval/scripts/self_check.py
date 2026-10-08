# -*- coding: UTF-8 -*-
"""Lightweight self checks for quant-benchmark-eval scripts.

This does not load model weights, allocate NPU memory, download datasets, or run
benchmarks. It verifies script wiring and local config assumptions.

`--remote` additionally cross-checks datasets.json against each dataset repo's
real file list (metadata only, no bulk download); it needs network and is
therefore opt-in.
"""

import argparse
import fnmatch
import os
import re
import subprocess
import sys
import tempfile

SKILL_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SKILL_DIR)

from config import (  # noqa: E402
    BENCH_ASSETS,
    DEFAULT_BIT_CONFIG,
    local_dataset_map,
    load_dataset_registry,
)
from utils import (  # noqa: E402
    NON_LM_EVAL_TASKS,
    TASK_ALIASES,
    ensure_supported_quant_target,
    local_dataset_files,
    read_dataset_file,
    resolve_task_metric_keys,
    split_list_arg,
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
# Scripts that assemble a quant model must expose --algos: amct builds those
# algorithm modules into every quant block, and ptq params only load when the
# keys match (constraints.md #10).
ALGOS_SCRIPTS = ["assemble_and_eval.py", "livecodebench_eval.py"]
_DATA_SUFFIXES = (".parquet", ".json", ".jsonl")


def _check_registry():
    """Validate datasets.json *declarations* (no filesystem, no network).

    Whether the declared globs actually match files on disk is
    `_check_prefetched_resolution`'s job, and whether they match the upstream
    repo layout is `--remote`'s job.
    """
    reg = load_dataset_registry()
    missing = sorted(EXPECTED_TASKS.difference(reg))
    if missing:
        raise AssertionError(f"datasets.json missing tasks: {missing}")
    for task in sorted(reg):
        spec = reg[task]
        for field in ("repo_id", "subdir", "splits"):
            if not spec.get(field):
                raise AssertionError(f"task '{task}' misses '{field}' in datasets.json")
        # Iterate the task's own declared splits: boolq / copa / piqa have no
        # "test" split (lm-eval scores them off "validation"), so a hardcoded
        # split name would fail on valid registry entries.
        for split, rels in spec["splits"].items():
            rels = [rels] if isinstance(rels, str) else rels
            if not any(isinstance(rel, str) and rel.strip() for rel in rels):
                raise AssertionError(
                    f"task '{task}' split '{split}' declares no path/glob"
                )
    if not DEFAULT_BIT_CONFIG.endswith(
        os.path.join("amct_pytorch", "configs", "w8a8.yaml")
    ):
        raise AssertionError(f"unexpected DEFAULT_BIT_CONFIG: {DEFAULT_BIT_CONFIG}")


def _check_prefetched_resolution():
    """For every already-prefetched task, each declared split must resolve.

    `_check_registry` runs with existing_only=False, so it cannot see the real
    failure mode: a split whose globs match nothing on disk (e.g. a repo that
    ships test.jsonl while the registry only lists test-*.parquet). Eval then
    dies with a misleading "run prefetch_data.py first". Only tasks that already
    hold data files are checked, so a fresh cache is not an error.
    """
    reg = load_dataset_registry()
    mapping = local_dataset_map()
    checked = 0
    for task in sorted(reg):
        subdir = os.path.join(BENCH_ASSETS, reg[task]["subdir"])
        if not _dir_has_data_files(subdir):
            continue
        for split in reg[task]["splits"]:
            try:
                files = local_dataset_files(
                    task, reg, mapping, split=split, existing_only=True
                )
            except FileNotFoundError as exc:
                raise AssertionError(
                    f"task '{task}' is prefetched under {subdir}, but split "
                    f"'{split}' resolves to no existing file: {exc}. Fix its "
                    f"globs in datasets.json to the layout the repo ships."
                ) from exc
            if not files:
                raise AssertionError(f"task {task} split {split} resolved to no files")
            checked += 1
    print(f"[self_check] prefetched splits resolve ({checked} checked)")


def _dir_has_data_files(subdir, max_depth=3):
    """True when `subdir` holds at least one dataset file (parquet/json/jsonl)."""
    if not os.path.isdir(subdir):
        return False
    for root, dirs, files in os.walk(subdir):
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        if root[len(subdir) :].count(os.sep) >= max_depth:
            dirs[:] = []
        if any(f.endswith(_DATA_SUFFIXES) for f in files):
            return True
    return False


def _check_remote_registry():
    """Cross-check datasets.json globs against each repo's real file list.

    Metadata only (HfApi.list_repo_files): no dataset is downloaded. This is the
    check that catches a registry entry whose globs no longer match upstream.
    """
    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("[self_check] skip remote registry check: huggingface_hub missing")
        return
    reg = load_dataset_registry()
    api = HfApi()
    default_allow = ["*.parquet", "*.json", "*.jsonl", "*.md"]
    problems = []
    for task, spec in sorted(reg.items()):
        try:
            repo_files = api.list_repo_files(spec["repo_id"], repo_type="dataset")
        except Exception as exc:  # network / private repo: report, don't crash
            problems.append(f"{task}: cannot list {spec['repo_id']}: {exc}")
            continue
        allow = spec.get("allow_patterns", default_allow)
        fetched = [f for f in repo_files if any(fnmatch.fnmatch(f, p) for p in allow)]
        for split, rels in spec["splits"].items():
            if isinstance(rels, str):
                rels = [rels]
            if not any(any(fnmatch.fnmatch(f, rel) for f in fetched) for rel in rels):
                problems.append(
                    f"{task} split '{split}': globs {rels} match none of "
                    f"{spec['repo_id']} files {fetched[:5]}"
                )
        print(f"[self_check] remote ok: {task} ({len(fetched)} fetchable files)")
    if problems:
        raise AssertionError(
            "datasets.json does not match upstream repos:\n  " + "\n  ".join(problems)
        )


def _check_split_list_arg():
    if split_list_arg("lwc,lac") != ["lwc", "lac"]:
        raise AssertionError("split_list_arg broke on a comma-separated string")
    if split_list_arg("", ["lwc", "lac"]) != ["lwc", "lac"]:
        raise AssertionError("split_list_arg ignored its default")
    if split_list_arg(["let"]) != ["let"]:
        raise AssertionError("split_list_arg broke on a list")


def _check_cli_flags():
    """The quant-assembly scripts must expose --algos in their CLI."""
    for script in ALGOS_SCRIPTS:
        path = os.path.join(SKILL_DIR, script)
        proc = subprocess.run(
            [sys.executable, path, "--help"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if "--algos" not in proc.stdout:
            raise AssertionError(f"{script} --help does not expose --algos")


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
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--remote",
        action="store_true",
        help="also validate datasets.json against each HF repo's file list (network)",
    )
    args = ap.parse_args()
    _check_registry()
    _check_prefetched_resolution()
    _check_split_list_arg()
    _check_jsonl_reader()
    _check_lcb_validation()
    _check_quant_target_guard()
    _check_skill_metric_keys()
    _check_cli_help()
    _check_cli_flags()
    if args.remote:
        _check_remote_registry()
    print("[self_check] ok")


if __name__ == "__main__":
    main()
