# -*- coding: UTF-8 -*-
"""Lightweight self checks for quant-benchmark-eval scripts.

This does not load model weights, allocate NPU memory, download datasets, or run
benchmarks. It verifies script wiring and local config assumptions.

`--remote` additionally cross-checks datasets.json against each dataset repo's
real file list (metadata only, no bulk download); it needs network and is
therefore opt-in.
"""

import argparse
import contextlib
import fnmatch
import io
import os
import re
import subprocess
import sys
import tempfile
import types

SKILL_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SKILL_DIR)

import config as config_module  # noqa: E402
from config import (  # noqa: E402
    BENCH_ASSETS,
    DEFAULT_BIT_CONFIG,
    local_dataset_map,
    load_dataset_registry,
    resolve_split_files,
    split_declaration,
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
            _mode, rels = split_declaration(rels)
            if not any(isinstance(rel, str) and rel.strip() for rel in rels):
                raise AssertionError(
                    f"task '{task}' split '{split}' declares no path/glob"
                )
    if not DEFAULT_BIT_CONFIG.endswith(
        os.path.join("amct_pytorch", "configs", "w8a8.yaml")
    ):
        raise AssertionError(f"unexpected DEFAULT_BIT_CONFIG: {DEFAULT_BIT_CONFIG}")


def _check_split_layout_semantics():
    """any_of must not double-read two cached layouts; all_of must merge shards.

    `_resolve_split_files` used to keep *every* existing candidate, so a
    BENCH_ASSETS holding both `test.jsonl` and `data/test-*.parquet` (upstream
    layout migration, or leftover cache) made bench_offline concat the same
    samples twice and every metric silently changed denominator. This runs on a
    throwaway dir: no NPU, no network, no model.
    """
    with tempfile.TemporaryDirectory() as tmp:
        os.makedirs(os.path.join(tmp, "data"))
        for rel in ("test.jsonl", os.path.join("data", "test-00000-of-00001.parquet")):
            open(os.path.join(tmp, rel), "w", encoding="utf-8").close()
        got = resolve_split_files(
            tmp, {"any_of": ["test.jsonl", "data/test-*.parquet"]}
        )
        if got != [os.path.join(tmp, "test.jsonl")]:
            raise AssertionError(f"any_of must pick one layout, got {got}")
        got = resolve_split_files(
            tmp, {"any_of": ["nope.jsonl", "data/test-*.parquet"]}
        )
        if got != [os.path.join(tmp, "data", "test-00000-of-00001.parquet")]:
            raise AssertionError(
                f"any_of must fall through to the next layout, got {got}"
            )
        got = resolve_split_files(tmp, {"any_of": ["a.jsonl", "b.jsonl"]})
        if len(got) != 2:
            raise AssertionError(
                f"any_of with no hit must return declared paths for the error, got {got}"
            )
        for name in ("test.jsonl", "test2.jsonl", "test3.jsonl"):
            open(os.path.join(tmp, name), "w", encoding="utf-8").close()
        got = resolve_split_files(
            tmp, {"all_of": ["test.jsonl", "test2.jsonl", "test3.jsonl"]}
        )
        if len(got) != 3:
            raise AssertionError(f"all_of must merge every shard, got {got}")

    reg = load_dataset_registry()
    for task, expected in (
        ("math500", "any_of"),
        ("livecodebench_release_v6", "all_of"),
    ):
        mode, rels = split_declaration(reg[task]["splits"]["test"])
        if mode != expected or len(rels) < 2:
            raise AssertionError(
                f"datasets.json '{task}' test split must stay '{expected}' with "
                f"multiple entries, got mode='{mode}' rels={rels}"
            )

    # End to end through the real registry with a throwaway BENCH_ASSETS: the
    # two MATH-500 layouts cached at once must resolve to ONE file, and the
    # LiveCodeBench cumulative set must still merge all six shards.
    with tempfile.TemporaryDirectory() as cache:
        m500_dir = os.path.join(cache, reg["math500"]["subdir"], "data")
        os.makedirs(m500_dir)
        for rel in (
            os.path.join(reg["math500"]["subdir"], "test.jsonl"),
            os.path.join(
                reg["math500"]["subdir"], "data", "test-00000-of-00001.parquet"
            ),
        ):
            open(os.path.join(cache, rel), "w", encoding="utf-8").close()
        lcb_dir = os.path.join(cache, reg["livecodebench_release_v6"]["subdir"])
        os.makedirs(lcb_dir)
        for rel in split_declaration(reg["livecodebench_release_v6"]["splits"]["test"])[
            1
        ]:
            open(os.path.join(lcb_dir, rel), "w", encoding="utf-8").close()

        saved_assets = config_module.BENCH_ASSETS
        config_module.BENCH_ASSETS = cache
        try:
            mapping = config_module.local_dataset_map()
        finally:
            config_module.BENCH_ASSETS = saved_assets

        def _files(task):
            spec = reg[task]
            key = (spec["repo_id"], spec.get("dataset_name"))
            return mapping[key]["test"]

        m500_files = _files("math500")
        if len(m500_files) != 1:
            raise AssertionError(
                f"math500 with both layouts cached must read exactly one file, "
                f"got {len(m500_files)}: {m500_files}"
            )
        lcb_files = _files("livecodebench_release_v6")
        if len(lcb_files) != 6:
            raise AssertionError(
                f"livecodebench_release_v6 must merge all 6 cumulative shards, "
                f"got {len(lcb_files)}: {lcb_files}"
            )
    print(
        "[self_check] split layout semantics ok (any_of single-layout / all_of merged)"
    )


# Upstream file lists as of 2026-10, used by the offline regression below so
# `--remote`'s matching logic stays testable without network. Kept minimal: only
# what each task's `allow_patterns` would keep anyway. A repo with no entry here
# gets a listing derived from its own datasets.json declaration
# (`_derived_listing`), so adding a benchmark never means editing this dict.
_FAKE_UPSTREAM = {
    "TIGER-Lab/MMLU-Pro": [
        "README.md",
        "data/test-00000-of-00001.parquet",
        "data/validation-00000-of-00001.parquet",
    ],
    "HuggingFaceH4/MATH-500": [
        "README.md",
        "data/test-00000-of-00001.parquet",
    ],
    "livecodebench/code_generation_lite": [
        "README.md",
        "code_generation_lite.py",
        "test.jsonl",
        "test2.jsonl",
        "test3.jsonl",
        "test4.jsonl",
        "test5.jsonl",
        "test6.jsonl",
    ],
    "allenai/ai2_arc": [
        "README.md",
        "ARC-Challenge/train-00000-of-00001.parquet",
        "ARC-Challenge/validation-00000-of-00001.parquet",
        "ARC-Challenge/test-00000-of-00001.parquet",
        "ARC-Easy/train-00000-of-00001.parquet",
        "ARC-Easy/validation-00000-of-00001.parquet",
        "ARC-Easy/test-00000-of-00001.parquet",
    ],
    "aps/super_glue": [
        "README.md",
        "boolq/train-00000-of-00001.parquet",
        "boolq/validation-00000-of-00001.parquet",
        "copa/train-00000-of-00001.parquet",
        "copa/validation-00000-of-00001.parquet",
    ],
    "baber/piqa": ["README.md", "piqa_train.parquet", "piqa_validation.parquet"],
    "allenai/sciq": [
        "README.md",
        "data/train-00000-of-00001.parquet",
        "data/validation-00000-of-00001.parquet",
        "data/test-00000-of-00001.parquet",
    ],
}


class _StubGap(AssertionError):
    """No offline file list exists for a repo_id the registry asks about.

    Kept distinct from a network failure: `_check_remote_registry` funnels both
    into its `problems` list, and reporting a *fixture* gap as
    "datasets.json does not match upstream repos" sends the reader off to fix
    registry globs that were already correct.
    """


_STUB_GAP_MARK = "[stub-gap]"


def _materialize_glob(rel):
    """Turn one declared path/glob into a concrete filename it still matches.

    `data/test-*.parquet` -> `data/test-00000-of-00001.parquet`;
    `**/test-*.parquet` -> `main/test-00000-of-00001.parquet`. Only used to
    synthesize an upstream listing for a repo that has no hand-written stub.
    """
    parts = []
    for part in rel.split("/"):
        if part == "**":
            parts.append("main")
        elif not any(ch in part for ch in "*?["):
            parts.append(part)
        else:
            base, dot, ext = part.rpartition(".")
            if not dot:
                base, ext = part, ""
            base = re.sub(r"[*?]+", "00000-of-00001", base)
            base = re.sub(r"\[[^\]]*\]", "x", base)
            parts.append(f"{base}.{ext}" if ext else base)
    return "/".join(parts)


def _derived_listing(spec):
    """Synthesize an upstream file list from a registry entry's own declaration.

    This is what keeps the documented extension path true -- "add a benchmark =
    add a datasets.json entry, no code change here" (bench_offline.py). Without
    it a new task made the *default* `self_check.py` fail with
    `--remote false-positive on a correct registry: <task>: cannot list
    <repo_id>`, i.e. a fixture gap dressed up as a registry bug.

    A derived listing is self-consistent with the declaration by construction,
    so for such a repo the offline step covers the any_of/all_of matching logic
    plus whether `allow_patterns` would really download every declared glob
    (prefetch downloads by `allow_patterns`; a declared glob it filters out can
    never resolve locally). Whether the globs fit the *real* upstream layout
    stays `--remote`'s job -- hand-written stubs exist for the known repos
    precisely because they pin layouts worth pinning (MATH-500 shipping parquet
    only, LiveCodeBench's six cumulative shards).
    """
    files = ["README.md"]
    for rels in spec["splits"].values():
        _mode, rel_list = split_declaration(rels)
        files.extend(_materialize_glob(rel) for rel in rel_list)
    return sorted(set(files))


def _upstream_stubs(reg):
    """`(file_lists, derived_repo_ids)` covering every repo_id in `reg`.

    Hand-written stubs win; anything else is derived. A stub no registry entry
    references anymore is an error: it would silently stop testing anything.
    """
    wanted = {spec["repo_id"] for spec in reg.values()}
    dead = sorted(set(_FAKE_UPSTREAM) - wanted)
    if dead:
        raise AssertionError(
            f"_FAKE_UPSTREAM holds repo(s) no datasets.json entry uses: {dead}. "
            "Drop them, or fix the repo_id they were written for."
        )
    file_lists, derived = {}, []
    for repo_id in sorted(wanted):
        if repo_id in _FAKE_UPSTREAM:
            file_lists[repo_id] = list(_FAKE_UPSTREAM[repo_id])
            continue
        spec = next(s for s in reg.values() if s["repo_id"] == repo_id)
        file_lists[repo_id] = _derived_listing(spec)
        derived.append(repo_id)
    return file_lists, derived


def _probe_entry(allow_patterns):
    """A registry entry for a repo that has no hand-written stub."""
    return {
        "repo_id": "openai/gsm8k",
        "subdir": "gsm8k",
        "dataset_name": "main",
        "splits": {"test": "main/test-*.parquet"},
        "allow_patterns": allow_patterns,
    }


def _run_remote_check(file_lists, reg=None):
    """Run `_check_remote_registry` against injected file lists; return problems.

    `huggingface_hub` is swapped for a stub so the *matching logic* is covered
    offline (this box may have no network, and `--remote` must not be the only
    thing that can catch a registry regression). `reg` optionally injects the
    registry as well, so an entry that is not in datasets.json yet can still be
    exercised. Returns the AssertionError text when the check fails, else "".
    """

    class _FakeApi:
        def list_repo_files(self, repo_id, repo_type="dataset"):
            try:
                return list(file_lists[repo_id])
            except KeyError:
                raise _StubGap(
                    f"{_STUB_GAP_MARK} no offline file list for {repo_id}; build "
                    "the mapping with _upstream_stubs(reg) so every registry "
                    "repo_id is covered (hand-written stub or derived)"
                ) from None

    stub = types.ModuleType("huggingface_hub")
    stub.HfApi = lambda *a, **k: _FakeApi()
    saved_hub = sys.modules.get("huggingface_hub")
    saved_loader = globals()["load_dataset_registry"]
    sys.modules["huggingface_hub"] = stub
    if reg is not None:
        globals()["load_dataset_registry"] = lambda: reg
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            _check_remote_registry()
    except AssertionError as exc:
        return str(exc)
    finally:
        if saved_hub is None:
            sys.modules.pop("huggingface_hub", None)
        else:
            sys.modules["huggingface_hub"] = saved_hub
        globals()["load_dataset_registry"] = saved_loader
    return ""


def _repo_of(reg, task):
    """repo_id a negative control breaks, resolved through the registry.

    Hardcoding the string lets a datasets.json rename quietly turn the control
    into a no-op -- it would mutate a repo no task reads and still pass.
    """
    try:
        return reg[task]["repo_id"]
    except KeyError:
        raise AssertionError(
            f"self_check's --remote negative controls need task '{task}' in "
            "datasets.json; update them alongside the registry"
        ) from None


def _check_remote_glob_semantics():
    """`--remote` must honor any_of / all_of instead of iterating dict keys.

    Regression for the reported false positive: `_check_remote_registry`
    normalized with `isinstance(rels, str)` only, so a dict declaration stayed a
    dict and `for rel in rels` yielded the key names `any_of` / `all_of`, which
    never fnmatch -- `self_check.py --remote` then failed math500, livecodebench
    and livecodebench_release_v6 against a perfectly correct datasets.json. Also
    covers the lax `any(...)` that let a missing all_of shard through, and the
    follow-up regression where the offline stub dict became a second file to
    edit whenever a benchmark was added to datasets.json.
    """
    # 1. Normalization: no declaration may leak its mode key as a path.
    reg = load_dataset_registry()
    for task, spec in sorted(reg.items()):
        for split, rels in spec["splits"].items():
            mode, rel_list = split_declaration(rels)
            if mode not in ("any_of", "all_of"):
                raise AssertionError(f"{task}/{split}: bad mode {mode!r}")
            leaked = [r for r in rel_list if r in ("any_of", "all_of")]
            if leaked:
                raise AssertionError(
                    f"{task}/{split}: mode key leaked into paths {rel_list}"
                )

    # 2. Matching semantics.
    if _unmatched_globs(
        "any_of",
        ["test.jsonl", "data/test-*.parquet"],
        ["data/test-00000-of-00001.parquet"],
    ):
        raise AssertionError("any_of must pass when one candidate matches")
    if _unmatched_globs(
        "any_of",
        ["test.jsonl", "data/test-*.parquet"],
        ["test.jsonl", "data/test-00000-of-00001.parquet"],
    ):
        raise AssertionError("any_of must pass when BOTH layouts are present")
    got = _unmatched_globs("any_of", ["a.jsonl", "b.jsonl"], ["c.jsonl"])
    if got != ["a.jsonl", "b.jsonl"]:
        raise AssertionError(
            f"any_of with no hit must report all candidates, got {got}"
        )
    got = _unmatched_globs(
        "all_of", ["test.jsonl", "test2.jsonl"], ["test.jsonl", "test2.jsonl"]
    )
    if got:
        raise AssertionError(f"all_of complete must pass, got {got}")
    got = _unmatched_globs(
        "all_of",
        ["test.jsonl", "test2.jsonl", "test3.jsonl"],
        ["test.jsonl", "test3.jsonl"],
    )
    if got != ["test2.jsonl"]:
        raise AssertionError(f"all_of must name the missing shard, got {got}")

    # 3. End to end against realistic upstream listings: zero problems. Every
    #    registry repo_id is covered, by a hand-written stub or a derived one.
    file_lists, derived = _upstream_stubs(reg)
    problems = _run_remote_check(file_lists, reg=reg)
    if _STUB_GAP_MARK in problems:
        raise AssertionError(
            f"offline upstream listings do not cover the registry:\n{problems}"
        )
    if problems:
        raise AssertionError(
            f"--remote false-positive on a correct registry:\n{problems}"
        )
    if derived:
        print(
            "[self_check] note: no hand-written upstream stub for "
            f"{', '.join(derived)} -- listing derived from the datasets.json "
            "declaration, so run --remote to verify them against the real repos"
        )

    # 4. Negative controls -- it must still catch real drift. repo_ids and
    #    shard names come from the registry, not from literals.
    lcb_repo = _repo_of(reg, "livecodebench_release_v6")
    mode, shards = split_declaration(reg["livecodebench_release_v6"]["splits"]["test"])
    if mode != "all_of" or len(shards) < 2:
        raise AssertionError(
            "self_check's --remote controls expect livecodebench_release_v6/test "
            f"to stay a multi-shard all_of declaration, got ({mode}, {shards})"
        )
    dropped = shards[len(shards) // 2]
    broken = {k: list(v) for k, v in file_lists.items()}
    broken[lcb_repo] = [f for f in broken[lcb_repo] if not fnmatch.fnmatch(f, dropped)]
    problems = _run_remote_check(broken, reg=reg)
    if dropped not in problems:
        raise AssertionError(
            f"--remote must report a missing all_of shard ({dropped}), "
            f"got:\n{problems or '<no problems>'}"
        )

    math_repo = _repo_of(reg, "math500")
    mode, cands = split_declaration(reg["math500"]["splits"]["test"])
    if mode != "any_of" or "test.jsonl" not in cands:
        raise AssertionError(
            "self_check's --remote controls expect math500/test to stay an any_of "
            f"declaration listing test.jsonl, got ({mode}, {cands})"
        )
    broken = {k: list(v) for k, v in file_lists.items()}
    broken[math_repo] = ["README.md", "test.jsonl"]
    problems = _run_remote_check(broken, reg=reg)
    if problems:
        raise AssertionError(
            f"MATH-500 shipping only test.jsonl is a valid any_of layout:\n{problems}"
        )

    sciq_repo = _repo_of(reg, "sciq")
    broken = {k: list(v) for k, v in file_lists.items()}
    broken[sciq_repo] = ["README.md"]
    problems = _run_remote_check(broken, reg=reg)
    if "sciq" not in problems:
        raise AssertionError(
            f"--remote must report a task whose files all vanished, got:\n{problems or '<no problems>'}"
        )

    # 5. Extension safety: a task added to datasets.json *alone* (no stub edit)
    #    must keep the default self-check green -- and must still be checked.
    probe = dict(reg)
    probe["gsm8k_probe"] = _probe_entry(["*.parquet", "README.md"])
    probe_lists, probe_derived = _upstream_stubs(probe)
    if "openai/gsm8k" not in probe_derived:
        raise AssertionError(
            "a repo without a hand-written stub must fall back to a derived "
            f"listing, got derived={probe_derived}"
        )
    problems = _run_remote_check(probe_lists, reg=probe)
    if problems:
        raise AssertionError(
            "adding a datasets.json entry must not break the default self-check "
            f"(no _FAKE_UPSTREAM stub edit required):\n{problems}"
        )
    # Not vacuous: a declared glob that allow_patterns would never download is
    # reported even when the listing is derived.
    probe = dict(reg)
    probe["gsm8k_probe"] = _probe_entry(["README.md"])
    probe_lists, _ = _upstream_stubs(probe)
    problems = _run_remote_check(probe_lists, reg=probe)
    if "gsm8k_probe" not in problems:
        raise AssertionError(
            "a declared glob excluded by allow_patterns must be reported even "
            f"when the listing is derived, got:\n{problems or '<no problems>'}"
        )
    print(
        "[self_check] remote glob semantics ok (any_of one-hit / all_of every "
        "shard / new task needs no stub)"
    )


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


def _glob_hits(rel, fetched):
    """True when one declared path/glob matches at least one fetched file."""
    return any(fnmatch.fnmatch(f, rel) for f in fetched)


def _unmatched_globs(mode, rel_list, fetched):
    """Declared rels that `fetched` fails to satisfy, per declaration mode.

    Must go through `config.split_declaration()`'s `(mode, rel_list)` -- the
    historical `isinstance(rels, str)` normalization leaves a dict declaration
    (`{"any_of": [...]}`) intact, so `for rel in rels` iterated the *key names*
    `any_of` / `all_of`. Those never fnmatch, and every dict-declared task was
    reported as "datasets.json does not match upstream repos" -- a false
    positive that sends you to "fix" a correct registry.

    The two modes also need different strictness:

    - ``any_of`` — alternative layouts of the same data; ONE candidate matching
      is enough (requiring all would demand a repo ship two layouts at once).
    - ``all_of`` — shards that all get merged; EVERY one must match. The old
      blanket `any(...)` was too lax here: a dropped shard (upstream rename,
      partial mirror) passed the check and then silently evaluated on fewer
      samples.
    """
    if mode == "any_of":
        return (
            [] if any(_glob_hits(rel, fetched) for rel in rel_list) else list(rel_list)
        )
    return [rel for rel in rel_list if not _glob_hits(rel, fetched)]


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
            mode, rel_list = split_declaration(rels)
            unmatched = _unmatched_globs(mode, rel_list, fetched)
            if unmatched:
                problems.append(
                    f"{task} split '{split}' ({mode}): declared {rel_list}, "
                    f"unmatched {unmatched} against {spec['repo_id']} "
                    f"files {fetched[:5]}"
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
    _check_split_layout_semantics()
    _check_remote_glob_semantics()
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
