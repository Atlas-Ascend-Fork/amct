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
"""Lightweight self checks for quant-benchmark-eval scripts.

This does not load model weights, allocate NPU memory, download datasets, or run
benchmarks. It verifies script wiring and local config assumptions.

Scope: factory calibration and silent-class regressions. Runtime guarding is
not here -- it lives in the eval scripts' fail-fast pre-flights and in the
automatic cross-check of the comparability keys in the result JSON. A check
belongs in this file only when the error it catches would *not* surface at run
time: a wrong number that still reads as a real result, a command surface an
agent would be fooled by, or a calibrated fact source drifting out of sync with
the docs. Purely cosmetic consistency (wording, spelling, counts) goes through
review instead.

`--remote` additionally cross-checks datasets.json against each dataset repo's
real file list (metadata only, no bulk download); it needs network and is
therefore opt-in.

Maintainer notes -- nothing below is agent-facing. The only doc that names this
file is SKILL.md #5, whose failure table sends whoever edits the registry here
with --remote; daily evaluation never runs it.

- `--list-checks` renders every wired check and marks which default ones pass
  silently, so a green run can be read without the docs keeping a copy of the
  check list.
- `_EXTERNAL_IDENTIFIERS` / `_EXTERNAL_FLAGS`: the explicit allow-lists of
  third-party names and flags the docs may mention (names found neither in
  these scripts nor in `$AMCT_HOME`). Admission bar: genuinely third-party and
  the docs must name it. An entry the docs no longer use is itself an error, so
  an allowance cannot outlive its reason.
- `_FAKE_UPSTREAM`: `--remote`'s offline regression stub. It pins the layout of
  the stub repos that were built; it is not the source of truth for any real
  upstream layout.
- `EXPECTED_TASKS`: a deliberate second copy of the registry's task list, so the
  registry cannot shrink silently. Deleting / renaming a task means editing both
  `datasets.json` and this set.
"""

import argparse
import ast
import contextlib
import fnmatch
import io
import json
import os
import re
import subprocess
import sys
import tempfile
import types

SKILL_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SKILL_DIR)

import config as config_module  # noqa: E402
from compare_results import (  # noqa: E402
    _COMPARABILITY_KEYS as COMPARABILITY_KEYS,
    _META_BOOKKEEPING_KEYS as META_BOOKKEEPING_KEYS,
    _OBJECT_SIDE_KEYS as OBJECT_SIDE_KEYS,
    _check_comparable as check_comparable,
    _run_metadata as run_metadata,
)
from config import (  # noqa: E402
    AMCT_HOME,
    DEFAULT_BIT_CONFIG,
    UnresolvedSplit,
    load_dataset_registry,
    require_bit_config,
    resolve_split_files,
    split_declaration,
)
from utils import (  # noqa: E402
    NON_LM_EVAL_TASKS,
    TASK_ALIASES,
    check_task_offline_ready,
    ensure_supported_quant_target,
    local_dataset_files,
    read_dataset_file,
    resolve_sample_limit,
    resolve_task_metric_keys,
    sample_limit_fields,
    split_list_arg,
    validate_lcb_docs,
)

# Second source of truth on purpose (see the module docstring's maintainer
# notes): `datasets.json` decides which tasks can actually run, this set only
# pins "the registry must not shrink silently".
# Deleting / renaming a task therefore means editing BOTH -- prefetch_data.py
# already fails loudly on a task the registry does not declare, so dropping this
# set would cost only the anti-shrink guarantee, not the loud failure.
EXPECTED_TASKS = {
    "mmlu_pro",
    "math500",
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
    "plan_device.py",
]
# self_check.py builds an ArgumentParser too, but it is the checker, not an
# agent-facing CLI: --remote / --list-checks are maintenance flags, and
# requiring the docs to name them would put this file's own upkeep into the
# agent's reading path. `_scripts_with_argparse` derives the list above from
# the scripts that actually take flags, minus this one, so a new entry point
# cannot ship with flags no doc has to mention.
_NON_CLI_SCRIPTS = ("self_check.py",)


def _scripts_with_argparse():
    """Every ``scripts/*.py`` that builds an ArgumentParser, by AST.

    Parsed rather than grepped: the word appears in comments and docstrings
    here, and a match in prose would keep a removed entry point on the list.
    """
    out = []
    for fname in sorted(os.listdir(SKILL_DIR)):
        if not fname.endswith(".py"):
            continue
        with open(os.path.join(SKILL_DIR, fname), encoding="utf-8") as f:
            tree = ast.parse(f.read())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if (isinstance(func, ast.Attribute) and func.attr == "ArgumentParser") or (
                isinstance(func, ast.Name) and func.id == "ArgumentParser"
            ):
                out.append(fname)
                break
    return out


# Scripts that assemble a quant model must expose --algos: amct builds those
# algorithm modules into every quant block, and ptq params only load when the
# keys match (constraints.md #3).
ALGOS_SCRIPTS = ["assemble_and_eval.py", "livecodebench_eval.py"]

# CLI flags the docs mention that belong to *other* tools, not to this skill's
# argparse: pip's installer flag and the official LiveCodeBench evaluator's own
# arguments (references/livecodebench.md tells the user to run lcb_runner).
# Every entry must actually appear in the docs, so a dead allowance fails.
_EXTERNAL_FLAGS = {
    "--user": "pip install --user",
    "--scenario": "lcb_runner.runner.custom_evaluator",
    "--custom_output_file": "lcb_runner.runner.custom_evaluator",
}

# Identifiers the docs name that belong to neither this skill's scripts nor the
# amct checkout: third-party names -- library kwargs / config keys, environment
# variables the interpreter or the HF stack reads, upstream lm-eval task names.
# Deliberately a text block parsed at import, not a dict literal --
# `_check_doc_code_drift` searches the scripts' own text, so a literal here
# would put every allowed name into the blob it searches, the lookup would
# "find" it, and the check would quietly stop verifying anything (the same trap
# the injected controls dodge by building stale names at runtime). Names that
# live in amct need no allowance: they are verified against $AMCT_HOME below,
# so an amct rename is caught. Every entry must cite the skill doc that uses
# the name, so a dead allowance reads wrong even when nothing reports it.
_EXTERNAL_IDENTIFIERS_TEXT = """\
attention_mask|transformers forward/generate kwarg (constraints.md #1)
cache_position|transformers attention forward kwarg (constraints.md #1)
past_key_values|transformers KV-cache container (constraints.md #1)
max_memory|transformers device_map kwarg (env-and-portability.md #2)
max_split_size_mb|allocator config key, not a Python symbol (env-and-portability.md #2)
PYTHONPATH|interpreter search path the amct prepend rule is about (env-and-portability.md #5)
HF_DATASETS_OFFLINE|datasets env switch, explicitly not our local read mechanism (env-and-portability.md #5)
HF_HUB_OFFLINE|hf_hub env switch, explicitly not our local read mechanism (env-and-portability.md #5)
HF_ENDPOINT|hf_hub endpoint switch, read by the hub client itself, not by this skill (env-and-portability.md #4/#5)
minerva_math500|lm-eval task name offered as another math500 prompt (SKILL.md #4)
multiple_choice|lm-eval task output_type: loglikelihood scoring, no generation (SKILL.md #3/#4)
generate_until|lm-eval task output_type: generation required (SKILL.md #3)
"""
_EXTERNAL_IDENTIFIERS = dict(
    line.split("|", 1) for line in _EXTERNAL_IDENTIFIERS_TEXT.strip().splitlines()
)


def _doc_identifiers(text):
    """``[(name, is_call)]`` for every identifier-shaped token the docs backtick.

    Covers all four spellings the docs actually use -- `name()`, `name`,
    `mod.name`, `mod.name()`. Anything else inside backticks (paths, shell
    lines, JSON, `--flags`) is skipped; those have their own checks below.
    """
    out = set()
    for token in re.findall(r"`([^`\n]+)`", text):
        token = token.strip()
        for pattern, is_call in (
            (r"([A-Za-z_][\w.]*)\(\)", True),
            (r"([A-Za-z_][\w.]*)", False),
        ):
            match = re.fullmatch(pattern, token)
            if match:
                out.add((match.group(1), is_call))
                break
    return sorted(out)


def _amct_identifiers():
    """Identifier set of $AMCT_HOME/amct_pytorch, or None when unresolvable.

    One pass over the checkout (~0.1s). The docs name amct-side hooks
    (`input_transform`, `extract_ptq_data`) that no script here defines, and
    those are exactly the names an amct refactor renames -- so resolve them
    against the source instead of allowing them by hand.
    """
    root = os.path.join(AMCT_HOME, "amct_pytorch")
    if not AMCT_HOME or not os.path.isdir(root):
        return None
    names = set()
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if not d.startswith((".", "__pycache__"))]
        for fname in filenames:
            if not fname.endswith(".py"):
                continue
            try:
                with open(
                    os.path.join(dirpath, fname), encoding="utf-8", errors="ignore"
                ) as f:
                    names.update(re.findall(r"[A-Za-z_]\w*", f.read()))
            except OSError:
                continue
    return names


def _doc_sources():
    """`{filename: text}` for every script, and `[(name, text)]` for every doc."""
    sources = {}
    for fname in sorted(os.listdir(SKILL_DIR)):
        if fname.endswith(".py"):
            with open(os.path.join(SKILL_DIR, fname), encoding="utf-8") as f:
                sources[fname] = f.read()
    ref_dir = os.path.join(os.path.dirname(SKILL_DIR), "references")
    docs = [("SKILL.md", SKILL_MD)]
    docs += [
        (f"references/{n}", os.path.join(ref_dir, n))
        for n in sorted(os.listdir(ref_dir))
        if n.endswith(".md")
    ]
    loaded = []
    for name, path in docs:
        with open(path, encoding="utf-8") as f:
            loaded.append((name, f.read()))
    return sources, loaded


def _check_doc_code_drift(docs=None, sources=None):
    """Docs must not name code that no longer exists.

    Regression for a reviewer-reported drift: the model loader was renamed to
    drop its leading underscore when a second script started reusing it, code
    references were updated, but SKILL.md's assembly flow and constraints.md
    #13 kept spelling the old private name -- a reader grepping the docs found
    nothing and would assume the function was deleted. Renames are exactly when
    this happens, and nothing else in the self-check looks at the docs except
    the metric table.

    Do not write a stale identifier in this file as a backticked call: the
    drift check scans every script here, so a literal in a docstring or a
    control satisfies it and the control passes for the wrong reason (build
    such names at runtime instead -- see `_check_doc_drift_controls`).

    Three mechanical cross-checks, no network / NPU:

    1. every code identifier the docs mention -- `name()`, `name`, `mod.name`
       alike -- is defined or called in scripts/, exists in $AMCT_HOME, or is an
       explicitly allowed third-party name (`_EXTERNAL_IDENTIFIERS`);
    2. every `scripts/<file>.py` the docs mention exists, and every runnable
       script is mentioned (an undocumented entry point is a doc gap too);
    3. every `--flag` the docs mention is declared by some argparse here, or is
       an explicitly allowed flag of another tool (`_EXTERNAL_FLAGS`).

    `docs` / `sources` can be injected so the reverse controls below need no
    edit to the real files.
    """
    if sources is None or docs is None:
        real_sources, real_docs = _doc_sources()
        sources = real_sources if sources is None else sources
        docs = real_docs if docs is None else docs
    # The allowance block is text inside one of the scanned sources; drop it so
    # an allowed name cannot satisfy the "found in scripts/" lookup by being
    # spelled in this very file (see _EXTERNAL_IDENTIFIERS_TEXT).
    blob = "\n".join(sources.values()).replace(_EXTERNAL_IDENTIFIERS_TEXT, "")
    text = "\n".join(body for _, body in docs)

    # 1. code identifiers. Matching only the `name()` call form was the shape
    #    of the one reported drift, not the class of it: most code mentions in
    #    these docs are bare (`input_transform`, `resolve_split_files`), and a
    #    rename hides in those just as well.
    identifiers = _doc_identifiers(text)
    amct_names = _amct_identifiers()
    if amct_names is None:
        print(
            "[self_check] note: $AMCT_HOME unresolved, so amct-side identifiers "
            "the docs name are not verified against the source"
        )
    unresolved = []
    for name, is_call in identifiers:
        leaf = name.split(".")[-1]
        needle = r"\b" + re.escape(leaf) + (r"\s*\(" if is_call else r"\b")
        if re.search(needle, blob):
            continue
        if leaf in _EXTERNAL_IDENTIFIERS:
            continue
        if amct_names is not None and leaf in amct_names:
            continue
        unresolved.append(f"`{name}{'()' if is_call else ''}`")
    if unresolved:
        raise AssertionError(
            f"docs name {unresolved}, but no script here defines or calls it"
            + (
                ", $AMCT_HOME/amct_pytorch has no such name,"
                if amct_names is not None
                else " (amct names unverifiable: $AMCT_HOME unresolved),"
            )
            + " and _EXTERNAL_IDENTIFIERS does not allow it: a rename or "
            "removal was not propagated to the docs"
        )
    for name, why in sorted(_EXTERNAL_IDENTIFIERS.items()):
        if not any(tok.split(".")[-1] == name for tok, _ in identifiers):
            raise AssertionError(
                f"_EXTERNAL_IDENTIFIERS allows '{name}' ({why}), but the docs no "
                "longer mention it; drop the allowance so it cannot silently "
                "start hiding a real drift"
            )

    # 2. script filenames, both directions. The reverse direction covers the
    #    scripts an agent runs: an import-only helper is reached through the
    #    entry point that imports it, and a table listing every file would only
    #    restate what the workflow commands already say (and drift from them).
    mentioned = set(re.findall(r"scripts/([A-Za-z_][\w]*\.py)", text))
    for fname in sorted(mentioned - set(sources)):
        raise AssertionError(f"docs reference scripts/{fname}, which does not exist")
    runnable = {name for name, src in sources.items() if "__main__" in src}
    for fname in sorted(runnable - mentioned):
        raise AssertionError(
            f"scripts/{fname} is runnable but no doc names it: an agent cannot "
            "invoke a script it has no way to find"
        )

    # 3. CLI flags
    # `--help` is argparse's own: every parser here is built with the default
    # add_help=True, so the flag exists without any add_argument call to match.
    declared = {"--help"} | set(
        re.findall(r"""add_argument\(\s*['"](--[a-z0-9_\-]+)['"]""", blob)
    )
    for flag in sorted(set(re.findall(r"(--[a-z][a-z0-9_\-]{2,})", text))):
        if flag in declared:
            continue
        if flag in _EXTERNAL_FLAGS and _EXTERNAL_FLAGS[flag] in text:
            continue
        raise AssertionError(
            f"docs mention {flag}, but no script declares it and it is not an "
            f"allowed flag of another tool ({sorted(_EXTERNAL_FLAGS)})"
        )
    for flag, owner in sorted(_EXTERNAL_FLAGS.items()):
        if flag not in text or owner not in text:
            raise AssertionError(
                f"_EXTERNAL_FLAGS allows {flag} for '{owner}', but the docs no "
                "longer mention that pair; drop the allowance so it cannot "
                "silently start hiding a real drift"
            )


def _check_doc_drift_controls():
    """The doc-drift check must fail on drift -- a cross-check that cannot fail
    reads as coverage while checking nothing. Inject docs; never edit the real
    ones. Each control keeps the full real doc set and perturbs one entry, so
    the failure it provokes is the intended one and not an artifact of passing a
    truncated doc.
    """
    sources, docs = _doc_sources()

    def _with(extra=None, drop=None):
        # `drop` applies to *every* doc: most scripts are mentioned in more than
        # one file (compare_results.py in SKILL.md and references/livecodebench.md),
        # so dropping it from one leaves the mention the check would find and the
        # control passes without proving anything.
        out = []
        for name, text in docs:
            body = text if drop is None else text.replace(drop[0], drop[1])
            if extra and name == "SKILL.md":
                body += extra
            out.append((name, body))
        return out

    def _must_fail(injected, needle, what, **kw):
        try:
            _check_doc_code_drift(docs=injected, sources=kw.pop("sources", sources))
        except AssertionError as exc:
            if needle not in str(exc):
                raise AssertionError(
                    f"{what}: drift error does not name '{needle}': {exc}"
                ) from exc
        else:
            raise AssertionError(f"{what}: drift passed the check")

    # 1. A callable the scripts no longer define -- the reported drift shape.
    #    Built at runtime: this file is one of the scanned sources, so writing
    #    the identifier as a literal here would make the drift check find it in
    #    its own control code and the control would pass for the wrong reason.
    stale = "_load_" + "float"
    _must_fail(
        _with(extra=f"\n装配用 `{stale}()` 加载模型。\n"),
        stale,
        "stale callable",
    )
    # 1b. The same drift spelled *without* the call parens, which is how most
    #     code mentions in these docs are written. A check that only matched
    #     `name()` passed this shape -- the reported bug's form was codified
    #     instead of the class of bug.
    _must_fail(
        _with(extra=f"\n装配阶段由 `{stale}` 负责加载权重。\n"),
        stale,
        "stale bare identifier",
    )
    # 1c. A dotted mention must be caught by its leaf too.
    _must_fail(
        _with(extra=f"\n见 `config.{stale}`。\n"),
        stale,
        "stale dotted identifier",
    )
    # 1d. An identifier the docs name that amct no longer defines must fail as
    #     well; it is resolved against $AMCT_HOME, not against an allow-list.
    if _amct_identifiers() is not None:
        gone = "input_trans" + "form_renamed"
        _must_fail(
            _with(extra=f"\n量化钩子 `{gone}` 由 amct 提供。\n"),
            gone,
            "stale amct identifier",
        )
    # 2a. A script path that does not exist.
    _must_fail(_with(extra="\n见 `scripts/nope.py`。\n"), "nope.py", "ghost script")
    # 2b. A script that exists but the docs stopped mentioning.
    _must_fail(
        _with(drop=("scripts/compare_results.py", "compare_results.py")),
        "compare_results.py",
        "undocumented script",
    )
    # 3. A CLI flag no argparse declares and no other tool owns.
    _must_fail(
        _with(extra="\n加上 `--bogus_flag_xyz` 即可。\n"),
        "--bogus_flag_xyz",
        "ghost flag",
    )
    # 4. An allowance for a flag the docs no longer mention must not linger.
    saved_flags = dict(_EXTERNAL_FLAGS)
    globals()["_EXTERNAL_FLAGS"] = dict(
        saved_flags, **{"--never_allowed": "a tool nobody documents"}
    )
    try:
        _must_fail(_with(), "--never_allowed", "dead allowance")
    finally:
        globals()["_EXTERNAL_FLAGS"] = saved_flags
    # 5. Same for an identifier allowance: a name nobody documents any more
    #    would sit there forever and silently excuse a future drift.
    saved_idents = dict(_EXTERNAL_IDENTIFIERS)
    globals()["_EXTERNAL_IDENTIFIERS"] = dict(
        saved_idents, **{"never_allowed_ident": "a kwarg no doc mentions"}
    )
    try:
        _must_fail(_with(), "never_allowed_ident", "dead identifier allowance")
    finally:
        globals()["_EXTERNAL_IDENTIFIERS"] = saved_idents
    print("[self_check] doc/code drift controls ok (7 reverse checks)")


def _check_registry():
    """Validate datasets.json *declarations* (no filesystem, no network).

    Whether the declared globs actually match files on disk is
    the entry points' pre-flight job (`check_task_offline_ready` /
    `resolve_dataset_files`, both of which raise when a declared split matches
    nothing), and whether they match the upstream
    repo layout is `--remote`'s job.
    """
    reg = load_dataset_registry()
    missing = sorted(EXPECTED_TASKS.difference(reg))
    if missing:
        raise AssertionError(
            f"datasets.json declares no entry for {missing}. EXPECTED_TASKS is "
            "a deliberate second copy of the registry's task list, so either an "
            "entry was dropped by accident (restore it in datasets.json), or the "
            "task was removed on purpose (drop it from EXPECTED_TASKS in this "
            "file too -- the pairing is what keeps the registry from shrinking "
            "silently)."
        )
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
    # The suffix alone says nothing about the file being there: a stale
    # AMCT_HOME used to fail inside the amct pipeline, after the float model was
    # already on the card.
    if AMCT_HOME and not os.path.isfile(DEFAULT_BIT_CONFIG):
        raise AssertionError(
            f"AMCT_HOME={AMCT_HOME} resolved but the default bit config is "
            f"missing: {DEFAULT_BIT_CONFIG}"
        )
    bogus = os.path.join(tempfile.gettempdir(), "selfcheck-no-such-bit-config.yaml")
    try:
        require_bit_config(bogus)
    except RuntimeError as exc:
        if "selfcheck-no-such-bit-config.yaml" not in str(exc):
            raise AssertionError(
                f"missing bit config error does not name the path: {exc}"
            ) from exc
    else:
        raise AssertionError("require_bit_config accepted a path that does not exist")
    if AMCT_HOME and require_bit_config() != DEFAULT_BIT_CONFIG:
        raise AssertionError("require_bit_config() must default to DEFAULT_BIT_CONFIG")


def _check_split_layout_semantics():
    """any_of must not double-read two cached layouts; all_of must merge shards.

    `resolve_split_files` used to keep *every* existing candidate, so a
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
        # A *partial* shard set must be refused, not merged. LiveCodeBench's
        # shards are cumulative, so scoring 3 of 4 reports a pass@1 whose
        # denominator silently shrank -- and it reads as a real result.
        try:
            resolve_split_files(
                tmp,
                {"all_of": ["test.jsonl", "test2.jsonl", "test3.jsonl", "gone.jsonl"]},
                owner="task 'selfcheck' split 'test'",
            )
        except FileNotFoundError as exc:
            for needle in ("gone.jsonl", "selfcheck", "all_of"):
                if needle not in str(exc):
                    raise AssertionError(
                        f"partial all_of error must name {needle!r}: {exc}"
                    ) from exc
        else:
            raise AssertionError(
                "a partial all_of shard set was merged instead of refused"
            )
        # Nothing on disk at all is *not* that error: the declared paths come
        # back verbatim so the caller's own message can name them.
        got = resolve_split_files(tmp, {"all_of": ["nope1.jsonl", "nope2.jsonl"]})
        if len(got) != 2 or any(os.path.exists(path) for path in got):
            raise AssertionError(
                f"an empty all_of must return the declared paths, got {got}"
            )

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
        "[self_check] split layout semantics ok (any_of single-layout / all_of "
        "merged / partial all_of refused)"
    )


def _check_partial_split_deferred():
    """A broken cache must fail its own task, not every task in the registry.

    `local_dataset_map()` resolves *all* of datasets.json, so raising while
    building it would block an eval of a healthy benchmark just because another
    one's shards are incomplete. The partial-shard error is stored
    (`UnresolvedSplit`) and raised by whoever reads that split -- still loud,
    but scoped to the data actually in use.
    """
    saved_reg = config_module.load_dataset_registry
    saved_assets = config_module.BENCH_ASSETS
    with tempfile.TemporaryDirectory() as tmp:
        for sub in ("good", "partial"):
            os.makedirs(os.path.join(tmp, sub))
        open(os.path.join(tmp, "good", "test.jsonl"), "w", encoding="utf-8").close()
        for name in ("test.jsonl", "test2.jsonl"):
            open(os.path.join(tmp, "partial", name), "w", encoding="utf-8").close()
        reg = {
            "good_task": {
                "repo_id": "selfcheck/good",
                "subdir": "good",
                "splits": {"test": "test.jsonl"},
            },
            "partial_task": {
                "repo_id": "selfcheck/partial",
                "subdir": "partial",
                "splits": {
                    "test": {"all_of": ["test.jsonl", "test2.jsonl", "test3.jsonl"]}
                },
            },
        }
        config_module.load_dataset_registry = lambda: reg
        config_module.BENCH_ASSETS = tmp
        try:
            mapping = config_module.local_dataset_map()
            held = mapping[("selfcheck/partial", None)]["test"]
            if not isinstance(held, UnresolvedSplit):
                raise AssertionError(
                    f"a partial all_of split must be deferred as UnresolvedSplit, "
                    f"got {held!r}"
                )
            files = local_dataset_files("good_task", reg, mapping)
            if len(files) != 1:
                raise AssertionError(
                    f"an unrelated healthy task broke because another task's "
                    f"cache is partial: {files}"
                )
            try:
                local_dataset_files("partial_task", reg, mapping)
            except FileNotFoundError as exc:
                for needle in ("partial_task", "test3.jsonl"):
                    if needle not in str(exc):
                        raise AssertionError(
                            f"deferred split error must name {needle!r}: {exc}"
                        ) from exc
            else:
                raise AssertionError(
                    "the task with a partial cache was served a truncated shard "
                    "set instead of failing"
                )
        finally:
            config_module.load_dataset_registry = saved_reg
            config_module.BENCH_ASSETS = saved_assets
    print("[self_check] partial all_of defers to the task that reads it")


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
    """Synthesize an upstream file list from registry declarations for one repo.

    This is what keeps the documented extension path true -- "add a benchmark =
    add a datasets.json entry, no code change here" (bench_offline.py). Without
    it a new task made the *default* `self_check.py` fail with
    `--remote false-positive on a correct registry: <task>: cannot list
    <repo_id>`, i.e. a fixture gap dressed up as a registry bug.

    `spec` is one entry or a list of entries that share a repo_id: several tasks
    may read the same repo (datasets.json already does that for arc_challenge /
    arc_easy on allenai/ai2_arc and boolq / copa on aps/super_glue; both repos
    only escape this branch because they have hand-written stubs). Deriving from
    the first entry alone leaves the second task's shards out of the listing, so
    the offline regression reports a *stub* gap as a registry false positive.
    Union them.

    A derived listing is self-consistent with the declaration by construction,
    so for such a repo the offline step covers the any_of/all_of matching logic
    plus whether `allow_patterns` would really download every declared glob
    (prefetch downloads by `allow_patterns`; a declared glob it filters out can
    never resolve locally). Whether the globs fit the *real* upstream layout
    stays `--remote`'s job -- hand-written stubs exist for the known repos
    precisely because they pin layouts worth pinning (MATH-500 shipping parquet
    only, LiveCodeBench's six cumulative shards).
    """
    specs = spec if isinstance(spec, (list, tuple)) else [spec]
    files = ["README.md"]
    for one in specs:
        for rels in one["splits"].values():
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
    # Group by repo_id: one repo can back several tasks, and a derived listing
    # must cover every task that reads it (see _derived_listing).
    by_repo = {}
    for task in sorted(reg):
        by_repo.setdefault(reg[task]["repo_id"], []).append(reg[task])
    for repo_id in sorted(wanted):
        if repo_id in _FAKE_UPSTREAM:
            file_lists[repo_id] = list(_FAKE_UPSTREAM[repo_id])
            continue
        file_lists[repo_id] = _derived_listing(by_repo[repo_id])
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


def _probe_entry_ssec(allow_patterns):
    """A *second* task on the same unstubbed repo, different subdir + layout.

    Regression for the derived-listing gap: `_upstream_stubs` used to take
    `next(s for s in reg.values() if s["repo_id"] == repo_id)`, so with two
    tasks on one repo only the first one's shards made it into the synthesized
    upstream listing, and the default `self_check.py` then failed with
    `--remote false-positive on a correct registry: gsm8k_ssec split 'test'
    (all_of): declared [...] unmatched [...]` -- a fixture gap reported as a
    registry bug. datasets.json already has this shape (arc_challenge /
    arc_easy), it just hides behind hand-written stubs.
    """
    return {
        "repo_id": "openai/gsm8k",
        "subdir": "gsm8k_ssec",
        "dataset_name": "ssec",
        "splits": {"test": {"all_of": ["ssec/test-*.parquet", "ssec/test2-*.parquet"]}},
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
    """`--remote` must not report a correct registry.

    A false positive sends the maintainer to "fix" a datasets.json that was
    right, and no runtime path catches it -- the run never consults `--remote`.
    The two shapes that produced one:

    - a dict declaration (`{"any_of": [...]}`) normalized with
      `isinstance(rels, str)` only, so `for rel in rels` yielded the mode key
      `any_of` / `all_of`, which never fnmatches -- math500 and
      livecodebench_release_v6 failed against a correct datasets.json;
    - `_upstream_stubs` taking `next(s for s in reg.values() if ...)`, so with
      two tasks on one repo only the first one's shards reached the synthesized
      listing and the second task read as a registry bug.

    One negative control (a task whose files all vanished) keeps the end-to-end
    half from passing vacuously.
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

    # 2. Matching: every layout a repo may legitimately ship must pass.
    for mode, rel_list, fetched in (
        ("any_of", ["test.jsonl", "data/test-*.parquet"], ["data/test-00000.parquet"]),
        (
            "any_of",
            ["test.jsonl", "data/test-*.parquet"],
            ["test.jsonl", "data/test-00000.parquet"],
        ),
        ("all_of", ["test.jsonl", "test2.jsonl"], ["test.jsonl", "test2.jsonl"]),
    ):
        got = _unmatched_globs(mode, rel_list, fetched)
        if got:
            raise AssertionError(
                f"{mode} {rel_list} against {fetched} is a valid layout, "
                f"got unmatched {got}"
            )

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

    # 4. Not vacuous: a task whose files all vanished is still reported.
    broken = {k: list(v) for k, v in file_lists.items()}
    broken[_repo_of(reg, "sciq")] = ["README.md"]
    problems = _run_remote_check(broken, reg=reg)
    if "sciq" not in problems:
        raise AssertionError(
            "--remote must report a task whose files all vanished, "
            f"got:\n{problems or '<no problems>'}"
        )

    # 5. A task added to datasets.json *alone* (no stub edit) must keep the
    #    default self-check green.
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

    # 6. Two tasks sharing one unstubbed repo_id: the derived listing must
    #    cover both, or the second task reads as a registry error.
    probe["gsm8k_ssec"] = _probe_entry_ssec(["*.parquet", "README.md"])
    probe_lists, _ = _upstream_stubs(probe)
    shared = probe_lists["openai/gsm8k"]
    if not any(f.startswith("main/") for f in shared) or not any(
        f.startswith("ssec/") for f in shared
    ):
        raise AssertionError(
            "derived listing for a repo with two tasks must union both tasks' "
            f"declared paths, got {shared}"
        )
    problems = _run_remote_check(probe_lists, reg=probe)
    if problems:
        raise AssertionError(
            "two tasks on one unstubbed repo_id must not break the default "
            f"self-check (derived listing has to aggregate them):\n{problems}"
        )
    print(
        "[self_check] remote glob semantics ok (no mode-key leak / valid layout "
        "passes / new task needs no stub / two tasks may share one repo)"
    )


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
        if "--algos" not in _help_output(script):
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


def _check_attn_target_from_path():
    """An attention param_dir names its target by path component, or the run says so.

    `attn-linear` and `attn-cache` save the same filename
    (`layer_<idx>_self_attn.pt`; constraints.md #1), so the files cannot say
    which PTQ round produced them. Assuming `attn-linear` for an attn-cache
    checkpoint loads cache-quantizer params into linear quantizers: a mixed
    object that still prints a score. The directory is the only surviving
    record of the target, so a path that names it must be read, and a path
    that does not must keep the fallback *and* warn that it was assumed.
    """
    from assemble_and_eval import (
        _attn_target_from_path,
        _preflight_param_dir,
        _resolve_quant_target,
    )

    # Exact component compare: a near-miss leaf and a path naming both targets
    # are the two ways "reads the path" could silently over-match.
    cases = {
        "/data/ptq_params/qwen3/attn-linear": "attn-linear",
        "/data/ptq_params/qwen3/attn-cache": "attn-cache",
        "/data/ptq_params/qwen3/attn-cache/": "attn-cache",
        "/data/ptq_params/qwen3/attn-cache-exp3": "",
        "/data/attn-linear/attn-cache": "",
        "/data/foo/bar": "",
    }
    for path, want in cases.items():
        got = _attn_target_from_path(path)
        if got != want:
            raise AssertionError(
                f"_attn_target_from_path({path!r}) = {got!r}, expected {want!r}"
            )

    with tempfile.TemporaryDirectory() as root:

        def _param_dir(leaf, files=("layer_0_self_attn.pt",)):
            d = os.path.join(root, leaf)
            os.makedirs(d, exist_ok=True)
            for name in files:
                open(os.path.join(d, name), "w", encoding="utf-8").close()
            return d

        named = _param_dir("attn-cache")
        unnamed = _param_dir("round1")
        mixed = _param_dir("attn-linear", ("layer_0_mlp.pt", "layer_0_self_attn.pt"))

        if _resolve_quant_target("qwen3", None, named) != "attn-cache":
            raise AssertionError("an attn-cache param_dir did not derive attn-cache")
        if _resolve_quant_target("qwen3", None, unnamed) != "attn-linear":
            raise AssertionError(
                "an unnamed param_dir did not fall back to attn-linear"
            )
        if _resolve_quant_target("qwen3", None, mixed) != "mlp,attn-linear":
            raise AssertionError("an mlp + attention param_dir lost its class union")
        if _resolve_quant_target("qwen3", "mlp", named) != "mlp":
            raise AssertionError("explicit --quant_target did not win over the path")

        # The warn is the whole point of the fallback: it must fire exactly
        # when the target was assumed, and stay quiet when it was read or
        # passed in (a warn on every attn run trains the reader to skip it).
        for param_dir, qt, want_warn in (
            (unnamed, "", True),
            (named, "", False),
            (unnamed, "attn-cache", False),
        ):
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                _preflight_param_dir(param_dir, quant_target=qt)
            out = buf.getvalue()
            if ("--quant_target attn-cache" in out) != want_warn:
                raise AssertionError(
                    f"_preflight_param_dir({param_dir}, quant_target={qt!r}) "
                    f"warned={not want_warn}, expected want_warn={want_warn}: {out!r}"
                )
    print(
        "[self_check] attention target read off param_dir path "
        f"({len(cases)} paths, warn only when assumed)"
    )


def _check_input_device_fallback():
    """`input_device()` must not guess `npu:0` when it cannot tell.

    With `--device_map auto` the embedding can sit on any card; feeding
    input_ids to a hardcoded npu:0 dies with a cross-device error deep inside
    forward -- or runs on a card that holds no weights. Requires torch.
    """
    try:
        import torch
    except ImportError as exc:
        print(f"[self_check] skip input_device check: {exc}")
        return
    from assemble_and_eval import input_device

    class _HasDevice:
        device = torch.device("cpu")

        def get_input_embeddings(self):
            raise AttributeError("stub has no embeddings")

    if input_device(_HasDevice()) != torch.device("cpu"):
        raise AssertionError("input_device ignored model.device")

    class _Opaque:
        def get_input_embeddings(self):
            raise AttributeError("stub has no embeddings")

    try:
        input_device(_Opaque())
    except RuntimeError as exc:
        if not isinstance(exc.__cause__, AttributeError):
            raise AssertionError(
                f"input_device dropped the reason it could not tell: {exc!r}"
            ) from exc
        if "npu:0" in str(exc):
            raise AssertionError("input_device still falls back to a hardcoded npu:0")
    else:
        raise AssertionError(
            "input_device invented a device for a model that exposes none"
        )
    print("[self_check] input_device refuses to guess a device")


def _check_plan_device_preflight():
    """`plan_device.py` must answer the card question cheaply and actionably.

    `--device_map` and `ASCEND_RT_VISIBLE_DEVICES` are launch-time decisions,
    while the `[device]` line that reveals the answer prints during the load:
    guessing wrong costs one full load and then an offload rejection. The entry
    point exists to move that answer ahead of the load, and these properties of
    it are invisible to a unit test of the shared formula:

    - it stays cheap. `assemble_and_eval.py` imports torch at module level, so
      no flag on it could ever answer in less than the interpreter's torch
      start-up; that is why the logic lives in `utils.py` behind its own CLI.
      Asserted structurally -- importing the module must not pull torch in --
      because a wall-clock assertion is flaky on a loaded machine.
    - the answer names both knobs the caller has to turn, or the pre-flight
      informs without being actionable.
    - the card list scales with the checkpoint. Both knobs are launch-time and
      the budget is per card, so a size that needs eight cards must not be told
      two suffice; the expectation is derived from the shared formula rather
      than pinned as a literal, and a floor keeps a regression to a constant
      short list red even if the budget moves.
    - an unknowable size exits non-zero. Returning 0 there would be the pre-flight
      guessing, which is the behaviour `utils.plan_devices` refuses to have.
    """
    from utils import cards_for_weights, estimate_weight_bytes

    script = os.path.join(SKILL_DIR, "plan_device.py")
    # 8B dense vs a 35B MoE: the two sides of `SINGLE_CARD_WEIGHT_BUDGET`.
    cases = (
        (
            "small",
            {
                "num_hidden_layers": 36,
                "hidden_size": 4096,
                "intermediate_size": 12288,
                "vocab_size": 151936,
            },
            "--device_map ''",
            1,
        ),
        (
            "big",
            {
                "num_hidden_layers": 62,
                "hidden_size": 4096,
                "intermediate_size": 12288,
                "vocab_size": 151936,
                "num_experts": 128,
                "moe_intermediate_size": 1536,
                "shared_expert_intermediate_size": 4096,
            },
            "--device_map 'auto'",
            3,
        ),
    )

    def _run(tmp, model_dir):
        return subprocess.run(
            [sys.executable, script, "--model", model_dir],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=tmp,
        )

    with tempfile.TemporaryDirectory() as tmp:
        for name, cfg, want_flag, min_cards in cases:
            model_dir = os.path.join(tmp, name)
            os.makedirs(model_dir)
            with open(
                os.path.join(model_dir, "config.json"), "w", encoding="utf-8"
            ) as f:
                json.dump(cfg, f)
            proc = _run(tmp, model_dir)
            if proc.returncode != 0:
                raise AssertionError(
                    f"plan_device.py on the {name} case exited "
                    f"{proc.returncode}: {proc.stderr.strip()}"
                )
            if want_flag not in proc.stdout:
                raise AssertionError(
                    f"plan_device.py on the {name} case never told the caller "
                    f"to set {want_flag}; got: {proc.stdout.strip()}"
                )
            found = re.search(r"ASCEND_RT_VISIBLE_DEVICES=(\d+(?:,\d+)*)", proc.stdout)
            if not found:
                raise AssertionError(
                    f"plan_device.py on the {name} case never named "
                    f"ASCEND_RT_VISIBLE_DEVICES with a card list; "
                    f"got: {proc.stdout.strip()}"
                )
            printed = found.group(1).split(",")
            want_cards = cards_for_weights(estimate_weight_bytes(model_dir))
            if len(printed) != want_cards:
                raise AssertionError(
                    f"plan_device.py on the {name} case printed {len(printed)} "
                    f"cards for a checkpoint the shared budget puts at "
                    f"{want_cards}: {proc.stdout.strip()}"
                )
            if len(printed) < min_cards:
                raise AssertionError(
                    f"plan_device.py on the {name} case printed a card list too "
                    f"short to hold the checkpoint ({found.group(1)}); "
                    f"got: {proc.stdout.strip()}"
                )
        # Control: no num_hidden_layers -> the size is unknowable.
        bad_dir = os.path.join(tmp, "unknowable")
        os.makedirs(bad_dir)
        with open(os.path.join(bad_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump({"hidden_size": 4096}, f)
        proc = _run(tmp, bad_dir)
        if proc.returncode == 0:
            raise AssertionError(
                "plan_device.py exited 0 on a config.json that declares no layer "
                f"count, i.e. it guessed: {proc.stdout.strip()}"
            )
        if "--device_map" not in proc.stdout:
            raise AssertionError(
                "the unknowable-size exit does not name the flag the caller must "
                f"pass instead: {proc.stdout.strip()}"
            )

    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; sys.path.insert(0, sys.argv[1]); import plan_device; "
            "assert 'torch' not in sys.modules, sorted(sys.modules)[:5]",
            SKILL_DIR,
        ],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if proc.returncode != 0:
        raise AssertionError(
            "importing plan_device pulled torch in, so the pre-flight is no "
            f"longer cheaper than loading: {proc.stderr.strip()}"
        )
    print("[self_check] plan_device preflight ok (2 sizes + unknowable control)")


_TASK_MANAGER = None


def _task_manager():
    """Cached lm-eval ``TaskManager``.

    Building one costs ~10s (importing lm_eval itself is 0.08s) and two checks
    need it -- `_check_skill_metric_keys` and `_check_task_offline_registration`
    each built their own, which is most of a 48s self-check run.
    """
    global _TASK_MANAGER
    from lm_eval.tasks import TaskManager

    if _TASK_MANAGER is None:
        _TASK_MANAGER = TaskManager()
    return _TASK_MANAGER


_HELP_CACHE = {}


def _help_output(script):
    """Cached ``<script> --help`` stdout.

    Spawning one costs ~5s, and `_check_cli_help` (5 scripts) plus
    `_check_cli_flags` (2 of the same scripts) spawned the entry points twice.
    """
    if script not in _HELP_CACHE:
        proc = subprocess.run(
            [sys.executable, os.path.join(SKILL_DIR, script), "--help"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if proc.returncode != 0:
            raise AssertionError(
                f"{script} --help failed with {proc.returncode}: {proc.stderr}"
            )
        _HELP_CACHE[script] = proc.stdout
    return _HELP_CACHE[script]


_TABLE_SEPARATOR_RE = re.compile(r"^\|[\s:|-]+\|$")


def _table_cells(line):
    """Split one markdown table row into stripped cell strings."""
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def _table_blocks(lines):
    """Yield every well-formed markdown table in `lines` as a list of rows.

    A block qualifies when it has a header row, a `|---|` separator under it and
    at least one data row, so a stray `|`-leading prose line is never treated as
    a table. Callers index the block: [0] header, [1] separator, [2:] data.
    """
    block = []
    for line in list(lines) + [""]:
        if line.lstrip().startswith("|"):
            block.append(line)
            continue
        if (
            len(block) >= 3
            and _TABLE_SEPARATOR_RE.match(block[1].strip())
            and not _TABLE_SEPARATOR_RE.match(block[0].strip())
        ):
            yield block
        block = []


def _parse_metric_table(path):
    """Read (task, metric) pairs out of SKILL.md's benchmark table.

    Columns are located **by header name**: the table whose header has a column
    containing 任务 and one containing 指标 is the benchmark table, and those two
    columns supply the pairs. Any number of names may be backticked in a cell
    (e.g. `acc,none` kept next to the default metric).

    Header names, not cell wording: the previous version required column 1 to
    start with 低 / 高 and read columns 2 / 3 unconditionally. Rewording the
    cost column then turned `_check_skill_metric_keys` red with "no metric table
    rows parsed" -- an error that names neither the real rule nor where it lives
    (the rule was a regex in this file, an implicit layout constraint on the
    doc). Tables that carry no 任务+指标 header pair are skipped, so §5's
    failure-mode table never matches.
    """
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        lines = [line.rstrip("\n") for line in f]
    for block in _table_blocks(lines):
        header_cells = _table_cells(block[0])
        task_cols = [i for i, h in enumerate(header_cells) if "任务" in h]
        metric_cols = [i for i, h in enumerate(header_cells) if "指标" in h]
        if not task_cols or not metric_cols:
            continue
        task_col, metric_col = task_cols[0], metric_cols[0]
        for line in block[2:]:
            cells = _table_cells(line)
            if len(cells) <= max(task_col, metric_col):
                raise AssertionError(
                    f"benchmark table row has {len(cells)} cells, needs index "
                    f"{max(task_col, metric_col)}: {line.strip()}"
                )
            tasks = re.findall(r"`([^`]+)`", cells[task_col])
            metrics = re.findall(r"`([^`]+)`", cells[metric_col])
            if not tasks or not metrics:
                raise AssertionError(
                    f"benchmark table row has no backticked name in the 任务 or "
                    f"指标 column: {line.strip()}"
                )
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
    for task, metric in rows:
        if task in NON_LM_EVAL_TASKS:
            continue
        lm_task = TASK_ALIASES.get(task, task)
        try:
            task_manager = _task_manager()
        except ImportError:
            print("[self_check] skip metric-key check: lm-eval not installed")
            return
        keys = resolve_task_metric_keys(lm_task, task_manager)
        if metric not in keys:
            raise AssertionError(
                f"SKILL.md documents metric '{metric}' for task '{task}', "
                f"but lm-eval emits {keys} for '{lm_task}'"
            )
    print(f"[self_check] metric keys ok ({len(rows)} rows)")


def _check_task_offline_registration():
    """Every lm-eval task this skill advertises must resolve to registered data.

    `check_task_offline_ready()` is the pre-flight that stops a typo'd or
    unregistered `--task` before the model is loaded; if it were to pass for a
    dataset that is not in datasets.json, the run would instead die on the hub
    request minutes into evaluation (constraints.md #2). Verify both
    directions here, against lm-eval's installed task index -- no download, no
    NPU. Skipped when lm-eval is absent.
    """
    try:
        task_manager = _task_manager()
    except ImportError as exc:
        print(f"[self_check] skip task-registration check: {exc}")
        return
    reg = load_dataset_registry()
    checked = 0
    for task in sorted(EXPECTED_TASKS & set(reg)):
        if task in NON_LM_EVAL_TASKS:
            continue
        lm_task = TASK_ALIASES.get(task, task)
        # math500 -> hendrycks_math500 declares dataset_name 'default' while
        # datasets.json keys the repo with no name: the shim's (path, None)
        # fallback must make that pass rather than demand a second entry.
        wanted = check_task_offline_ready(lm_task, reg, task_manager)
        if not wanted:
            raise AssertionError(f"task '{lm_task}' resolved to no dataset key")
        checked += 1
    if checked < 5:
        raise AssertionError(
            f"only {checked} registry tasks cross-checked against lm-eval; "
            "the advertised task list must stay covered"
        )
    # Negative controls: a misspelling and a task whose dataset was never
    # registered are the two mistakes this pre-flight exists to catch, and they
    # must be told apart (index miss vs. registry miss) in the message.
    try:
        check_task_offline_ready("arc-challenge", reg, task_manager)
    except RuntimeError as exc:
        msg = str(exc)
        if "not in the lm-eval task index" not in msg:
            raise AssertionError(
                f"unknown task is not reported as an index miss: {exc}"
            )
        if "TIGER-Lab/MMLU-Pro" not in msg:
            raise AssertionError(
                f"unknown task error does not list registered repos: {exc}"
            )
    else:
        raise AssertionError("unknown task 'arc-challenge' passed the pre-flight check")
    try:
        check_task_offline_ready("gsm8k", reg, task_manager)
    except RuntimeError as exc:
        if "openai/gsm8k" not in str(exc) or "datasets.json" not in str(exc):
            raise AssertionError(
                f"unregistered dataset error does not name the missing pair: {exc}"
            )
    else:
        raise AssertionError(
            "task 'gsm8k' has no datasets.json entry but passed the pre-flight check"
        )

    # A task lm-eval *knows* but that declares no dataset of its own is a
    # different mistake: no datasets.json entry could fix it, so telling the
    # user to "check the spelling" sends them the wrong way.
    class _StubIndex:
        def __init__(self, index):
            self.task_index = index

    stub = _StubIndex({"no_dataset": types.SimpleNamespace(cfg={"task": "no_dataset"})})
    try:
        check_task_offline_ready("no_dataset", reg, stub)
    except RuntimeError as exc:
        msg = str(exc)
        if (
            "declares no dataset_path" not in msg
            or "cannot be served offline" not in msg
        ):
            raise AssertionError(
                f"a dataset-less task is not reported as such: {exc}"
            ) from exc
    else:
        raise AssertionError("a task with no dataset_path passed the pre-flight")
    try:
        check_task_offline_ready("no_such_task", reg, _StubIndex({}))
    except RuntimeError as exc:
        if "cannot be served offline" in str(exc):
            raise AssertionError(
                f"an index miss borrowed the dataset-less wording: {exc}"
            )
    else:
        raise AssertionError("an unknown task passed the pre-flight")
    # A registry that does not actually declare the task's repo must fail too:
    # the check has to read the registry, not just the task index.
    try:
        check_task_offline_ready(
            "arc_easy", {"selfcheck": {"repo_id": "other/repo"}}, task_manager
        )
    except RuntimeError:
        pass
    else:
        raise AssertionError(
            "arc_easy passed against a registry that declares no allenai/ai2_arc"
        )
    print(f"[self_check] task offline registration ok ({checked} tasks)")


def _check_cli_help():
    """Every CLI entry point must still answer --help."""
    for script in CLI_SCRIPTS:
        _help_output(script)


def _self_ast_index():
    """``{name: FunctionDef}`` for every ``_check_*`` defined in this file."""
    with open(os.path.abspath(__file__), encoding="utf-8") as f:
        tree = ast.parse(f.read())
    return {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name.startswith("_check_")
    }


def _self_line_count():
    """This file's own line count, printed on every green run.

    Not an assert: a hard budget goes red for a legitimate addition and gets
    bumped, which teaches the reader to ignore it. Printing the size instead
    puts the current volume in front of whoever reads the run, so growth is
    visible at the moment it happens rather than at the next audit.
    """
    with open(os.path.abspath(__file__), encoding="utf-8") as f:
        return sum(1 for _ in f)


def _prints_on_success(node):
    """True when a check calls ``print()``, i.e. a green run shows a line for it."""
    return any(
        isinstance(n, ast.Call) and getattr(n.func, "id", None) == "print"
        for n in ast.walk(node)
    )


def _first_doc_line(node):
    doc = (ast.get_docstring(node) or "").strip()
    return doc.splitlines()[0].strip() if doc else "(no docstring)"


def _list_checks():
    """Print every check as ``[group][prints|silent] name: docstring first line``.

    `prints` vs `silent` is derived from each function's AST, not from a
    hand-maintained table: a reader of a green run needs to know which checks
    passed without saying anything, otherwise "no output" and "not wired" look
    identical.
    """
    index = _self_ast_index()
    for group, checks in (("default", DEFAULT_CHECKS), ("--remote", REMOTE_CHECKS)):
        for fn in checks:
            node = index[fn.__name__]
            tag = "prints" if _prints_on_success(node) else "silent"
            print(f"[{group}][{tag}] {fn.__name__}: {_first_doc_line(node)}")
    print(
        f"[self_check] {len(DEFAULT_CHECKS)} default + "
        f"{len(REMOTE_CHECKS)} --remote check(s)"
    )


def _check_checklist_wiring():
    """Every `_check_*` defined here is wired into exactly one run list.

    A check that is defined but never called protects nothing while still
    reading as coverage, and one listed twice doubles the runtime (the lm-eval
    index build alone is ~10s). This also verifies `--list-checks` renders
    exactly the wired set, so that output stays the only list of checks -- no
    doc keeps a copy of it.
    """
    index = _self_ast_index()
    wired = [fn.__name__ for fn in DEFAULT_CHECKS + REMOTE_CHECKS]
    never_called = sorted(set(index).difference(wired))
    if never_called:
        raise AssertionError(
            f"defined but never called: {never_called} "
            "(add to DEFAULT_CHECKS or REMOTE_CHECKS)"
        )
    undefined = [name for name in wired if name not in index]
    if undefined:
        raise AssertionError(f"listed without a definition: {undefined}")
    dupes = sorted({name for name in wired if wired.count(name) > 1})
    if dupes:
        raise AssertionError(f"listed more than once: {dupes}")
    if not REMOTE_CHECKS:
        raise AssertionError("REMOTE_CHECKS is empty: --remote would be a no-op")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        _list_checks()
    rendered = [
        line
        for line in buf.getvalue().splitlines()
        if line.startswith("[default]") or line.startswith("[--remote]")
    ]
    if len(rendered) != len(wired):
        raise AssertionError(
            f"--list-checks rendered {len(rendered)} rows for {len(wired)} checks"
        )
    listed = {line.split("] ", 1)[1].split(":", 1)[0] for line in rendered}
    if listed != set(wired):
        raise AssertionError(
            f"--list-checks out of sync: {sorted(listed ^ set(wired))}"
        )
    print(
        f"[self_check] check wiring ok ({len(wired)} checks, "
        f"--list-checks in sync, {_self_line_count()} lines in this file)"
    )


def _doc_scope(path):
    """`(full text, {"N": section body})` for the `## N.` sections of one doc.

    `### N.M` sub-headings fall inside their parent section's body: the regex
    only matches `## <digits>.`, so a citation of `#4` covers 4.1-4.3 as well.
    """
    with open(path, encoding="utf-8") as f:
        lines = f.read().splitlines()
    sections, current = {}, None
    for line in lines:
        head = re.match(r"^##\s+(\d+)\.", line)
        if head:
            current = head.group(1)
            sections.setdefault(current, [])
        elif current is not None:
            sections[current].append(line)
    return "\n".join(lines), sections


def _cited_doc(name):
    """Absolute path of a doc an allow-list entry may cite, or None."""
    root = os.path.dirname(SKILL_DIR)
    for cand in (os.path.join(root, name), os.path.join(root, "references", name)):
        if os.path.isfile(cand):
            return cand
    return None


# Numbered section citations, in every spelling this tree uses: `<doc>.md §4.2`
# / `<doc>.md #1` (cross-file; the `)` of a markdown link may sit between),
# bare `§N` (the same doc -- or SKILL.md when the citation is in a script), and
# bare `#N` inside a .md (constraints.md's trigger index routes to its own
# entries). A bare `#N` in a .py file is a comment, not a citation.
_CITE_RE = re.compile(
    r"(?:(?P<doc>[A-Za-z0-9_\-]+\.md)\)?\s*)?[§#]\s*(?P<sec>\d+(?:\.\d+)?)"
)


def _citation_files():
    """`[(label, path, default_doc)]` for every doc and script in this skill."""
    ref_dir = os.path.join(os.path.dirname(SKILL_DIR), "references")
    files = [("SKILL.md", SKILL_MD, "SKILL.md")]
    files += [
        (f"references/{n}", os.path.join(ref_dir, n), n)
        for n in sorted(os.listdir(ref_dir))
        if n.endswith(".md")
    ]
    files += [
        (f"scripts/{n}", os.path.join(SKILL_DIR, n), "SKILL.md")
        for n in sorted(os.listdir(SKILL_DIR))
        if n.endswith(".py")
    ]
    return files


def _citation_sites():
    """`[(label, doc, section)]` for every numbered citation in the skill."""
    sites = []
    for label, path, default_doc in _citation_files():
        with open(path, encoding="utf-8") as f:
            lines = f.read().splitlines()
        is_md = path.endswith(".md")
        in_fence = False
        for line in lines:
            if is_md and line.lstrip().startswith("```"):
                in_fence = not in_fence
                continue
            if is_md and not in_fence and re.match(r"^#{1,6}\s", line):
                continue  # a heading numbers itself; it cites nothing
            for m in _CITE_RE.finditer(line):
                if not m.group("doc") and "§" not in m.group(0):
                    # a bare `#N` is a comment in Python and in a shell block
                    if not is_md or in_fence:
                        continue
                sites.append((label, m.group("doc") or default_doc, m.group("sec")))
    return sites


def _validate_citations(sites):
    """Raise unless every cited section really is a heading of the cited doc."""
    scopes, offenders = {}, []
    for label, doc, sec in sites:
        if doc not in scopes:
            path = _cited_doc(doc)
            scopes[doc] = None if path is None else ((path,) + _doc_scope(path))
        entry = scopes[doc]
        if entry is None:
            offenders.append(f"{label} cites {doc} §{sec}: no such doc")
            continue
        path, full, sections = entry
        top = sec.split(".")[0]
        if top not in sections:
            offenders.append(
                f"{label} cites {doc} §{sec}: {doc} has no `## {top}.` "
                f"(have: {sorted(sections, key=int)})"
            )
        elif "." in sec and not re.search(
            rf"^#{{3,6}}\s+{re.escape(sec)}(?=\s|$)", full, re.M
        ):
            offenders.append(f"{label} cites {doc} §{sec}: no `### {sec}` heading")
    if offenders:
        raise AssertionError(
            "dead section citations -- a renumbering was not propagated: "
            + "; ".join(offenders)
        )


def _check_citation_targets():
    """Every numbered section citation in the skill resolves to a real heading.

    Compressing SKILL.md from 7 sections to 6 left every citation behind: three
    argparse help strings sent `--help` readers to a section the compression had
    deleted, and this file's own docstrings kept calling the failure table by its
    old number. Nothing else looks at a citation's target, so the drift stayed
    green while sitting in user-visible help. This scans every numbered
    citation in `SKILL.md`, `references/*.md` and
    `scripts/*.py` instead, so the next renumbering cannot ship dead pointers.

    Spelling convention: a `.md` writes
    `<doc>.md §N`, a `.py` writes `<doc>.md #N` -- a bare `#N` in Python is a
    comment, so the doc qualifier is what makes it a citation. A bare `#N` in a
    `.md` is a self-citation (constraints.md's trigger index routes to its own
    entries) and stays.
    """
    sites = _citation_sites()
    if not sites:
        raise AssertionError(
            "no section citations parsed: the scanner stopped matching, so dead "
            "pointers would go unnoticed"
        )
    _validate_citations(sites)
    stale = f"{9}.{9}"  # built at runtime: a literal would be scanned as a citation
    try:
        _validate_citations(list(sites) + [("scripts/_control.py", "SKILL.md", stale)])
    except AssertionError:
        pass
    else:
        raise AssertionError(
            "a citation to a section that never existed passed: the scan "
            "validates nothing"
        )
    print(f"[self_check] citation targets ok ({len(sites)} citations)")


def _check_reference_mentions():
    """Every references/*.md is linked from SKILL.md, and every link resolves.

    `scripts/*.py` already had this two-way check; references did not, so a new
    reference file nobody links becomes an orphan no reader ever reaches --
    progressive disclosure silently loses a whole file, and a stale link sends a
    reader to a path that is not there.
    """
    ref_dir = os.path.join(os.path.dirname(SKILL_DIR), "references")
    on_disk = {n for n in os.listdir(ref_dir) if n.endswith(".md")}
    with open(SKILL_MD, encoding="utf-8") as f:
        skill_text = f.read()
    _, docs = _doc_sources()
    linked = set()
    for _, text in docs:
        linked.update(re.findall(r"references/([A-Za-z0-9_.\-]+\.md)", text))
    orphan = sorted(
        on_disk - set(re.findall(r"references/([A-Za-z0-9_.\-]+\.md)", skill_text))
    )
    if orphan:
        raise AssertionError(
            f"references/*.md never linked from SKILL.md (the agent's only entry "
            f"point): {orphan}"
        )
    dangling = sorted(linked - on_disk)
    if dangling:
        raise AssertionError(f"docs link references that do not exist: {dangling}")
    print(f"[self_check] references linked both ways ({len(on_disk)} files)")


def _trigger_index_rows(lines):
    """Data-row count of constraints.md's 触发条件索引 table.

    One row per constraint is what makes 按需读 safe to promise: the index is
    the reader's only router into the file, so a constraint without a row is a
    constraint the router never sends anybody to.
    """
    start = None
    for i, line in enumerate(lines):
        if line.startswith("## 触发条件索引"):
            start = i + 1
        elif start is not None and line.startswith("## "):
            body = lines[start:i]
            break
    else:
        body = lines[start:] if start is not None else None
    if start is None or not body:
        raise AssertionError(
            "constraints.md has no `## 触发条件索引` section with a table, so "
            "按需读 has nothing to route by"
        )
    blocks = list(_table_blocks(body))
    if not blocks:
        raise AssertionError("constraints.md's 触发条件索引 section has no table")
    return len(blocks[0][2:])


def _cost_tier_rows(lines):
    """`{成本档 cell: row count}` for SKILL.md's benchmark table."""
    for block in _table_blocks(lines):
        header = _table_cells(block[0])
        if not any("任务" in h for h in header) or not any("指标" in h for h in header):
            continue
        tiers_col = [i for i, h in enumerate(header) if "成本档" in h]
        if not tiers_col:
            raise AssertionError(
                "SKILL.md's benchmark table has no 成本档 column: it is the "
                "column an agent picks a smoke-worthy task by"
            )
        col = tiers_col[0]
        tiers = {}
        for line in block[2:]:
            cells = _table_cells(line)
            tiers[cells[col]] = tiers.get(cells[col], 0) + 1
        return tiers
    raise AssertionError("no benchmark table (任务 + 指标 header) found in SKILL.md")


def _shared_param_rows(skill_lines):
    """Data rows of SKILL.md §4.2's「与量化侧共享的参数」table.

    Located by its header (`参数 | 从哪来`), not by its section number: the
    「结构税」 of renumbering killed a content-located check here once already.
    Exactly one such table, or this check could silently read the wrong one --
    §4.2 has one more 参数-headed table (必知参数) with a different second
    column.
    """
    hits = [
        block
        for block in _table_blocks(skill_lines)
        if _table_cells(block[0])[:2] == ["参数", "从哪来"]
    ]
    if not hits:
        raise AssertionError(
            "SKILL.md has no `参数 | 从哪来` table: §4.2's shared-parameter "
            "table is the only place an agent is told which settings must match "
            "the quantization run"
        )
    if len(hits) > 1:
        raise AssertionError(
            f"SKILL.md has {len(hits)} `参数 | 从哪来` tables; keep the "
            "shared-parameter table the only one so this check cannot pick "
            "the wrong one"
        )
    return hits[0][2:]


def _shared_param_offenders(rows, keys, declared):
    """`[reason]` for every object-side key the shared-parameter table omits."""
    table = "\n".join(rows)
    out = []
    for key in keys:
        flag = f"--{key}"
        if flag not in declared:
            out.append(f"{key}: no {flag} on either entry point")
        elif flag not in table:
            out.append(f"{key}: {flag} has no row in the shared-parameter table")
    return out


def _check_shared_param_table():
    """Every object-side comparability key is a row of §4.2's shared-param table.

    `compare_results.py`'s `_OBJECT_SIDE_KEYS` is what the code refuses to
    subtract across; the table is what an agent reads before taking over
    somebody else's run. A key in one and not the other means the tool now
    rejects a delta the docs never told anybody to align -- or promises an
    alignment nothing checks. Rows, not counts: the row is the thing that
    drifts and the thing the agent consumes, and a row needs no number in the
    prose to go stale.
    """
    sources, _ = _doc_sources()
    with open(SKILL_MD, encoding="utf-8") as f:
        skill_lines = f.read().splitlines()
    rows = _shared_param_rows(skill_lines)
    flags = _script_flags(sources)
    declared = set().union(*(flags[name] for name in _ENTRY_SCRIPTS))
    offenders = _shared_param_offenders(rows, OBJECT_SIDE_KEYS, declared)
    if offenders:
        raise AssertionError(
            "object-side comparability keys the shared-parameter table does not "
            f"cover: {offenders} -- row them in SKILL.md §4.2, or drop them from "
            "compare_results.py's _OBJECT_SIDE_KEYS if they are not comparable"
        )
    # Controls: a dropped row and an unknown key must both fail, otherwise the
    # pairing proves nothing.
    carrier = next(
        (r for r in rows if any(f"--{k}" in r for k in OBJECT_SIDE_KEYS)), None
    )
    if carrier is None:
        raise AssertionError("control found no row carrying an object-side key")
    if not _shared_param_offenders(
        [r for r in rows if r != carrier], OBJECT_SIDE_KEYS, declared
    ):
        raise AssertionError(
            "a dropped table row passed: the pairing does not compare rows"
        )
    if not _shared_param_offenders(rows, OBJECT_SIDE_KEYS + ("no_such_key",), declared):
        raise AssertionError("an unknown key passed: the scan ignores the key list")
    print(
        f"[self_check] shared param table ok "
        f"({len(OBJECT_SIDE_KEYS)} object-side keys, {len(rows)} rows)"
    )


def _metadata_payload():
    """An `--output_path` payload carrying every comparability key."""
    obj = {"results": {}, "tag": "quant"}
    for i, key in enumerate(COMPARABILITY_KEYS):
        obj[key] = i
    return obj


def _metadata_gaps(meta):
    """Comparability keys a metadata dict does not carry."""
    return [k for k in COMPARABILITY_KEYS if k not in meta]


def _check_metadata_covers_comparability_keys():
    """`_run_metadata` reads back every key `_check_comparable` gates on.

    A comparability key that is declared but never extracted cannot reach the
    gate: `_check_comparable` skips keys absent from the metadata, so the two
    sides always look like they agree and a delta across differing bit configs
    or algorithms prints as fact. The gate's key list is the only source here,
    so this re-derives coverage from it instead of pinning a count.
    """
    payload = _metadata_payload()
    gaps_in_payload = _metadata_gaps(payload)
    if gaps_in_payload:
        raise AssertionError(f"the synthetic payload omits {gaps_in_payload}")
    gaps = _metadata_gaps(run_metadata(payload))
    if gaps:
        raise AssertionError(
            f"_run_metadata does not read back these comparability keys: {gaps} "
            "-- compare_results.py cannot gate on a value it never extracts. "
            "Read it from the payload, or drop it from _COMPARABILITY_KEYS if "
            "it is not comparable"
        )
    # Controls: the coverage test has to be able to see a gap, and the
    # not-a-payload guard has to still return {} -- otherwise both assertions
    # above pass on an empty key list.
    dropped = {k: v for k, v in run_metadata(payload).items() if k != "task"}
    if not _metadata_gaps(dropped):
        raise AssertionError("a dropped key passed: the coverage test is vacuous")
    if run_metadata({"results": {}}) != {}:
        raise AssertionError(
            "_run_metadata returned metadata for an object that is not an "
            "--output_path payload (no 'tag'): a log or an lcb *_eval.json "
            "would then gate on keys it does not have"
        )
    if run_metadata({"tag": "quant"}) != {}:
        raise AssertionError("_run_metadata returned metadata without 'results'")
    print(
        f"[self_check] metadata covers comparability keys ok "
        f"({len(COMPARABILITY_KEYS)} keys read back)"
    )


def _check_comparable_side_tags():
    """The gate refuses two runs that are not baseline-minus-quantized.

    `--direct` builds no algorithm module, so it writes `algos: None` exactly
    like a bf16 baseline does, and the object-side None exemption cannot tell
    them apart: a direct-conversion run passed as `--bf16` against a PTQ run
    gated clean and the difference went out as `delta_quant_minus_bf16`, next
    to `compared_from: "run metadata (both sides)"`. `tag` is the only field in
    a payload that says which side a run was.
    """

    def meta(tag, **over):
        obj = {key: "shared" for key in COMPARABILITY_KEYS}
        obj.update({key: None for key in OBJECT_SIDE_KEYS}, tag=tag)
        obj.update(over)
        return obj

    def gate(bf16_meta, quant_meta):
        try:
            check_comparable(bf16_meta, quant_meta, "bf16.json", "quant.json")
        except SystemExit as exc:
            return str(exc)
        return None

    if gate(meta("bf16"), meta("quant")) is not None:
        raise AssertionError("a baseline-vs-quantized pair stopped passing")
    # The case that shipped: a --direct run standing in for the baseline.
    msg = gate(meta("quant", algos=None), meta("quant", algos="lwc"))
    if msg is None:
        raise AssertionError(
            "a direct-conversion run passed as the bf16 side: the delta would "
            "measure the PTQ algorithm under the name delta_quant_minus_bf16"
        )
    if "bf16.json" not in msg:
        raise AssertionError(f"the refusal does not name the offending side: {msg}")
    msg = gate(meta("bf16"), meta("bf16"))
    if msg is None:
        raise AssertionError("two baselines passed: the delta would be noise")
    if "quant.json" not in msg:
        raise AssertionError(f"the refusal does not name the quant side: {msg}")
    print("[self_check] comparability side tags ok (bf16 vs quant enforced)")


def _check_compare_output_names_object():
    """The comparison output says which object the quant side scored.

    SKILL.md #4.3 requires the report to state whether the object was amct
    direct conversion rather than a PTQ checkpoint, and which decoder layers
    were quantized. Both entries write `direct` and `quant_scope` into their
    result payload, and the comparison prints them because the log line is not
    always at hand when a reviewer reads a delta. Printing a key is not enough
    on its own: `_extract_metadata` reads back only what
    `_META_BOOKKEEPING_KEYS` names, so a key left out of that tuple prints
    `null` for every run -- the same thing "this side carried no metadata"
    prints, and a direct-conversion delta becomes indistinguishable from a PTQ
    one. Hence the end-to-end control: two pairs through the CLI, where the two
    quant sides differ in `direct`. A run that printed `null` both times cannot
    pass it.
    """
    for key in ("direct", "quant_scope"):
        if key not in META_BOOKKEEPING_KEYS:
            raise AssertionError(
                f"`{key}` is not in compare_results._META_BOOKKEEPING_KEYS, so "
                "_extract_metadata never reads it back and the comparison "
                "prints null for every run"
            )

    def payload(tag, direct, quant_scope, score):
        """A minimal result payload of the shape both entries write."""
        quantized = tag == "quant"
        return {
            "results": {"mmlu_pro": {"exact_match,custom-extract": score}},
            "tag": tag,
            "task": "mmlu_pro",
            "limit": None,
            "max_gen_toks": 512,
            "sample_len": 100,
            "limited": None,
            "model_name": "m",
            "direct": direct,
            "quant_scope": quant_scope,
            "algos": "lwc" if quantized and not direct else None,
            "quant_target": "mlp" if quantized else None,
            "bit_config": "w8a8" if quantized else None,
            "quant_dtype": "int8" if quantized else None,
            "is_per_tensor": True if quantized else None,
        }

    def write(path, obj):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f)

    def compare(bf16_path, quant_path):
        proc = subprocess.run(
            [
                sys.executable,
                os.path.join(SKILL_DIR, "compare_results.py"),
                "--bf16",
                bf16_path,
                "--quant",
                quant_path,
            ],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if proc.returncode != 0:
            raise AssertionError(
                f"compare_results.py exited {proc.returncode} on two runs that "
                f"agree on every comparability key: {proc.stderr}"
            )
        return json.loads(proc.stdout)

    scope = {"layers": "0..1", "count": 2, "total": 2, "holes": []}
    with tempfile.TemporaryDirectory() as tmp:
        bf16_path = os.path.join(tmp, "bf16.json")
        write(bf16_path, payload("bf16", False, None, 0.3))
        outs = []
        for name, direct, score in (
            ("direct.json", True, 0.28),
            ("ptq.json", False, 0.29),
        ):
            path = os.path.join(tmp, name)
            write(path, payload("quant", direct, scope, score))
            outs.append(compare(bf16_path, path))

    for got, name, direct in (
        (outs[0], "a --direct run", True),
        (outs[1], "a PTQ run", False),
    ):
        if got.get("direct") is not direct:
            raise AssertionError(
                f"the comparison printed direct={got.get('direct')!r} for "
                f"{name}: the report cannot say which object was scored"
            )
        if got.get("quant_scope") != scope:
            raise AssertionError(
                f"the comparison printed quant_scope={got.get('quant_scope')!r} "
                f"for {name}, not what the run recorded: a partly quantized "
                "checkpoint would read as a full one"
            )
    if outs[0]["direct"] == outs[1]["direct"]:
        raise AssertionError(
            "two quant sides that differ in `direct` printed the same value, so "
            "the control cannot tell a live field from a constant null"
        )
    print(
        "[self_check] compare output names the quant object ok "
        "(direct + quant_scope printed from the payload)"
    )


def _check_comparable_none_rules():
    """The gate skips None only on the object side of the comparability split.

    Both entries write every evaluation-side key, and None is a value there:
    `limit: None` says the run had no sample cap. Exempting None on every key
    let a full run be subtracted from a capped one, and the delta went out with
    `compared_from: "run metadata (both sides)"` -- which reads as checked.
    """

    def meta(**over):
        obj = {key: "shared" for key in COMPARABILITY_KEYS}
        obj.update({key: None for key in OBJECT_SIDE_KEYS}, tag="quant")
        obj.update(over)
        return obj

    def gate(bf16_meta, quant_meta):
        try:
            check_comparable(bf16_meta, quant_meta, "bf16.json", "quant.json")
        except SystemExit as exc:
            return str(exc)
        return None

    bf16 = meta(tag="bf16")
    if gate(bf16, meta()) is not None:
        raise AssertionError(
            "a bf16 baseline's object-side None stopped being exempt: every "
            "bf16-vs-quant comparison would now fail"
        )
    if gate(bf16, meta(bit_config="w4a8")) is not None:
        raise AssertionError(
            "an object-side key carried by one side only failed the gate"
        )
    # Last cell of the grid: an evaluation-side key that is None on BOTH sides
    # is the value "no cap" / no pin, not an absence to exempt and not a
    # mismatch. Two full runs must stay comparable, or the gate sends somebody
    # to re-run a benchmark that already ran in full.
    both_none = ("limit", "max_gen_toks")
    if (
        gate(
            meta(tag="bf16", **{k: None for k in both_none}),
            meta(**{k: None for k in both_none}),
        )
        is not None
    ):
        raise AssertionError(
            "two full runs stopped being comparable: an evaluation-side None "
            "on both sides read as a mismatch, so every pair without --limit "
            "would be refused"
        )
    # `release_version` is written only by the LiveCodeBench sidecar: an
    # lm-eval-vs-lm-eval pair carries no such key on either side, and a key
    # absent from both sides is not comparable at all, so it is skipped.
    absent_left, absent_right = meta(tag="bf16"), meta()
    absent_left.pop("release_version")
    absent_right.pop("release_version")
    if gate(absent_left, absent_right) is not None:
        raise AssertionError(
            "a key absent from both sides read as a mismatch: lm-eval pairs "
            "carry no release_version at all"
        )
    for left, right, key in (
        (bf16, meta(limit=5), "limit"),
        (
            meta(tag="bf16", quant_target="mlp"),
            meta(quant_target="attn-linear"),
            "quant_target",
        ),
    ):
        msg = gate(left, right)
        if msg is None:
            raise AssertionError(
                f"{key} differed across the two sides and the gate passed: the "
                "delta would not describe quantization"
            )
        if f"{key}:" not in msg:
            raise AssertionError(f"the gate failed without naming {key}: {msg}")
    print(
        "[self_check] comparability None rules ok (object-side None exempt, "
        "eval-side None a value, both-None and both-absent pass)"
    )


def _check_constraint_index_wiring():
    """constraints.md's 触发条件索引 routes to every constraint, one row each.

    按需读 is only a safe promise if the index is complete: it is the reader's
    only router into that file, so a constraint with no row is one no symptom
    ever sends anybody to. Structural, and deliberately number-free -- the docs
    state no count, so there is nothing here to drift out of sync with a
    renumbering.
    """
    ref_dir = os.path.join(os.path.dirname(SKILL_DIR), "references")
    with open(os.path.join(ref_dir, "constraints.md"), encoding="utf-8") as f:
        con_lines = f.read().splitlines()
    with open(SKILL_MD, encoding="utf-8") as f:
        skill_lines = f.read().splitlines()
    total = sum(1 for line in con_lines if re.match(r"^##\s+\d+\.", line))
    if not total:
        raise AssertionError("constraints.md declares no `## N.` constraint")
    index_rows = _trigger_index_rows(con_lines)
    if index_rows != total:
        raise AssertionError(
            f"constraints.md's 触发条件索引 has {index_rows} rows for {total} "
            "constraints -- one row each, or a rowless constraint is one 按需读 "
            "never routes to"
        )
    # The 成本档 column is how SKILL.md's benchmark table gets navigated (which
    # task is cheap enough to smoke with); a cell outside 低 / 高 drops that row
    # out of the routing without any error.
    tiers = _cost_tier_rows(skill_lines)
    stray = sorted(set(tiers) - {"低", "高"})
    if stray:
        raise AssertionError(
            f"SKILL.md's benchmark table has 成本档 cells outside 低 / 高: "
            f"{stray} -- the tier column is how a task gets picked"
        )
    if not {"低", "高"} <= set(tiers):
        raise AssertionError(
            f"SKILL.md's benchmark table lost a 成本档 tier: {sorted(tiers)}"
        )
    # Control: one fewer index row must break the equality, else it compares
    # nothing.
    start = next(
        i for i, ln in enumerate(con_lines) if ln.startswith("## 触发条件索引")
    )
    nxt = [
        i for i in range(start + 1, len(con_lines)) if con_lines[i].startswith("## ")
    ]
    end = nxt[0] if nxt else len(con_lines)
    body = [
        i
        for i in range(start, end)
        if con_lines[i].lstrip().startswith("|")
        and not _TABLE_SEPARATOR_RE.match(con_lines[i].strip())
    ]
    perturbed = list(con_lines)
    del perturbed[body[-1]]
    if _trigger_index_rows(perturbed) != total - 1:
        raise AssertionError("control could not drop an index row")
    print(
        f"[self_check] constraint index wiring ok "
        f"({total} constraints, one index row each)"
    )


# ---------------------------------------------------------------------------
# Doc / CLI surface guards. These four protect the parts of the skill an agent
# actually reads at run time -- the command templates, the log-line table, the
# flag list and the cross-pointers -- where a drift is invisible to every
# functional test below (no test runs lm-eval, so no test reads SKILL.md).
# ---------------------------------------------------------------------------

# The two entry points an agent runs to produce a score. The log-table guard
# stops at them on purpose: check_env.py / prefetch_data.py / compare_results.py
# belong to other workflow steps, and their tags (`[env]`, `[prefetch]`) are not
# part of SKILL.md §4.3's runtime log table -- pulling them in would report rows
# the table was never meant to carry.
_ENTRY_SCRIPTS = ("assemble_and_eval.py", "livecodebench_eval.py")

# Flags the docs deliberately do not name. SKILL.md §4.2 sends the
# reader to `--help` for the full list, so a debug-only switch no workflow step
# needs stays out of the prose. Every entry must be absent from the docs *and*
# declared by a CLI here: a documented flag or a stale name listed here is a
# dead allowance and fails.
_UNDOCUMENTED_FLAGS = {
    "--log_samples": "optional per-sample lm-eval debug dump, no workflow step needs it",
}

_ARG_FLAG_RE = re.compile(r"""add_argument\(\s*['"](--[a-z0-9_\-]+)['"]""")
_BLOCK_FLAG_RE = re.compile(r"(?<![\w`-])(--[a-z][a-z0-9_\-]*)")
# A tag is `[name]` at the very start of a printed string: that is the shape
# §4.3's table documents (`[run] ...`, `[ptq] ...`), and anchoring it keeps list
# indexing (rows[0]) and a mid-message bracket out of the set.
_TAG_RE = re.compile(r"^\[([A-Za-z][A-Za-z0-9\-]*)\]")
_LOG_TAG_ROW_RE = re.compile(r"^\|\s*`\[([A-Za-z0-9\-]+)\]`\s*\|")


def _script_flags(sources):
    """`{fname: {--flag}}` for every script here; `--help` is argparse's own."""
    return {
        fname: {"--help"} | set(_ARG_FLAG_RE.findall(text))
        for fname, text in sources.items()
    }


def _fenced_blocks(text):
    """`[(opening_fence_lineno, block_text)]` for every fenced block in a doc."""
    out, buf, start = [], [], None
    for lineno, line in enumerate(text.splitlines(), 1):
        if line.lstrip().startswith("```"):
            if start is None:
                start, buf = lineno, []
            else:
                out.append((start, "\n".join(buf)))
                start = None
            continue
        if start is not None:
            buf.append(line)
    return out


def _block_scripts(block, names):
    """Scripts a command block invokes -- the union, when it invokes several."""
    return sorted(name for name in names if name in block)


def _flag_scope_offenders(blocks, sources):
    """`[(label, lineno, block, script, flag)]` for out-of-scope flag uses.

    Blocks invoking no script of this skill (pip, the official lcb_runner) are
    skipped: their flags are not ours to declare. Blocks invoking several are
    checked against the union, so §4.1's check_env + prefetch block keeps its
    `--task` even though only prefetch_data.py declares it.
    """
    flags = _script_flags(sources)
    offenders = []
    for label, lineno, block in blocks:
        scoped = _block_scripts(block, flags)
        if not scoped:
            continue
        allowed = set(_EXTERNAL_FLAGS)
        for name in scoped:
            allowed |= flags[name]
        for flag in sorted(set(_BLOCK_FLAG_RE.findall(block))):
            if flag in allowed:
                continue
            owners = sorted(n for n, f in flags.items() if flag in f)
            offenders.append(
                (
                    label,
                    lineno,
                    "/".join(scoped),
                    flag,
                    owners[0] if owners else "no script here",
                )
            )
    return offenders


def _all_fenced_blocks(docs):
    return [
        (label, lineno, block)
        for label, text in docs
        for lineno, block in _fenced_blocks(text)
    ]


def _check_flag_scope_per_entry():
    """A command block may only pass a flag that the script it invokes declares.

    `_check_doc_code_drift` rule 3 asks whether *some* script here declares a
    flag the docs name -- one global pool. Both entry points declare `--model`,
    `--limit` and `--algos`, so a template that grows `--dataset_task` (only
    livecodebench_eval.py has it) after someone copies the LiveCodeBench line
    into the lm-eval block stays green there while the command dies at runtime
    with "unrecognized arguments". Scope the pool to the scripts each block
    actually invokes.
    """
    sources, docs = _doc_sources()
    blocks = _all_fenced_blocks(docs)
    offenders = _flag_scope_offenders(blocks, sources)
    if offenders:
        raise AssertionError(
            "command blocks pass a flag to a script that does not declare it: "
            + "; ".join(
                f"{label}:{lineno} gives {flag} to {scripts} (only {owner} declares it)"
                for label, lineno, scripts, flag, owner in offenders
            )
        )
    scoped = sum(1 for _, _, b in blocks if _block_scripts(b, sources))
    if scoped < 4:
        raise AssertionError(
            f"only {scoped} fenced blocks invoke a script of this skill: the "
            "block scanner stopped matching, so the scope rule went unchecked"
        )
    flags = _script_flags(sources)
    foreign = sorted(
        flags["livecodebench_eval.py"] - flags["assemble_and_eval.py"] - {"--help"}
    )
    if not foreign:
        raise AssertionError(
            "the two entry points declare identical flags: the control below "
            "can no longer build an out-of-scope flag"
        )
    control = _flag_scope_offenders(
        list(blocks)
        + [
            (
                "references/_control.md",
                1,
                f"python <skill>/scripts/assemble_and_eval.py {foreign[0]} x",
            )
        ],
        sources,
    )
    if not control:
        raise AssertionError(
            f"a flag only livecodebench_eval.py declares ({foreign[0]}) passed "
            "in an assemble_and_eval.py block: the scope rule verifies nothing"
        )
    print(
        f"[self_check] per-entry flag scope ok ({scoped} blocks, {len(flags)} scripts)"
    )


def _printed_log_tags(sources, names):
    """`{tag: [script]}` for every `[tag]` the named scripts print.

    Only string constants passed to a `print()` call count, and only a tag at
    the very start of the string (see `_TAG_RE`).
    """
    tags = {}
    for fname in names:
        for node in ast.walk(ast.parse(sources[fname])):
            if not (
                isinstance(node, ast.Call) and getattr(node.func, "id", None) == "print"
            ):
                continue
            for child in ast.walk(node):
                if isinstance(child, ast.Constant) and isinstance(child.value, str):
                    match = _TAG_RE.match(child.value)
                    if match:
                        tags.setdefault(match.group(1), set()).add(fname)
    return {tag: sorted(scripts) for tag, scripts in tags.items()}


def _documented_log_tags(text):
    """Tag column of SKILL.md §4.3's log-line table."""
    return {
        m.group(1)
        for m in (_LOG_TAG_ROW_RE.match(line) for line in text.splitlines())
        if m
    }


def _log_tag_diff(printed, table):
    """`(undocumented, stale)` between printed tags and the table's tag column."""
    return sorted(set(printed) - set(table)), sorted(set(table) - set(printed))


def _check_log_tag_table():
    """Every `[tag]` the entry points print has a row in §4.3's log-line table.

    The table is how an agent reads a run log: `[run]` to check the two sides
    are comparable, `[ptq]` for loaded/missing, `[warn]` for what must go into
    the report. A printed tag with no row is a log line nobody explains --
    `[smoke]` and the four LiveCodeBench generation tags shipped that way, and
    the reader had to guess whether silence or a missing row was the problem.
    The reverse is checked too: a row no script prints sends the reader hunting
    a line that never appears.
    """
    sources, docs = _doc_sources()
    printed = _printed_log_tags(sources, _ENTRY_SCRIPTS)
    if len(printed) < 5:
        raise AssertionError(
            f"only {len(printed)} log tags parsed out of {_ENTRY_SCRIPTS}: the "
            "print scanner stopped matching, so the table went unverified"
        )
    skill_text = dict(docs)["SKILL.md"]
    table = _documented_log_tags(skill_text)
    if not table:
        raise AssertionError(
            "no log-tag rows parsed out of SKILL.md: §4.3's table changed shape "
            "and the comparison went vacuous"
        )
    missing, stale = _log_tag_diff(printed, table)
    if missing or stale:
        raise AssertionError(
            "SKILL.md §4.3's log-line table drifted from what the entry points "
            f"print -- undocumented rows to add: {missing}"
            + (f"; table rows no script prints: {stale}" if stale else "")
        )
    injected = f"ctl{9}"  # built at runtime: a literal would sit in the table scan
    ctl_missing, _ = _log_tag_diff(
        dict(printed, **{injected: ["assemble_and_eval.py"]}), table
    )
    if ctl_missing != [injected]:
        raise AssertionError(
            "an injected log tag passed: the table is not really compared"
        )
    _, ctl_stale = _log_tag_diff(
        {t: v for t, v in printed.items() if t != sorted(printed)[0]}, table
    )
    if not ctl_stale:
        raise AssertionError(
            "dropping a printed tag passed: stale table rows are not caught"
        )
    print(f"[self_check] log tag table ok ({len(table)} tags, both directions)")


def _check_cli_flags_documented():
    """Every flag this skill's CLIs declare is named in the docs.

    SKILL.md §4.2 says the full list and the defaults live in `--help`, so the
    docs name only the flags a workflow step needs. That is a deliberate line,
    and nothing held it: `--smoke` shipped as a declared flag with no doc
    mention at all, so the one switch that turns a multi-minute assembly into a
    cheap link check was invisible to an agent reading the skill, and the
    `--limit 10` smoke stayed the only offered option. Either the docs name a
    flag or `_UNDOCUMENTED_FLAGS` says why it may stay hidden.

    Which scripts are CLIs is derived from their AST, not hand-listed: the
    hand-listed `CLI_SCRIPTS` had no guard, so a new entry point stayed out of
    it and every flag it declared went unchecked.
    """
    derived = set(_scripts_with_argparse()) - set(_NON_CLI_SCRIPTS)
    listed = set(CLI_SCRIPTS)
    if derived != listed:
        raise AssertionError(
            f"CLI_SCRIPTS is out of sync with the scripts that take flags: "
            f"unlisted {sorted(derived - listed)}, listed without an "
            f"ArgumentParser {sorted(listed - derived)}. An unlisted CLI keeps "
            f"flags no doc has to mention -- --skip_lm_eval shipped that way."
        )
    sources, docs = _doc_sources()
    flags = _script_flags(sources)
    declared = {"--help"}
    for name in CLI_SCRIPTS:
        declared |= flags[name]
    declared.discard("--help")
    text = "\n".join(body for _, body in docs)
    mentioned = set(re.findall(r"(--[a-z][a-z0-9_\-]*)", text))
    hidden = sorted(declared - mentioned)
    offenders = sorted(set(hidden) - set(_UNDOCUMENTED_FLAGS))
    if offenders:
        raise AssertionError(
            "CLI flags no doc names and no allowance covers: "
            f"{offenders} -- document them or add them to _UNDOCUMENTED_FLAGS "
            "with the reason they may stay hidden"
        )
    for flag, why in sorted(_UNDOCUMENTED_FLAGS.items()):
        if flag in mentioned:
            raise AssertionError(
                f"_UNDOCUMENTED_FLAGS allows {flag} ({why}), but the docs now "
                "name it; drop the allowance so it cannot hide a future gap"
            )
        if flag not in declared:
            raise AssertionError(
                f"_UNDOCUMENTED_FLAGS allows {flag} ({why}), but no CLI here "
                "declares it; the allowance is stale"
            )
    print(
        f"[self_check] cli flags documented ok "
        f"({len(declared) - len(hidden)} named, {len(hidden)} allowed hidden)"
    )


def _argparse_default(source, flag):
    """The `default=` this script's argparse gives `flag`, as source text."""
    for node in ast.walk(ast.parse(source)):
        if not (
            isinstance(node, ast.Call)
            and getattr(node.func, "attr", None) == "add_argument"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value == flag
        ):
            continue
        for kw in node.keywords:
            if kw.arg == "default":
                return ast.unparse(kw.value)
        return "<omitted>"
    return "<not declared>"


def _check_limit_sentinel():
    """Both entries spell "no sample cap" the same way and refuse one that caps
    nothing, and the reader reads every row on the sentinel.

    `limit` is a comparability key (compare_results.py's `_EVAL_SIDE_KEYS`) and
    each entry writes it into its own payload, so an unlimited run has to read
    the same whichever entry produced it. livecodebench_eval.py declared
    `default=0` while assemble_and_eval.py declared `None`: a full
    LiveCodeBench run wrote `limit: 0`, every consumer had to know that 0 means
    "everything" (`if a.limit and a.limit > 0`), and the two payloads disagreed
    on the one field that says how many problems were scored. `load_lcb_docs`
    is exercised here too -- it has no other runtime coverage, and it is the
    function that has to read None as "every row".
    """
    sources, _ = _doc_sources()
    probe = 'ap.add_argument("--limit", type=int, default=0)\n'
    if (
        _argparse_default(probe, "--limit") != "0"
        or _argparse_default(probe, "--nope") != "<not declared>"
    ):
        raise AssertionError(
            "the argparse default parser stopped reading: the --limit sentinel "
            "comparison went vacuous"
        )
    defaults = {n: _argparse_default(sources[n], "--limit") for n in _ENTRY_SCRIPTS}
    diverged = {n: v for n, v in defaults.items() if v != "None"}
    if diverged:
        raise AssertionError(
            f"--limit defaults diverge from None (= no cap): {diverged} -- an "
            "unlimited run writes a different `limit` into each entry's payload"
        )
    # A non-positive cap scores everything on both entries while recording the
    # cap, and compare_results.py gates on the record -- so it is refused here
    # instead, and both entries must go through the two shared helpers.
    if resolve_sample_limit(None) is not None or resolve_sample_limit(10) != 10:
        raise AssertionError("resolve_sample_limit stopped passing a real cap through")
    for bad in (0, -1):
        try:
            resolve_sample_limit(bad)
        except SystemExit as exc:
            if "--limit" not in str(exc):
                raise AssertionError(
                    f"the --limit {bad} refusal does not name the flag: {exc}"
                ) from None
        else:
            raise AssertionError(
                f"--limit {bad} was accepted: it scores the full benchmark "
                "while recording a cap, so the run then fails the "
                "comparability gate against one that omitted --limit"
            )
    if sample_limit_fields(None) != {"limited": False, "limit": None} or (
        sample_limit_fields(10) != {"limited": True, "limit": 10}
    ):
        raise AssertionError("sample_limit_fields stopped describing the cap")
    for name in _ENTRY_SCRIPTS:
        for helper in ("resolve_sample_limit(", "sample_limit_fields("):
            if helper not in sources[name]:
                raise AssertionError(
                    f"{name} does not call {helper[:-1]}: a non-positive "
                    "--limit would be accepted there, or `limited` would be "
                    "spelled locally again"
                )
        if '"limited":' in sources[name]:
            raise AssertionError(
                f'{name} writes the "limited" payload key by hand instead of '
                "through sample_limit_fields"
            )
    from livecodebench_eval import load_lcb_docs  # imports torch: kept lazy

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "release.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for i in range(3):
                f.write(
                    json.dumps({"question_id": f"q{i}", "question_content": "x"}) + "\n"
                )
        if len(load_lcb_docs([path], None)) != 3:
            raise AssertionError(
                "load_lcb_docs(files, None) did not read every row: the "
                "unlimited sentinel is not handled"
            )
        if len(load_lcb_docs([path], 1)) != 1:
            raise AssertionError("load_lcb_docs(files, 1) did not cap the frame")
    print(
        "[self_check] --limit sentinel ok (both entries default None, refuse "
        "a non-positive cap, share `limited`; None reads every row, 1 caps)"
    )


# ---------------------------------------------------------------------------
# Check registry. main() drives the run from these two lists, so the run order,
# `--list-checks` and the "every _check_* is wired exactly once" audit cannot
# disagree. SKILL.md points here instead of hand-copying a list of checks.
# ---------------------------------------------------------------------------
DEFAULT_CHECKS = [
    _check_registry,
    _check_split_layout_semantics,
    _check_partial_split_deferred,
    _check_remote_glob_semantics,
    _check_split_list_arg,
    _check_jsonl_reader,
    _check_lcb_validation,
    _check_quant_target_guard,
    _check_attn_target_from_path,
    _check_input_device_fallback,
    _check_plan_device_preflight,
    _check_doc_code_drift,
    _check_doc_drift_controls,
    _check_citation_targets,
    _check_reference_mentions,
    _check_constraint_index_wiring,
    _check_shared_param_table,
    _check_metadata_covers_comparability_keys,
    _check_comparable_none_rules,
    _check_comparable_side_tags,
    _check_compare_output_names_object,
    _check_flag_scope_per_entry,
    _check_log_tag_table,
    _check_cli_flags_documented,
    _check_limit_sentinel,
    _check_skill_metric_keys,
    _check_task_offline_registration,
    _check_cli_help,
    _check_cli_flags,
    _check_checklist_wiring,
]
REMOTE_CHECKS = [_check_remote_registry]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--remote",
        action="store_true",
        help="also validate datasets.json against each HF repo's file list (network)",
    )
    ap.add_argument(
        "--list-checks",
        action="store_true",
        help="print every check (name + summary + whether a green run shows a line) and exit",
    )
    args = ap.parse_args()
    if args.list_checks:
        _list_checks()
        return
    for check in DEFAULT_CHECKS:
        check()
    if args.remote:
        for check in REMOTE_CHECKS:
            check()
    print("[self_check] ok")


if __name__ == "__main__":
    main()
