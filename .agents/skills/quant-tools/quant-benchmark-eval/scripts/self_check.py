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

`--remote` additionally cross-checks datasets.json against each dataset repo's
real file list (metadata only, no bulk download); it needs network and is
therefore opt-in.
"""

import argparse
import ast
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
    AMCT_HOME,
    BENCH_ASSETS,
    DEFAULT_BIT_CONFIG,
    UnresolvedSplit,
    local_dataset_map,
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
# amct checkout: kwargs / config keys of third-party libraries. Deliberately a
# text block parsed at import, not a dict literal -- `_check_doc_code_drift`
# searches the scripts' own text, so a literal here would put every allowed name
# into the blob it searches, the lookup would "find" it, and the check would
# quietly stop verifying anything (the same trap the injected controls dodge by
# building stale names at runtime). Names that live in amct need no allowance:
# they are verified against $AMCT_HOME below, so an amct rename is caught.
_EXTERNAL_IDENTIFIERS_TEXT = """\
attention_mask|transformers forward/generate kwarg (constraints.md #7)
cache_position|transformers attention forward kwarg (constraints.md #7)
past_key_values|transformers KV-cache container (constraints.md #7)
max_memory|transformers device_map kwarg (constraints.md #11)
max_split_size_mb|allocator config key, not a Python symbol (constraints.md #11)
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
    2. every `scripts/<file>.py` the docs mention exists, and every script that
       exists is mentioned (an undocumented script is a doc gap too);
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

    # 2. script filenames, both directions
    mentioned = set(re.findall(r"scripts/([A-Za-z_][\w]*\.py)", text))
    for fname in sorted(mentioned - set(sources)):
        raise AssertionError(f"docs reference scripts/{fname}, which does not exist")
    for fname in sorted(set(sources) - mentioned):
        raise AssertionError(
            f"scripts/{fname} exists but the docs never mention it; the file "
            "table in SKILL.md must cover every script"
        )

    # 3. CLI flags
    declared = set(re.findall(r"""add_argument\(\s*['"](--[a-z0-9_\-]+)['"]""", blob))
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


def _check_shim_key_collisions():
    """Two tasks sharing one `(repo_id, dataset_name)` must not silently win.

    The offline shim keys `_LOCAL` by the pair lm-eval passes to
    `datasets.load_dataset(path=..., name=...)`, so it is per *dataset config*,
    not per task. Sharing a repo is fine and already happens (arc_challenge /
    arc_easy on allenai/ai2_arc, livecodebench / livecodebench_release_v6 on
    livecodebench/code_generation_lite) because `dataset_name` tells them apart.
    Sharing the whole key while declaring different local data is a registry
    bug that would otherwise be silent: one task's files get served for the
    other, and the run scores data that is not the one datasets.json claims
    (constraints.md #2). Inject registries -- no filesystem, no network, no NPU.
    """
    saved = config_module.load_dataset_registry

    def _entry(subdir, name, rel="test-*.parquet"):
        return {
            "repo_id": "selfcheck/collide",
            "subdir": subdir,
            "dataset_name": name,
            "splits": {"test": rel},
            "allow_patterns": ["*.parquet"],
        }

    try:
        # 1. Same repo, different dataset_name -> two distinct keys, both kept.
        config_module.load_dataset_registry = lambda: {
            "task_a": _entry("a", "cfg-a"),
            "task_b": _entry("b", "cfg-b"),
        }
        got = config_module.local_dataset_map()
        if len(got) != 2:
            raise AssertionError(
                "two tasks on one repo with different dataset_name must produce "
                f"two shim keys, got {sorted(got)}"
            )
        # 2. Same key, different local data -> must raise, not overwrite.
        config_module.load_dataset_registry = lambda: {
            "task_a": _entry("a", "same"),
            "task_b": _entry("b", "same"),
        }
        try:
            config_module.local_dataset_map()
        except ValueError as exc:
            msg = str(exc)
            for needle in ("task_a", "task_b", "('selfcheck/collide', 'same')"):
                if needle not in msg:
                    raise AssertionError(
                        f"collision error must name both tasks and the key, "
                        f"missing {needle!r}: {exc}"
                    )
        else:
            raise AssertionError(
                "two tasks declaring different data for one (repo_id, "
                "dataset_name) were silently merged into one shim key"
            )
        # 3. Same key, identical declaration -> harmless duplicate, still works.
        config_module.load_dataset_registry = lambda: {
            "task_a": _entry("same_dir", "same"),
            "task_b": _entry("same_dir", "same"),
        }
        got = config_module.local_dataset_map()
        if len(got) != 1:
            raise AssertionError(
                "an identical duplicate declaration must collapse to one key, "
                f"got {sorted(got)}"
            )
    finally:
        config_module.load_dataset_registry = saved

    # 4. Same as (3), but the cache is *partial*: both duplicates then hold a
    #    deferred UnresolvedSplit. Their messages name different tasks (that is
    #    what makes the deferred error actionable), so comparing messages called
    #    this a shim-key collision -- a misleading "declare different local data
    #    ('same_dir' vs 'same_dir')" raised while the map is built, i.e. one
    #    task's incomplete cache blocking *every* task again, which is exactly
    #    what the deferral exists to prevent. Equality must key on the
    #    declaration, not the message.
    saved_assets = config_module.BENCH_ASSETS
    with tempfile.TemporaryDirectory() as tmp:
        os.makedirs(os.path.join(tmp, "same_dir"))
        for name in ("test-00000-of-00002.parquet",):
            open(os.path.join(tmp, "same_dir", name), "w", encoding="utf-8").close()
        partial = {
            "all_of": ["test-00000-of-00002.parquet", "test-00001-of-00002.parquet"]
        }
        config_module.load_dataset_registry = lambda: {
            "task_a": _entry("same_dir", "same", rel=partial),
            "task_b": _entry("same_dir", "same", rel=partial),
        }
        config_module.BENCH_ASSETS = tmp
        try:
            try:
                got = config_module.local_dataset_map()
            except ValueError as exc:
                raise AssertionError(
                    "an identical duplicate declaration was reported as a "
                    "shim-key collision just because both copies defer the same "
                    f"partial-cache error: {exc}"
                ) from exc
            if len(got) != 1:
                raise AssertionError(
                    "an identical duplicate declaration with a partial cache "
                    f"must still collapse to one key, got {sorted(got)}"
                )
            held = got[("selfcheck/collide", "same")]["test"]
            if not isinstance(held, UnresolvedSplit):
                raise AssertionError(
                    f"the folded duplicate lost its deferred error: {held!r}"
                )
            # Folding must not swallow the failure: reading the split still raises.
            try:
                held.raise_it()
            except FileNotFoundError:
                pass
            else:
                raise AssertionError(
                    "the folded duplicate's deferred error no longer raises"
                )
        finally:
            config_module.load_dataset_registry = saved
            config_module.BENCH_ASSETS = saved_assets
    # The shipped registry must be collision-free.
    if len(local_dataset_map()) != len(
        {
            (s["repo_id"], s.get("dataset_name"))
            for s in load_dataset_registry().values()
        }
    ):
        raise AssertionError("datasets.json declares duplicate shim keys")
    print(
        "[self_check] shim key collisions rejected (same repo ok, same key not; "
        "identical duplicate folds even with a partial cache)"
    )


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
    may read the same repo from different subdirs (datasets.json already does
    that for livecodebench / livecodebench_release_v6, and both only escape this
    branch because they have hand-written stubs). Deriving from the first entry
    alone leaves the second task's shards out of the listing, so the offline
    regression reports a *stub* gap as a registry false positive. Union them.

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
    registry bug. datasets.json already has this shape (livecodebench /
    livecodebench_release_v6), it just hides behind hand-written stubs.
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
    # Two tasks sharing one unstubbed repo_id: the derived listing must cover
    # both, otherwise the second task's shards look like a registry error.
    probe = dict(reg)
    probe["gsm8k_probe"] = _probe_entry(["*.parquet", "README.md"])
    probe["gsm8k_ssec"] = _probe_entry_ssec(["*.parquet", "README.md"])
    probe_lists, probe_derived = _upstream_stubs(probe)
    if "openai/gsm8k" not in probe_derived:
        raise AssertionError(
            "a repo without a hand-written stub must fall back to a derived "
            f"listing, got derived={probe_derived}"
        )
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
    # Not vacuous: drop the second task's shard from the shared listing and the
    # check must name that task, proving both are really being matched.
    broken = {k: list(v) for k, v in probe_lists.items()}
    broken["openai/gsm8k"] = [f for f in shared if not f.startswith("ssec/test2-")]
    problems = _run_remote_check(broken, reg=probe)
    if "gsm8k_ssec" not in problems:
        raise AssertionError(
            "--remote must report the second task when one of its all_of shards "
            f"is missing from the shared repo listing, got:\n{problems or '<no problems>'}"
        )
    print(
        "[self_check] remote glob semantics ok (any_of one-hit / all_of every "
        "shard / new task needs no stub / two tasks may share one repo)"
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


# Data-resolution / registry pre-flight calls that must run *before* any weight
# is put on a card, and the calls that put weights on a card. A pre-flight that
# drifts below the model load still "works" in every functional test -- it only
# stops being a pre-flight, and the failure it was written to catch returns to
# costing minutes of assembly plus a traceback that does not name its cause
# (constraints.md #2).
_PREFLIGHT_CALLS = {
    "livecodebench_eval.py": ("resolve_dataset_files",),
    "assemble_and_eval.py": ("check_task_offline_ready",),
}
_PREFLIGHT_IMPORTS = {"assemble_and_eval.py": ("bench_offline",)}
_MODEL_LOAD_CALLS = ("load_float", "assemble_quant_model")


def _check_data_preflight_order():
    """Registry / dataset pre-flight must precede the model load in each CLI.

    Two distinct fail-late traps, both invisible to a functional test:

    - `livecodebench_eval.py` resolves its LiveCodeBench cache through
      datasets.json. Done after assembly, an unprefetched (or partially
      prefetched `all_of`) cache reports itself only once the float model is on
      the card and every quant block is swapped in -- and the traceback never
      mentions `--dataset_task`.
    - `assemble_and_eval.py` builds the offline shim map by importing
      `bench_offline` (`_LOCAL = local_dataset_map()`). That import is where a
      contradictory datasets.json -- two tasks on one `(repo_id, dataset_name)`
      declaring different local data -- raises, so importing it next to
      `install()` (after the model load) turns a registry bug into a
      post-assembly failure.

    Static check on the parsed `main()` bodies: no NPU, no network, no model.
    """
    for fname, calls in sorted(_PREFLIGHT_CALLS.items()):
        with open(os.path.join(SKILL_DIR, fname), encoding="utf-8") as f:
            tree = ast.parse(f.read(), filename=fname)
        main_fn = next(
            (
                node
                for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.name == "main"
            ),
            None,
        )
        if main_fn is None:
            raise AssertionError(f"{fname} has no main() to check pre-flight order in")
        found = {}
        loads = []
        for node in ast.walk(main_fn):
            if isinstance(node, ast.Call):
                func = node.func
                name = (
                    func.id if isinstance(func, ast.Name) else getattr(func, "attr", "")
                )
                if name in calls:
                    # Earliest occurrence: `ast.walk` is breadth-first, so a
                    # pre-flight nested inside `if not a.smoke:` is visited
                    # *after* a shallower node, and `bench_offline` is imported
                    # twice on purpose (early to build the shim map, later next
                    # to install()). What matters is the first one.
                    found[name] = min(found.get(name, node.lineno), node.lineno)
                elif name in _MODEL_LOAD_CALLS:
                    loads.append((node.lineno, name))
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in _PREFLIGHT_IMPORTS.get(fname, ()):
                        found[alias.name] = min(
                            found.get(alias.name, node.lineno), node.lineno
                        )
        if not loads:
            raise AssertionError(
                f"{fname}'s main() calls neither {' nor '.join(_MODEL_LOAD_CALLS)}: "
                "this pre-flight-order check no longer tests anything"
            )
        first_load = min(loads)
        for wanted in tuple(calls) + _PREFLIGHT_IMPORTS.get(fname, ()):
            if wanted not in found:
                raise AssertionError(
                    f"{fname}'s main() never runs the {wanted} pre-flight, so the "
                    f"registry / dataset problem it catches would surface only "
                    f"after {first_load[1]}() (line {first_load[0]})"
                )
            if found[wanted] > first_load[0]:
                raise AssertionError(
                    f"{fname}: the {wanted} pre-flight sits at line "
                    f"{found[wanted]}, *after* {first_load[1]}() at line "
                    f"{first_load[0]} -- it is no longer a pre-flight, so the "
                    f"error it exists to raise early would arrive only after the "
                    f"model was loaded and assembled (constraints.md #2)"
                )
    print("[self_check] dataset / registry pre-flight runs before the model load")


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


def _check_offline_shim_install():
    """`bench_offline.install()` must intercept the call lm-eval really makes.

    lm-eval 0.4.12 resolves the loader through the module attribute at call time
    (`api/task.py`: `datasets.load_dataset(path=..., name=...)`), so the patch
    that matters is the one on `datasets.load_dataset`. A layout that instead
    binds `load_dataset` into `lm_eval.api.task` at import time keeps its own
    reference and would go straight to the network / HF cache, scoring data that
    is not the pinned `$BENCH_ASSETS` copy (constraints.md #2); `install()`
    patches that name too when it exists. Assert both, then drive one call the
    way lm-eval does and check it lands on a local file -- no network, no NPU.
    """
    try:
        import datasets
    except ImportError as exc:
        print(f"[self_check] skip shim-install check: {exc}")
        return

    import bench_offline

    task_mod = None
    try:
        import lm_eval.api.task  # noqa: F401

        task_mod = sys.modules["lm_eval.api.task"]
    except ImportError as exc:
        print(f"[self_check] shim-install check runs without lm-eval: {exc}")

    saved_datasets = datasets.load_dataset
    saved_task = getattr(task_mod, "load_dataset", None)
    saved_local = bench_offline._LOCAL
    with tempfile.TemporaryDirectory() as tmp:
        probe = os.path.join(tmp, "sample.jsonl")
        with open(probe, "w", encoding="utf-8") as f:
            f.write('{"question_id":"q1","answer":"42"}\n')
            f.write('{"question_id":"q2","answer":"7"}\n')
        bench_offline._LOCAL = {("selfcheck/probe", None): {"test": [probe]}}
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                bench_offline.install()
            if datasets.load_dataset is not bench_offline._patched:
                raise AssertionError("install() did not patch datasets.load_dataset")
            if task_mod is not None and hasattr(task_mod, "load_dataset"):
                if task_mod.load_dataset is not bench_offline._patched:
                    raise AssertionError(
                        "this lm-eval binds load_dataset into lm_eval.api.task at "
                        "import time and install() left it unpatched, so tasks "
                        "would read from the network / HF cache instead of "
                        "$BENCH_ASSETS"
                    )
            # The keyword call shape lm-eval uses must reach the local file.
            got = datasets.load_dataset(
                path="selfcheck/probe", name=None, data_dir=None
            )
            if list(got) != ["test"]:
                raise AssertionError(f"shim returned splits {list(got)}, want ['test']")
            ids = list(got["test"]["question_id"])
            if ids != ["q1", "q2"]:
                raise AssertionError(f"shim did not serve the local jsonl rows: {ids}")

            # A task whose `dataset_kwargs` declares `split` must get that one
            # Dataset back, not a DatasetDict: `datasets.load_dataset` returns
            # the narrower type, and lm-eval forwards dataset_kwargs verbatim
            # (`api/task.py:869`), so the harness would index a DatasetDict with
            # a split name it already resolved away.
            one = datasets.load_dataset(path="selfcheck/probe", name=None, split="test")
            if isinstance(one, datasets.DatasetDict):
                raise AssertionError(
                    "shim returned a DatasetDict for load_dataset(split='test'); "
                    "datasets.load_dataset returns a single Dataset in that form"
                )
            if list(one["question_id"]) != ["q1", "q2"]:
                raise AssertionError(
                    f"shim(split='test') served rows {list(one['question_id'])}"
                )
            # Same split asked for positionally (index 4 in datasets' signature).
            positional = datasets.load_dataset(
                "selfcheck/probe", None, None, None, "test"
            )
            if isinstance(positional, datasets.DatasetDict):
                raise AssertionError(
                    "shim ignored a positional `split` argument (index 4 in "
                    "datasets.load_dataset's signature) and returned a "
                    "DatasetDict; _bind() must normalize positionals before "
                    "matching against the registry"
                )
            if list(positional["question_id"]) != ["q1", "q2"]:
                raise AssertionError("shim ignored a positional `split` argument")
            # A second declared split that was never prefetched must not break
            # a run that only asked for `test`: mmlu_pro / boolq / copa declare
            # `validation`, and loading every declared split turned that into a
            # failure of an unrelated request (plus I/O nobody scores).
            bench_offline._LOCAL[("selfcheck/probe", None)]["validation"] = [
                os.path.join(tmp, "never-downloaded.parquet")
            ]
            try:
                one = datasets.load_dataset(
                    path="selfcheck/probe", name=None, split="test"
                )
            except FileNotFoundError as exc:
                raise AssertionError(
                    f"shim validated a split the caller did not ask for: {exc}"
                ) from exc
            if list(one["question_id"]) != ["q1", "q2"]:
                raise AssertionError(
                    f"shim(split='test') served {list(one['question_id'])}"
                )
            # Asking for the unprefetched split must still fail, naming it.
            try:
                datasets.load_dataset(
                    path="selfcheck/probe", name=None, split="validation"
                )
            except FileNotFoundError as exc:
                if "validation" not in str(exc):
                    raise AssertionError(
                        f"unprefetched split error does not name the split: {exc}"
                    ) from exc
            else:
                raise AssertionError("shim served a split whose file does not exist")
            del bench_offline._LOCAL[("selfcheck/probe", None)]["validation"]
            # Unknown split must name what the registry does declare.
            try:
                datasets.load_dataset(path="selfcheck/probe", name=None, split="train")
            except KeyError as exc:
                if "no split 'train'" not in str(exc) or "'test'" not in str(exc):
                    raise AssertionError(f"unclear unknown-split error: {exc}")
            else:
                raise AssertionError(
                    "shim accepted split='train' that the registry does not declare"
                )
            # Passthrough of an unregistered (path, name) must stay diagnosable:
            # the real loader's own message (FSTimeoutError on these boxes) says
            # nothing about datasets.json, which is the thing the user has to fix.
            # Stub _ORIG so this needs no network.
            saved_orig = bench_offline._ORIG

            def _boom(*_a, **_kw):
                raise OSError("simulated FSTimeoutError")

            bench_offline._ORIG = _boom
            try:
                datasets.load_dataset(path="not/registered", name=None)
            except RuntimeError as exc:
                msg = str(exc)
                if "datasets.json" not in msg or "not/registered" not in msg:
                    raise AssertionError(
                        "passthrough failure does not name the unregistered path "
                        f"or datasets.json: {exc}"
                    )
                if "selfcheck/probe" not in msg:
                    raise AssertionError(
                        f"passthrough failure does not list registered keys: {exc}"
                    )
                if exc.__cause__ is None:
                    raise AssertionError(
                        "passthrough failure dropped the loader's original error"
                    )
            else:
                raise AssertionError(
                    "shim swallowed an unregistered path instead of failing"
                )
            finally:
                bench_offline._ORIG = saved_orig
        finally:
            datasets.load_dataset = saved_datasets
            if saved_task is not None:
                task_mod.load_dataset = saved_task
            bench_offline._LOCAL = saved_local
    print("[self_check] offline shim intercepts lm-eval's load_dataset call shape")


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


def _check_task_offline_registration():
    """Every lm-eval task this skill advertises must resolve to registered data.

    `check_task_offline_ready()` is the pre-flight that stops a typo'd or
    unregistered `--task` before the model is loaded; if it were to pass for a
    dataset that is not in datasets.json, the run would instead die as
    `FSTimeoutError` minutes into evaluation (constraints.md #2). Verify both
    directions here, against lm-eval's installed task index -- no download, no
    NPU. Skipped when lm-eval is absent.
    """
    try:
        from lm_eval.tasks import TaskManager
    except ImportError as exc:
        print(f"[self_check] skip task-registration check: {exc}")
        return
    task_manager = TaskManager()
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
    _check_shim_key_collisions()
    _check_split_layout_semantics()
    _check_partial_split_deferred()
    _check_remote_glob_semantics()
    _check_split_list_arg()
    _check_jsonl_reader()
    _check_lcb_validation()
    _check_quant_target_guard()
    _check_data_preflight_order()
    _check_input_device_fallback()
    _check_offline_shim_install()
    _check_doc_code_drift()
    _check_doc_drift_controls()
    _check_skill_metric_keys()
    _check_task_offline_registration()
    _check_cli_help()
    _check_cli_flags()
    if args.remote:
        _check_remote_registry()
    print("[self_check] ok")


if __name__ == "__main__":
    main()
