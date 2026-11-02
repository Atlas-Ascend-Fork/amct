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
"""Assemble the user's amct fake-quant model and run a generative benchmark.

What gets quantized is derived from --param_dir, not assumed: a dir holding
layer_*_self_attn/linear_attn params evaluates MLP + attention (the object the
user actually produced), otherwise MLP only. The cost of the attention path is
real -- most amct attention wrappers do not pass `past_key_values` through, so
quantized attn forces generate(use_cache=False) and every token recomputes the
full context (minutes/question instead of ~6 s). That trade-off belongs to the
user, so the script gates it behind --allow_attn_quant and warns when a narrow
target leaves attention params unread. See references/constraints.md #1.

Assembly reuses amct blockwise building blocks (float_model + build_quant_block +
load_selected_layer_ptq_params) but assembles ALL layers into one generatable model
(amct's own eval never does this — it goes block-by-block for PPL).

--direct evaluates amct direct conversion, the object a quantize round with no
algorithm produces: there are no ptq params to read, so every decoder layer gets
a quant block, no algorithm module is built, and each block computes its weight /
activation scales on the fly at forward time (same as amct's own
`eval --eval_mode quant`). --quant_target must be passed explicitly — there is no
directory to derive it from — and --algos must not be passed.

--bf16 skips quant entirely and runs the original model through the SAME harness,
for an apples-to-apples baseline. Always run both and report the delta.
"""

import argparse
import json
import os
import re
import sys
import torch

SKILL_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SKILL_DIR)  # for bench_offline / config
from config import DEFAULT_BIT_CONFIG, require_bit_config  # noqa: E402
from utils import (  # noqa: E402
    NON_LM_EVAL_TASKS,
    TASK_ALIASES,
    check_environment,
    check_task_offline_ready,
    dtype_kwargs,
    ensure_direct_mode_args,
    ensure_single_attention_target,
    ensure_supported_quant_target,
    has_attention_quant,
    infer_device_map,
    resolve_sample_limit,
    sample_limit_fields,
    split_list_arg,
    split_quant_targets,
    warn_trust_remote_code,
)

# amct names per-layer ptq param files `layer_<idx>_<unit>.pt`
# (`amct_pytorch/common/models/llm/common/ptq_params.py`). The `<unit>` part
# names the training target (`amct_pytorch/common/models/llm/common/base.py`:
# `iter_ptq_units`): attention units are `self_attn` / `linear_attn`, the
# dense-MLP unit is `mlp`, MoE experts are `expert_<j>`. For mlp / moe the
# filename is the ground truth of what the user actually quantized -- amct
# treats mlp / moe / attn-linear / attn-cache as FLAT targets (one per PTQ
# round; `cli/llm/args.py:141`), and one param_dir may hold several rounds'
# outputs, so the derivation below reads the directory per class instead of
# assuming any "base" target or forcing an mlp/moe companion onto an
# attention-only checkpoint.
#
# The two attention targets are the exception: `iter_ptq_units` names the unit
# from the layer type alone, so an attn-linear round and an attn-cache round
# both save `layer_<idx>_self_attn.pt` -- the file cannot say which one
# produced it (constraints.md #1). The target survives only in the directory:
# amct's `_resolve_quant_param_dir` auto-creates
# `<output_dir>/ptq_params/<model_name>/<quant_target>` from the target
# literal, and every `examples/models/` script keeps that leaf name.
# Group 1 is the layer index in all three: `_resolve_quant_target` /
# `_param_dir_has_attn` only need a match, `_param_layer_indices` needs the
# index. Both attention targets share one class -- `iter_ptq_units` names the
# unit from the layer type alone, so an attn-linear round and an attn-cache
# round both write `layer_<idx>_self_attn.pt` and only the directory says which
# produced it (constraints.md #1).
_ATTN_PARAM_RE = re.compile(r"^layer_(\d+)_(?:self|linear)_attn\.pt$")
_MLP_PARAM_RE = re.compile(r"^layer_(\d+)_mlp\.pt$")
_MOE_PARAM_RE = re.compile(r"^layer_(\d+)_expert_\d+\.pt$")
_ATTN_TARGETS = ("attn-linear", "attn-cache")
_LAYER_RE_BY_TARGET = {
    "mlp": _MLP_PARAM_RE,
    "moe": _MOE_PARAM_RE,
    "attn-linear": _ATTN_PARAM_RE,
    "attn-cache": _ATTN_PARAM_RE,
}


def _attn_target_from_path(param_dir):
    """Attention target named by a path component of `param_dir`, else "".

    Components are compared exactly, so a sibling such as `attn-cache-exp3`
    does not match, and a path naming both targets is ambiguous and returns ""
    as well. Empty means "the directory does not say": `_resolve_quant_target`
    then keeps `attn-linear`, and `_preflight_param_dir` warns that the target
    was assumed rather than read.
    """
    parts = os.path.normpath(param_dir).split(os.sep)
    hits = [t for t in _ATTN_TARGETS if t in parts]
    return hits[0] if len(hits) == 1 else ""


def _param_dir_has_attn(param_dir):
    """True when `param_dir` holds per-layer attention ptq params.

    The directory is the only thing that says what the user actually quantized:
    a checkpoint trained with attn-linear has `layer_*_self_attn.pt` next to the
    MLP ones. Reading it back is what keeps the evaluated object equal to the
    object the user produced (constraints.md #1).
    """
    if not param_dir or not os.path.isdir(param_dir):
        return False
    try:
        names = os.listdir(param_dir)
    except OSError:
        return False
    return any(_ATTN_PARAM_RE.match(name) for name in names)


def _param_layer_indices(param_dir, quant_target):
    """Decoder-layer indices `--param_dir` holds params for.

    The set is the layer span the user's PTQ round actually covered.
    amct trains `range(start_block_idx, end_block_idx)` and saves per layer
    (`amct_pytorch/workflows/llm_ptq.py`), so a round stopped early -- agent
    timeout, OOM, or split into chunks on purpose -- leaves a directory that
    covers only part of the stack. Assembly used to build quant blocks for ALL
    layers regardless, and amct's strict=False load left the uncovered ones on
    construction defaults: a mixed PTQ/direct object nobody produced, scored as
    if it were the user's checkpoint. Layers outside the set now keep their
    original bf16 module, so the evaluated object is the one that was trained.

    Only the file classes the requested targets read are counted: a directory
    may hold several rounds' outputs side by side, and evaluating `mlp` against
    a dir that also holds an attn round must take its span from the mlp files.
    """
    # --direct never reaches this function: it has no directory to read, so the
    # caller sets the span to every layer itself. On the PTQ path both entry
    # points hard-require --param_dir, and _preflight_param_dir raises on a
    # non-directory before assembly, so the guards below are unreachable in the
    # documented flow. They raise rather than return None: a None used to mean
    # "no directory to read" and made assembly quantize every layer on
    # construction defaults -- an object nobody produced -- while [quant-scope]
    # reported it as direct quant.
    if not param_dir or not os.path.isdir(param_dir):
        raise RuntimeError(
            f"--param_dir {param_dir!r} is missing or is not a directory, so the "
            f"quantized layer span cannot be read. Pass the ptq params directory "
            f"of this run, --direct to score amct direct conversion (no params "
            f"to read), or --bf16 to score the baseline."
        )
    try:
        names = os.listdir(param_dir)
    except OSError as exc:
        raise RuntimeError(
            f"--param_dir {param_dir} cannot be listed ({exc}), so the quantized "
            f"layer span cannot be read."
        ) from exc
    patterns = {
        _LAYER_RE_BY_TARGET[t]
        for t in split_quant_targets(quant_target)
        if t in _LAYER_RE_BY_TARGET
    }
    if not patterns:
        raise RuntimeError(
            f"--quant_target {quant_target!r} names no unit class this skill "
            f"reads layer indices from, so the quantized layer span is "
            f"unknowable and every layer would be quantized on construction "
            f"defaults. amct flat targets: mlp / moe / attn-linear / attn-cache."
        )
    idxs = set()
    for name in names:
        for rx in patterns:
            m = rx.match(name)
            if m:
                idxs.add(int(m.group(1)))
                break
    return idxs


def _describe_quant_scope(quant_layers, num_layers, source):
    """Print `[quant-scope]` and return the span as a payload record.

    `source` names where the span came from: a `--param_dir` in PTQ mode, a
    fixed string in `--direct` mode (nothing on disk says "all layers").

    The span is part of the object's identity: a score for layers 0..10 of a
    60-layer model is not comparable with one for all 60, and nothing else in
    the log says which was evaluated.
    """
    covered = sorted(quant_layers)
    holes = [i for i in range(covered[0], covered[-1] + 1) if i not in quant_layers]
    full = covered == list(range(num_layers))
    print(
        f"[quant-scope] {source} covers layers {covered[0]}..{covered[-1]} "
        f"({len(covered)} of {num_layers} decoder layers)"
        + (f", holes at {holes[:8]}" if holes else "")
        + (
            ": the whole stack is quantized."
            if full
            else f": the other {num_layers - len(covered)} stay bf16, so the score "
            f"describes a partially-quantized checkpoint. State the quantized "
            f"layer span in the report."
        )
    )
    return {
        "layers": f"{covered[0]}..{covered[-1]}",
        "count": len(covered),
        "total": num_layers,
        "holes": holes,
    }


def _config_num_layers(model_path):
    """Decoder-layer count from config.json, or None when it cannot tell.

    Text-only and VL-nested checkpoints both handled (same nesting rule as
    `utils.estimate_weight_bytes`); used only to sharpen the param_dir preflight.
    """
    try:
        with open(os.path.join(model_path, "config.json"), encoding="utf-8") as f:
            cfg = json.load(f)
    except (OSError, ValueError):
        return None
    text_cfg = cfg.get("text_config") or cfg
    return text_cfg.get("num_hidden_layers")


def _preflight_param_dir(param_dir, num_layers=None, quant_target=""):
    """Static, second-level preflight on --param_dir *before* the model loads.

    The runtime tally (`_require_ptq_params_landed`) only runs after assembly,
    and amct's `strict=False` load stays silent on partial misses -- so a
    mistyped filename or a truncated directory surfaces as a `[warn]`-only
    mixed object minutes in. This check is directory-only (no model, no NPU):
    name shape, per-index coverage, and -- when the layer count is already
    known from config.json -- a count match, reported as actionable lines
    before any expensive work starts.

    It also names the one question the directory cannot answer on its own:
    which attention target produced the attention params, both targets saving
    the same filename. `quant_target` is what the caller passed explicitly;
    empty means the target is being derived here.
    """
    if not param_dir:
        return
    if not os.path.isdir(param_dir):
        raise RuntimeError(
            f"--param_dir {param_dir} is not a directory. No ptq params can be "
            f"loaded from it, so the run would silently be direct quant."
        )
    names = os.listdir(param_dir)
    layer_units = {}
    stray = []
    for name in names:
        m = re.match(r"^layer_(\d+)_([a-z0-9_]+)\.pt$", name)
        if m:
            layer_units.setdefault(int(m.group(1)), set()).add(m.group(2))
        elif name.endswith(".pt"):
            stray.append(name)
    if not layer_units:
        raise RuntimeError(
            f"--param_dir {param_dir} holds no layer_<idx>_<unit>.pt files "
            f"(dir head={sorted(names)[:6]}). Nothing would load, so the run "
            f"would silently be direct quant; check the directory and --algos."
        )
    if stray:
        print(
            f"[warn] {param_dir} holds .pt files that do not match the "
            f"layer_<idx>_<unit>.pt naming (they will never load): {stray[:6]}"
        )
    idxs = sorted(layer_units)
    gaps = [i for i in range(idxs[0], idxs[-1] + 1) if i not in layer_units]
    if gaps:
        print(
            f"[warn] {param_dir} misses ptq params for layer indices {gaps}: "
            f"a hole inside the covered span usually means the ptq round was "
            f"interrupted rather than stopped at a boundary. Those layers stay "
            f"bf16 (the span assembly quantizes is reported as [quant-scope]), "
            f"so the evaluated object has a gap in it -- confirm the span is "
            f"what the user meant (constraints.md #4)."
        )
    if num_layers is not None and idxs[-1] >= num_layers:
        print(
            f"[warn] {param_dir} holds layer indices up to {idxs[-1]} but this "
            f"model has {num_layers} decoder layers: extra files never load."
        )
    # Coverage, not span end: a directory holding layer_1..31 of a 32-layer
    # model has no hole inside its span and its last index is in range, so
    # judging by idxs[-1] alone reads it as complete while layer 0 stays bf16.
    if num_layers is not None and len(idxs) < num_layers:
        uncovered = [i for i in range(num_layers) if i not in layer_units]
        shown = uncovered[:8]
        print(
            f"[ptq-pre] {param_dir} holds params for {len(idxs)} of "
            f"{num_layers} decoder layers (span {idxs[0]}..{idxs[-1]}, "
            f"uncovered {shown}{'...' if len(uncovered) > len(shown) else ''})."
        )
    units = sorted({u for us in layer_units.values() for u in us})
    print(
        f"[ptq-pre] {param_dir}: layers {idxs[0]}..{idxs[-1]} "
        f"({len(idxs)} indices), units {units}"
    )
    if not quant_target and any(_ATTN_PARAM_RE.match(n) for n in names):
        if not _attn_target_from_path(param_dir):
            print(
                f"[warn] {param_dir} holds attention ptq params, but both "
                f"attention targets save the same filename and this path "
                f"names neither, so the target is assumed to be attn-linear "
                f"(constraints.md #1). Params from an attn-cache round load "
                f"into different quantizers: confirm which target produced "
                f"them, and re-run with --quant_target attn-cache if so."
            )


# Algorithm modules built into every quant block. Must match the algorithms the
# ptq params were trained with, or loading them raises
# "Submodule '...algorithms.<algo>' is not found in module 'Quant...'"
# (see references/constraints.md #3).
DEFAULT_ALGOS = ["lwc", "lac"]

# amct's own CLI defaults `--end_block_idx` to 61 (`cli/llm/args.py:219`), which
# is DeepSeek-V3's layer count. This skill assembles *every* layer, so the value
# is only a placeholder until the model is loaded; `assemble_quant_model`
# overwrites it with the real `pipe.num_layers` (see the note there).
_AMCT_CLI_DEFAULT_END_BLOCK_IDX = 61


def _resolve_quant_target(model_name, quant_target, param_dir=""):
    """Explicit --quant_target wins; otherwise derive per amct semantics.

    amct treats mlp / moe / attn-linear / attn-cache as flat targets, one per
    PTQ round; a param_dir may hold several rounds' outputs side by side. The
    derivation reads the directory per class and unions what is actually there
    -- no assumed "base" target, and an attention-only checkpoint (MLP kept
    bf16 on purpose) evaluates as attention-only instead of being forced into
    "mlp,attn-linear". `model_name` is kept for error readability only.

    Attention needs one input the filenames cannot give: both attention
    targets save the same name, so that target is read off a `--param_dir`
    path component (`_attn_target_from_path`). When the path names neither,
    `attn-linear` is kept and `_preflight_param_dir` warns it was assumed.

    A directory with no recognizable class fails loudly: guessing would
    evaluate an object the user never produced (constraints.md #1).
    """
    if quant_target:
        return quant_target
    if not param_dir or not os.path.isdir(param_dir):
        raise RuntimeError(
            "--quant_target was not passed and --param_dir is missing or not a "
            "directory, so the target cannot be derived. Pass --quant_target "
            "explicitly (amct flat targets: mlp / moe / attn-linear / attn-cache)."
        )
    names = os.listdir(param_dir)
    # mlp/moe first, attention last: keeps the historical "mlp,attn-linear"
    # string shape (logs, [ptq-pre] output) byte-identical with the old code.
    targets = []
    if any(_MLP_PARAM_RE.match(n) for n in names):
        targets.append("mlp")
    if any(_MOE_PARAM_RE.match(n) for n in names):
        targets.append("moe")
    if any(_ATTN_PARAM_RE.match(n) for n in names):
        targets.append(_attn_target_from_path(param_dir) or "attn-linear")
    if not targets:
        raise RuntimeError(
            f"--quant_target was not passed and {param_dir} holds no "
            f"recognizable layer_<idx>_<unit>.pt class "
            f"(dir head={sorted(names)[:6]}). Nothing would load, so this run "
            f"would silently be direct quant; check the directory or pass "
            f"--quant_target explicitly (amct flat targets: mlp / moe / "
            f"attn-linear / attn-cache)."
        )
    return ",".join(targets)


def input_device(model):
    """Device the model's input_ids should go to (embedding's card, multi-card safe).

    Never falls back to a hardcoded `npu:0`: with `--device_map auto` the
    embedding can sit on any card, and feeding input_ids to the wrong one makes
    every forward die with a cross-device error (or, worse, silently run on a
    card that holds no weights). If neither the embedding nor `model.device`
    tells us where the model lives, say so and keep the reason.
    """
    try:
        return model.get_input_embeddings().weight.device
    except Exception as exc:
        device = getattr(model, "device", None)
        if device is None:
            raise RuntimeError(
                f"cannot tell which device holds the model's inputs: "
                f"get_input_embeddings() failed with {type(exc).__name__}: {exc}, "
                f"and {type(model).__name__} exposes no `.device`. Pass a model "
                f"loaded through this skill's load_float/assemble_quant_model."
            ) from exc
        return device


def _check_no_offload(model):
    dmap = getattr(model, "hf_device_map", {}) or {}
    off = sorted({str(v) for v in dmap.values() if str(v) in ("cpu", "disk")})
    if off:
        raise RuntimeError(
            f"device_map offload to {off}: insufficient HBM. amct quant can't load "
            f"offloaded weights → wrong results. Add cards (ASCEND_RT_VISIBLE_DEVICES) "
            f"or lower precision."
        )
    used = sorted({str(v) for v in dmap.values()}) if dmap else ["npu:0"]
    print(f"[device] model spans: {used}")


def _get_layers(model):
    """Locate the decoder layer ModuleList (dense: model.model.layers;
    VL-nested MoE: model.model.language_model.layers)."""
    for getter in (lambda m: m.model.layers, lambda m: m.model.language_model.layers):
        try:
            return getter(model)
        except AttributeError:
            continue
    raise RuntimeError("cannot locate decoder layers on this model")


def _validate_algos(algos):
    """Fail before assembly if an algo key is not in amct's ALGO_REGISTRY.

    Otherwise the mismatch surfaces much later as
    "Submodule '<unit>.algorithms.<algo>' is not found in module 'Quant...'"
    while loading ptq params, which does not point back to --algos.
    """
    from amct_pytorch.algorithms.registry_factory import ALGO_REGISTRY

    if not algos:
        raise RuntimeError(
            "--algos resolved to an empty list; pass at least one amct algorithm "
            f"key (registered: {ALGO_REGISTRY.list_all()})."
        )
    unknown = [a for a in algos if not ALGO_REGISTRY.has(a)]
    if unknown:
        raise RuntimeError(
            f"--algos {unknown} not in amct ALGO_REGISTRY "
            f"(registered: {ALGO_REGISTRY.list_all()}). The keys must match the "
            f"algorithms the ptq params were trained with."
        )
    return algos


def _tally(stats, per_target_result):
    """Accumulate amct's {target: {loaded, missing}} over layers."""
    for target, res in (per_target_result or {}).items():
        entry = stats.setdefault(target, {"loaded": 0, "missing": 0})
        entry["loaded"] += len(res.get("loaded") or [])
        entry["missing"] += len(res.get("missing") or [])
    return stats


def _require_ptq_params_landed(param_dir, stats, algos, quant_target=""):
    """Fail loud when a user-supplied param_dir loaded nothing.

    amct loads ptq params with strict=False: an empty/mismatched dir silently
    falls back to direct quant, so the run would still print numbers -- for a
    different object than the PTQ result the user asked to benchmark.

    The success path also scans the directory: "nothing loaded" was the only
    silent-narrowing it caught, but a *narrowly requested* target leaves amct
    with nothing to report either (it is never asked for the attn units), so an
    attn checkpoint evaluated as MLP-only used to print a clean
    `(mlp: loaded=N missing=0)`. That is a human trade-off, not a bug -- the
    narrow object may be exactly what was wanted -- so it warns and keeps going
    instead of raising (constraints.md #1).
    """
    if not param_dir:
        return
    detail = (
        ", ".join(
            f"{t}: loaded={s['loaded']} missing={s['missing']}"
            for t, s in sorted(stats.items())
        )
        or "no quant target reported"
    )
    total_loaded = sum(s["loaded"] for s in stats.values())
    total_missing = sum(s["missing"] for s in stats.values())
    if total_loaded == 0:
        try:
            head = sorted(os.listdir(param_dir))[:6]
        except OSError as exc:
            raise RuntimeError(
                f"--param_dir {param_dir} is unreadable ({exc}); no ptq params were "
                f"loaded ({detail}), so this run would silently be direct quant."
            ) from exc
        raise RuntimeError(
            f"no ptq params were loaded from --param_dir {param_dir} ({detail}); "
            f"amct would silently fall back to direct quant, so these numbers would "
            f"not describe your PTQ result. Expected files named "
            f"layer_<idx>_<unit>.pt, dir head={head}. Check --param_dir, "
            f"--model_name/--quant_target and --algos={','.join(algos)}."
        )
    if total_missing:
        print(
            f"[warn] {total_missing} quant unit(s) found no ptq params under "
            f"{param_dir} ({detail}); those units fall back to direct quant, so the "
            f"evaluated object is mixed PTQ/direct."
        )
    else:
        print(f"[ptq] params loaded from {param_dir} ({detail})")
    if not has_attention_quant(quant_target) and _param_dir_has_attn(param_dir):
        print(
            f"[warn] {param_dir} holds layer_*_self_attn/linear_attn params but "
            f"this run evaluated quant_target={quant_target!r} only: attention "
            f"stays bf16, so the object under test is MLP-only and the score "
            f"does not describe the fully-quantized checkpoint. State that in the "
            f"report; to evaluate attention too, confirm the cost with the user "
            f"(KV cache off, minutes per question) and re-run with "
            f"--allow_attn_quant."
        )


def load_float(model_path, device_map, trust_remote_code=False):
    """Load the bf16 float model.

    Public: `livecodebench_eval.py` loads its `--bf16` baseline through it, so
    both sides of a comparison go through the same dtype/device path.

    `trust_remote_code` is an explicit opt-in, matching amct's own CLI default
    (`amct_pytorch/cli/llm/args.py`): it executes custom modeling code from the
    given directory, so it must never be silently on. See constraints.md #5.

    The weight-dtype kwarg name differs across transformers releases (`dtype`
    from 4.56, `torch_dtype` before); `dtype_kwargs()` picks the one the
    installed version accepts. See env-and-portability.md's dtype section.
    """
    import torch_npu  # noqa: F401
    from transformers import AutoModelForCausalLM

    if device_map:
        m = AutoModelForCausalLM.from_pretrained(
            model_path,
            **dtype_kwargs(torch.bfloat16),
            device_map=device_map,
            trust_remote_code=trust_remote_code,
        )
        _check_no_offload(m)
    else:
        m = AutoModelForCausalLM.from_pretrained(
            model_path,
            **dtype_kwargs(torch.bfloat16),
            trust_remote_code=trust_remote_code,
        ).to("npu:0")
    return m.eval()


def build_pipeline_args(
    model,
    model_name,
    param_dir,
    quant_target,
    bit_cfg,
    seq_len,
    output_dir,
    algos=None,
    trust_remote_code=False,
    quant_dtype="int",
    is_per_tensor=False,
):
    from amct_pytorch.quantization.bit_policy import BitPolicy

    quant_targets = split_quant_targets(quant_target)
    ns = argparse.Namespace(
        model=model,
        model_name=model_name,
        trust_remote_code=trust_remote_code,
        device="npu:0",
        # Pinned, not a CLI flag: amct's `--granularity` also takes "model"
        # (whole-model PPL path) and "tensor" (deploy path), and neither builds
        # the per-layer quant blocks this assembly swaps into the decoder stack.
        # "model" in particular does not quantize at all, so exposing it here
        # would let a caller score an unquantized model and call it a delta.
        granularity="block",
        seed=0,
        quant_target=quant_targets,
        seq_len=seq_len,
        data_dir="",
        output_dir=output_dir,
        nsamples=128,
        eval_mode="quant",
        quant_dtype=quant_dtype,
        cali_bsz=4,
        base_lr=1e-5,
        optimizer="adamw",
        weight_decay=0.0,
        momentum=0.9,
        lr_scheduler="cosine",
        min_lr=0.0,
        lr_step_size=1,
        lr_gamma=0.1,
        epochs=15,
        algos=list(algos) if algos is not None else list(DEFAULT_ALGOS),
        is_per_tensor=is_per_tensor,
        k_size=128,
        start_block_idx=0,
        end_block_idx=_AMCT_CLI_DEFAULT_END_BLOCK_IDX,
        bit_config=bit_cfg,
        attn_linear_param_dir=param_dir if "attn-linear" in quant_targets else "",
        attn_cache_param_dir=param_dir if "attn-cache" in quant_targets else "",
        moe_mlp_param_dir=param_dir
        if {"mlp", "moe"}.intersection(quant_targets)
        else "",
        wikitext_final_out="",
    )
    ns.bit_policy = BitPolicy.from_yaml(bit_cfg)
    return ns


def assemble_quant_model(
    model,
    model_name,
    param_dir,
    bit_cfg,
    seq_len,
    device_map=None,
    quant_target=None,
    output_dir="/tmp/qbe_out",
    allow_attn_quant=False,
    algos=None,
    trust_remote_code=False,
    quant_dtype="int",
    is_per_tensor=False,
    direct=False,
):
    from amct_pytorch.common.models.llm import register_llm_models
    from amct_pytorch.quantization.dtypes import register_dtype
    from amct_pytorch.algorithms.quant import register_algorithms
    from amct_pytorch.common.models import MODEL_REGISTRY
    from amct_pytorch.quantization.modules.quant_linear import QuantLinear
    from amct_pytorch.common.models.llm.common.quant_apply import set_model_to_observe

    # Before anything is loaded: a missing bit config used to surface inside
    # the amct pipeline, after the float model was on the card.
    require_bit_config(bit_cfg)
    register_llm_models()
    register_dtype()
    register_algorithms()

    qt = _resolve_quant_target(model_name, quant_target, param_dir)
    try:
        ensure_supported_quant_target(qt, allow_attn_quant=allow_attn_quant)
    except RuntimeError as exc:
        if quant_target or not _param_dir_has_attn(param_dir):
            raise
        raise RuntimeError(
            f"{exc} --quant_target was not passed: it was derived as {qt!r} "
            f"because {param_dir} holds layer_*_self_attn/linear_attn params, "
            f"i.e. the checkpoint was quantized with attention. Confirm the cost "
            f"with the user (KV cache off, minutes per question) and re-run with "
            f"--allow_attn_quant; to evaluate the narrow non-attention object "
            f"instead, pass --quant_target with only the classes you want "
            f"(e.g. mlp / moe) and report that the score does not describe the "
            f"fully-quantized checkpoint."
        ) from exc
    # Direct conversion builds no algorithm module: amct iterates
    # `self.algorithms.values()` (quant_base.py) and an empty list makes that a
    # no-op, so the scales come from the quantizers alone. `_validate_algos`
    # raises on an empty list by design -- it guards a *PTQ* assembly, where
    # empty algos means the params cannot land -- so it is skipped here.
    algo_list = [] if direct else split_list_arg(algos, DEFAULT_ALGOS)
    if not direct:
        _validate_algos(algo_list)
    args = build_pipeline_args(
        model,
        model_name,
        param_dir,
        qt,
        bit_cfg,
        seq_len,
        output_dir,
        algo_list,
        trust_remote_code=trust_remote_code,
        quant_dtype=quant_dtype,
        is_per_tensor=is_per_tensor,
    )
    pipe = MODEL_REGISTRY.get(model_name)(args)
    print(
        f"[assemble] loading float_model (bf16, device_map={device_map}, target={qt})..."
    )
    m = load_float(model, device_map, trust_remote_code)
    layers = _get_layers(m)
    n = pipe.num_layers
    # amct's own CLI defaults `--end_block_idx` to 61 (DeepSeek-V3's layer
    # count), and `build_pipeline_args` inherits that literal. Nothing on this
    # assemble path reads it -- the loop below is driven by `pipe.num_layers` --
    # but `pipe` keeps the same Namespace object (`base.py`: `self.args = args`),
    # so any amct helper that does read it (ptq_provider metadata, llm_ptq's
    # range) would silently describe the wrong layer span on a model that is not
    # 61 layers deep. Sync it to the real count instead of leaving a
    # model-specific magic number in a generic skill.
    args.end_block_idx = n
    if n > len(layers):
        raise RuntimeError(
            f"amct pipeline reports num_layers={n} but the model's decoder stack "
            f"only has {len(layers)} layers (model_name={model_name!r}). The "
            f"assembly loop below would IndexError on layers[{n - 1}] and leave "
            f"a half-swapped model. Check --model_name against the checkpoint."
        )
    if n < len(layers):
        # Legitimate for MTP checkpoints: amct adapters set num_layers to
        # config.num_hidden_layers while the HF stack also carries the trailing
        # MTP block(s) (glm5_2 / hy_v3 quantize num_hidden_layers of
        # num_hidden_layers + num_nextn). Those tail layers stay bf16 on purpose,
        # but a wrong --model_name looks identical from here, so say it out loud
        # rather than failing a path amct itself runs.
        print(
            f"[warn] amct pipeline quantizes num_layers={n} but the checkpoint "
            f"has {len(layers)} decoder layers: the trailing {len(layers) - n} "
            f"stay bf16. Expected for MTP models (glm5_2 / hy_v3). If this "
            f"checkpoint has no MTP layer, --model_name={model_name!r} likely "
            f"does not match it and the score below describes a half-quantized "
            f"model."
        )
    print(
        f"[assemble] algos={','.join(algo_list) or '<none: direct quant>'} "
        f"param_dir={param_dir or '<none>'}"
    )
    if direct:
        # Nothing on disk names the span: amct's own direct path quantizes every
        # layer of `pipeline.num_layers` (workflows/llm_eval.py loops
        # range(self.pipeline.num_layers)), so that is the object here too.
        quant_layers = set(range(n))
        source = "<direct quant: no ptq params>"
        print(
            "[direct] no ptq params: the score describes amct direct "
            "conversion, not a PTQ checkpoint."
        )
    else:
        # Quantize exactly the span the user's ptq round covered. Without this the
        # loop below quantizes all `n` layers, and any layer the directory has no
        # params for keeps amct's construction defaults instead of staying bf16 --
        # a mixed PTQ/direct object nobody produced (constraints.md #4).
        quant_layers = _param_layer_indices(param_dir, qt)
        if not quant_layers:
            raise RuntimeError(
                f"--param_dir {param_dir} holds no layer_<idx>_<unit>.pt file for the "
                f"requested quant_target {qt!r}, so no layer would be quantized. Unit "
                f"names are mlp / expert_<j> / self_attn / linear_attn: check the "
                f"directory against the target that produced it."
            )
        source = param_dir
    quant_scope = _describe_quant_scope(quant_layers, n, source)
    # Multi-card: device_map=auto attaches accelerate AlignDevicesHooks that move
    # activations to each layer's card as data flows layer→layer. Replacing a layer
    # object drops its hook → cross-card activations no longer realigned → matmul
    # "tensors on npu:0 and npu:1" crash at generate. So we re-attach the original
    # layer's hook (carries execution_device) onto the swapped-in quant block.
    try:
        from accelerate.hooks import add_hook_to_module
    except Exception as exc:  # keep the reason for the fail-fast below
        add_hook_to_module = None
        hook_import_error = exc
    else:
        hook_import_error = None
    load_stats = {}
    quantized = 0
    target_count = len(quant_layers)
    for i in range(n):
        if i not in quant_layers:
            # No params were trained for this layer: leave the original bf16
            # module in place (its accelerate hook, if any, stays attached).
            continue
        # place quant block on the SAME card as the original layer (multi-card safe)
        dev = next(layers[i].parameters()).device
        old_hook = getattr(layers[i], "_hf_hook", None)
        if old_hook is not None and add_hook_to_module is None:
            # Silently skipping the re-attach produces a multi-card model whose
            # cross-card activations are never realigned; the failure surfaces
            # much later as "Expected all tensors to be on the same device"
            # inside generate, pointing at nothing in this skill.
            raise RuntimeError(
                f"layer {i} carries an accelerate hook (device_map={device_map!r} "
                f"spread the model over several cards) but accelerate.hooks."
                f"add_hook_to_module is unavailable ({hook_import_error}). The "
                f"quant block would lose its cross-card hook and generate() "
                f"would crash on mismatched devices. Install/upgrade accelerate, "
                f"or run single-card with an empty --device_map."
            )
        qblock = pipe.build_quant_block(i)
        if not direct:
            try:
                _tally(
                    load_stats,
                    pipe.load_selected_layer_ptq_params(i, qblock, strict=False),
                )
            except KeyError as exc:
                # amct raises a bare KeyError naming a submodule the built quant
                # block does not have. Neither of its two causes is visible in the
                # message, and both are a mismatch between the directory and how
                # this run assembled it.
                raise RuntimeError(
                    f"layer {i}: the ptq params under {param_dir} name submodules "
                    f"this quant block does not have ({exc}). Two causes, both a "
                    f"mismatch between the directory and this run's assembly: the "
                    f"attention target -- attn-linear and attn-cache save the same "
                    f"filename, so a cache round's file loaded into a linear block "
                    f"(or the reverse) names quantizers that were never built "
                    f"(constraints.md #1); or --algos={','.join(algo_list)} is not "
                    f"the set the params were trained with, so the file names "
                    f"'<unit>.algorithms.<algo>' submodules absent here "
                    f"(constraints.md #3). Confirm which target and which algorithms "
                    f"produced {param_dir} and re-run with those."
                ) from exc
        set_model_to_observe(qblock, False)
        for mod in qblock.modules():
            if isinstance(mod, QuantLinear):
                mod.eval_mode = True
                mod.cached_eval_weight = None
        qblock = qblock.to(dev).eval().bfloat16()
        if old_hook is not None and add_hook_to_module is not None:
            add_hook_to_module(qblock, old_hook)  # preserve cross-card activation move
        layers[i] = qblock
        quantized += 1
        if quantized % 6 == 0 or quantized == target_count:
            print(
                f"[assemble] {quantized}/{target_count} layers quantized"
                f"{' (+hook)' if old_hook is not None else ''}"
            )
    _require_ptq_params_landed(param_dir, load_stats, algo_list, qt)
    import torch_npu

    torch_npu.npu.empty_cache()
    if has_attention_quant(qt):
        m.config.use_cache = False
        print(
            "[warn] attention quant enabled; model.config.use_cache=False for generate"
        )
    print("[assemble] done.")
    # pipe.tokenizer is deliberately NOT returned: amct builds it with the
    # use_fast default and its own --trust_remote_code value, so it can differ
    # from the tokenizer used on the --bf16 baseline. Callers keep their own
    # tokenizer so
    # both sides of the comparison tokenize identically. Return the resolved
    # quant_target instead (needed to decide generate(use_cache=...)), plus the
    # quantized layer span (part of the object's identity, recorded in the
    # result payload).
    return m, qt, quant_scope


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="model dir")
    ap.add_argument(
        "--model_name",
        required=True,
        help="amct pipeline key (qwen3, qwen3_5, qwen3_5_moe, qwen3_6_moe)",
    )
    ap.add_argument(
        "--param_dir",
        default="",
        help="ptq params dir (mlp for dense, moe for MoE); ignored with --bf16",
    )
    ap.add_argument(
        "--quant_target",
        default="",
        help=(
            "override the derived target. Omit to derive from --param_dir: a dir "
            "holding layer_*_self_attn/linear_attn params evaluates '<mlp|moe>,"
            "attn-linear' (the object the user quantized), otherwise '<mlp|moe>'."
        ),
    )
    ap.add_argument(
        "--algos",
        default=None,
        help=(
            "comma-separated amct algorithm keys built into every quant block "
            "(registered: lwc, lac, let, autoround, flatquant, gptq). Must match "
            f"the algorithms the ptq params were trained with; omit for "
            f"{','.join(DEFAULT_ALGOS)}. Not accepted with --direct, which "
            "builds no algorithm module."
        ),
    )
    ap.add_argument(
        "--bit_config",
        default=DEFAULT_BIT_CONFIG,
        help="default derived from $AMCT_HOME",
    )
    ap.add_argument(
        "--quant_dtype",
        default="int",
        choices=["int", "mxfp", "hifp", "fp"],
        help=(
            "amct quantization data type, same choices as amct's own "
            "--quant_dtype. Must match the dtype the ptq params were trained "
            "with: 'int' builds int quantizers, so an mxfp / HiFloat8 checkpoint "
            "assembled under the default evaluates a different object."
        ),
    )
    ap.add_argument(
        "--is_per_tensor",
        action="store_true",
        help=(
            "use per-tensor statistics in activation clipping (amct's "
            "--is_per_tensor); pass it when the ptq params were trained with it"
        ),
    )
    ap.add_argument(
        "--output_dir",
        required=True,
        help=(
            "output dir handed to the amct pipeline, same parameter name as "
            "upstream's ptq param dir convention; the eval path writes no file "
            "here (results go to --output_path). Required so the two entry "
            "points do not fork (SKILL.md #4.3)."
        ),
    )
    ap.add_argument("--seq_len", type=int, default=2048)
    ap.add_argument(
        "--device_map",
        default=None,
        help=(
            "'auto' for multi-card (35B MoE); ''=single npu:0 (8B). Omit to infer "
            "from config.json (fails when the checkpoint size is unknowable)."
        ),
    )
    ap.add_argument(
        "--task",
        required=True,
        help=(
            "benchmark task (see SKILL.md's task table). Required: the old "
            "mmlu_pro default picked the benchmark for the user."
        ),
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="sample cap; omit for full benchmark",
    )
    ap.add_argument("--max_gen_toks", type=int, default=512)
    ap.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="lm-eval batch size (was hardcoded to 1)",
    )
    ap.add_argument(
        "--bootstrap_iters",
        type=int,
        default=0,
        help=(
            "lm-eval bootstrap iterations for stderr. 0 (default) emits no error "
            "bars, so a delta must not be called statistically insignificant."
        ),
    )
    ap.add_argument(
        "--output_path",
        default="",
        help=(
            "write the lm-eval result dict plus the run's own parameters to this "
            "JSON, so compare_results.py reads them back instead of parsing the "
            "log and taking --sample_len / --limited by hand."
        ),
    )
    ap.add_argument(
        "--direct",
        action="store_true",
        help=(
            "evaluate amct direct conversion instead of a PTQ checkpoint: no ptq "
            "params are read, every decoder layer gets a quant block, no "
            "algorithm module is built, and scales are computed on the fly. "
            "Requires an explicit --quant_target (there is no directory to "
            "derive it from) and rejects --param_dir / --algos / --bf16."
        ),
    )
    ap.add_argument(
        "--bf16", action="store_true", help="baseline: original model, no quant"
    )
    ap.add_argument(
        "--allow_attn_quant",
        action="store_true",
        help="explicitly allow attn-linear/attn-cache quant targets",
    )
    ap.add_argument("--smoke", action="store_true", help="only generate smoke test")
    ap.add_argument("--log_samples", action="store_true")
    ap.add_argument(
        "--trust_remote_code",
        action="store_true",
        default=False,
        help=(
            "allow HuggingFace to execute custom modeling code from --model. "
            "Default off, same as amct's own CLI; only enable for a model "
            "directory you trust (constraints.md #5)."
        ),
    )
    a = ap.parse_args()

    a.limit = resolve_sample_limit(a.limit)
    ensure_direct_mode_args(a.direct, a.bf16, a.param_dir, a.quant_target, a.algos)
    algo_list = [] if a.direct else split_list_arg(a.algos, DEFAULT_ALGOS)

    check_environment(require_lm_eval=not a.smoke)
    # An omitted --device_map is inferred from config.json; an explicitly passed
    # '' still means single card. Guessing on an unknowable size fails here
    # instead of OOMing inside from_pretrained (env-and-portability.md #2).
    dmap = infer_device_map(a.model) if a.device_map is None else a.device_map
    dmap = dmap or None
    tag = "bf16" if a.bf16 else "quant"
    # One line naming everything a reader of the log has to know to judge
    # whether two runs are comparable: the three parameters that decide the
    # accuracy (task / limit / max_gen_toks) plus what was actually evaluated.
    print(
        f"[run] tag={tag} task={a.task} model_name={a.model_name} "
        f"param_dir={a.param_dir or '<none>'} direct={a.direct} "
        f"quant_target={a.quant_target or '<derived from param_dir>'} "
        f"algos={','.join(algo_list) or '<none: direct quant>'} "
        f"bit_config={a.bit_config} "
        f"quant_dtype={a.quant_dtype} is_per_tensor={a.is_per_tensor} "
        f"limit={a.limit} "
        f"max_gen_toks={a.max_gen_toks} seq_len={a.seq_len} batch_size={a.batch_size} "
        f"bootstrap_iters={a.bootstrap_iters} device_map={dmap or '<single npu:0>'} "
        f"output_dir={a.output_dir} output_path={a.output_path or '<none>'} "
        f"trust_remote_code={a.trust_remote_code} smoke={a.smoke}"
    )
    from transformers import AutoTokenizer

    warn_trust_remote_code(a.model, a.trust_remote_code)
    # Second-level static preflight, before the model is loaded: a mistyped /
    # truncated --param_dir used to surface only after assembly as a silent
    # mixed object (constraints.md #4). Layer count (when config.json gives
    # one) sharpens the report; None keeps it shape-only.
    if not a.bf16 and not a.direct:
        _preflight_param_dir(
            a.param_dir,
            num_layers=_config_num_layers(a.model),
            quant_target=a.quant_target,
        )
    lm_task = TASK_ALIASES.get(a.task, a.task)
    # lm-eval has no task index entry for the LiveCodeBench names this skill
    # accepts: they are generated here and scored by the official lcb_runner, so
    # the result key is a bare `pass@1` in a separate `*_eval.json`, not a
    # "<metric>,<filter>" pair. Without this branch the name reaches lm-eval and
    # fails as a task-index miss that never says which entry point to use.
    if lm_task in NON_LM_EVAL_TASKS or a.task in NON_LM_EVAL_TASKS:
        sys.exit(
            f"--task {a.task!r} is not an lm-eval task: it is scored by the "
            f"official lcb_runner. Generate with livecodebench_eval.py instead "
            f"(same assembly, writes the custom_evaluator input JSON), then "
            f"score it and compare the two *_eval.json files -- SKILL.md #4.2 "
            f"and references/livecodebench.md."
        )
    if not a.smoke:
        # Before the model is loaded: an unregistered / misspelled --task only
        # fails deep inside lm-eval, after assembly, as a network timeout that
        # does not name --task (see utils.check_task_offline_ready).
        wanted = check_task_offline_ready(lm_task)
        print(f"[task] {lm_task} reads {sorted(wanted)} from $BENCH_ASSETS")
        # Importing builds the shim map (`_LOCAL = local_dataset_map()`), which
        # is where a contradictory datasets.json -- two tasks on one
        # (repo_id, dataset_name) declaring different local data -- raises. Do
        # it here, not next to install() below: that line runs after the float
        # model is on the card and every quant block is assembled, so a registry
        # bug would cost minutes and report itself far from its cause
        # (constraints.md #2). `install()` stays where it is: patching
        # load_dataset only matters once lm_eval is imported.
        import bench_offline  # noqa: F401
    tok = AutoTokenizer.from_pretrained(
        a.model, use_fast=False, trust_remote_code=a.trust_remote_code
    )
    quant_target = a.quant_target or ""
    quant_scope = None
    if a.bf16:
        print(f"[bf16] loading original model (no quant, device_map={dmap})...")
        model = load_float(a.model, dmap, a.trust_remote_code)
    else:
        if not a.param_dir and not a.direct:
            sys.exit("--param_dir required unless --bf16 or --direct")
        ensure_single_attention_target(a.quant_target)
        model, quant_target, quant_scope = assemble_quant_model(
            a.model,
            a.model_name,
            a.param_dir,
            a.bit_config,
            a.seq_len,
            device_map=dmap,
            quant_target=(a.quant_target or None),
            output_dir=a.output_dir,
            allow_attn_quant=a.allow_attn_quant,
            algos=algo_list,
            trust_remote_code=a.trust_remote_code,
            quant_dtype=a.quant_dtype,
            is_per_tensor=a.is_per_tensor,
            direct=a.direct,
        )

    generate_use_cache = not has_attention_quant(quant_target)
    print(f"[smoke] generate (use_cache={generate_use_cache})...")
    ids = tok("The capital of France is", return_tensors="pt").input_ids.to(
        input_device(model)
    )
    with torch.no_grad():
        out = model.generate(
            ids, max_new_tokens=16, do_sample=False, use_cache=generate_use_cache
        )
    print(
        "[smoke] ->", repr(tok.decode(out[0][ids.shape[1] :], skip_special_tokens=True))
    )
    if a.smoke:
        return

    import bench_offline

    bench_offline.install()
    from lm_eval.models.huggingface import HFLM
    from lm_eval import simple_evaluate

    lm = HFLM(
        pretrained=model,
        tokenizer=tok,
        batch_size=a.batch_size,
        max_gen_toks=a.max_gen_toks,
    )
    print(f"[eval] {tag} task={lm_task} limit={a.limit} max_gen_toks={a.max_gen_toks}")
    res = simple_evaluate(
        model=lm,
        tasks=[lm_task],
        limit=a.limit,
        bootstrap_iters=a.bootstrap_iters,
        log_samples=a.log_samples,
    )
    for t, mtr in res["results"].items():
        print(f"[RESULT] {tag} {t}: {mtr}")
    if a.output_path:
        # Everything a report needs and the log does not reliably carry: a full
        # run prints no sample count, and compare_results.py used to take
        # --sample_len / --limited by hand, where a wrong value goes unchallenged.
        n_samples = (res.get("n-samples") or {}).get(lm_task) or {}
        payload = {
            "tag": tag,
            "task": lm_task,
            "requested_task": a.task,
            "sample_len": n_samples.get("effective", n_samples.get("original")),
            **sample_limit_fields(a.limit),
            "max_gen_toks": a.max_gen_toks,
            "batch_size": a.batch_size,
            "bootstrap_iters": a.bootstrap_iters,
            "model_name": a.model_name,
            # A bf16 baseline builds no quant block, so these describe nothing.
            # Recording the CLI defaults here anyway would let a report read a
            # bit width / algorithm set off the baseline JSON, and would make
            # compare_results.py's quant-side comparability keys compare a real
            # value against a meaningless one. None = "not applicable".
            # `seq_len` belongs here for the same reason: it only feeds the
            # quant assembly, so on the bf16 side the CLI default describes no
            # model.
            "seq_len": None if a.bf16 else a.seq_len,
            "param_dir": None if a.bf16 or a.direct else a.param_dir,
            "quant_target": None if a.bf16 else quant_target,
            # A comma-joined string, not a list: this is a comparability
            # key, and result JSONs written before --direct carried the
            # string form. None on the bf16 and direct sides -- neither
            # builds an algorithm module, so there is nothing to align.
            "algos": None if a.bf16 or a.direct else ",".join(algo_list),
            # Which of the two quant objects was scored: False = a PTQ
            # checkpoint whose params came from param_dir, True = amct direct
            # conversion. Same treatment as quant_scope -- recorded, not a
            # comparability key, because the bf16 side quantizes nothing.
            "direct": a.direct,
            "bit_config": None if a.bf16 else a.bit_config,
            "quant_dtype": None if a.bf16 else a.quant_dtype,
            "is_per_tensor": None if a.bf16 else a.is_per_tensor,
            # Which decoder layers were actually quantized: derived from
            # --param_dir, so a ptq round that covered part of the stack scores
            # a partially-quantized checkpoint. None on the bf16 side.
            "quant_scope": quant_scope,
            "device_map": dmap,
            "bf16": a.bf16,
            "results": res["results"],
        }
        out_dir = os.path.dirname(os.path.abspath(a.output_path))
        os.makedirs(out_dir, exist_ok=True)
        with open(a.output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"[out] wrote {a.output_path} (sample_len={payload['sample_len']})")


if __name__ == "__main__":
    main()
