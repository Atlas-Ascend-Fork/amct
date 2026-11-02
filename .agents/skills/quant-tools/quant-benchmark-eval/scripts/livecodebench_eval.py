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
"""Generate LiveCodeBench custom-evaluator inputs from assembled AMCT models.

This script runs only the Ascend/NPU side: load bf16 or assembled fake-quant
AMCT model, generate code for LiveCodeBench codegeneration prompts, and dump the
official lcb_runner custom-evaluator JSON format:

[
    {"question_id": "id1", "code_list": ["code1"]},
    {"question_id": "id2", "code_list": ["code2"]}
]

Run lcb_runner.runner.custom_evaluator on that JSON to compute the official
hidden-test pass@1/pass@k metric.

The quant side accepts the same two objects as assemble_and_eval.py: a PTQ
checkpoint read from --param_dir, or amct direct conversion under --direct (no
params, every decoder layer quantized, no algorithm module). Kept in step with
that entry point so the two do not diverge.
"""

import argparse
import json
import os
import re
import sys

import pandas as pd
import torch

SKILL_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SKILL_DIR)

from assemble_and_eval import (  # noqa: E402
    DEFAULT_ALGOS,
    load_float,
    assemble_quant_model,
    input_device,
    _config_num_layers,
    _preflight_param_dir,
)
from config import DEFAULT_BIT_CONFIG, local_dataset_map, load_dataset_registry  # noqa: E402
from utils import (  # noqa: E402
    check_environment,
    ensure_direct_mode_args,
    ensure_single_attention_target,
    has_attention_quant,
    infer_device_map,
    lcb_run_meta_path,
    local_dataset_files,
    read_dataset_file,
    resolve_sample_limit,
    sample_limit_fields,
    split_list_arg,
    validate_lcb_docs,
    warn_trust_remote_code,
)


def resolve_dataset_files(task="livecodebench_release_v6"):
    """Local files backing `task`, resolved through datasets.json + $BENCH_ASSETS.

    Called *before* the model is loaded: an unprefetched cache (or a partial
    `all_of` shard set, deferred as `config.UnresolvedSplit`) used to surface
    only after the float model was on the card and every quant block assembled,
    i.e. minutes of work and a traceback that never mentions `--dataset_task`.
    Resolving first turns it into an immediate, actionable error and also builds
    the shim map, so a contradictory datasets.json fails here too
    (constraints.md #2).
    """
    files = local_dataset_files(task, load_dataset_registry(), local_dataset_map())
    print(f"[data] {task}: {len(files)} local file(s) from $BENCH_ASSETS")
    return files


def load_lcb_docs(files, limit):
    """Read already-resolved dataset files into LiveCodeBench doc records."""
    frames = [read_dataset_file(fp) for fp in files]
    df = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
    if limit is not None:
        df = df.head(limit)
    return df.to_dict("records")


def build_prompt(doc):
    question = (
        doc.get("question_content") or doc.get("prompt") or doc.get("content") or ""
    )
    starter = doc.get("starter_code") or ""
    prompt = (
        "You are solving a programming problem. Write a complete Python 3 solution.\n"
        "Return only code inside one Python code block.\n\n"
        f"{question}\n"
    )
    if starter:
        prompt += f"\nStarter code:\n```python\n{starter}\n```\n"
    return prompt


def extract_code(text):
    blocks = re.findall(
        r"```(?:python|py)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL
    )
    if blocks:
        return blocks[-1].strip()
    # Fall back to stripping common chatty preambles.
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.lstrip().startswith(("import ", "from ", "def ", "class ")):
            return "\n".join(lines[i:]).strip()
    return text.strip()


def generate_code(model, tok, prompt, max_new_tokens, use_cache=True):
    encoded = tok(prompt, return_tensors="pt")
    encoded = {k: v.to(input_device(model)) for k, v in encoded.items()}
    with torch.no_grad():
        out = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=use_cache,
        )
    return tok.decode(out[0][encoded["input_ids"].shape[1] :], skip_special_tokens=True)


def write_json(path, rows):
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)


def write_jsonl(path, rows):
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument(
        "--model_name",
        required=True,
        help=(
            "amct pipeline key (qwen3, qwen3_5, qwen3_5_moe, qwen3_6_moe). "
            "Required: a MoE checkpoint assembled under a dense key loads no "
            "expert params. Kept in step with assemble_and_eval.py so the two "
            "entry points do not diverge."
        ),
    )
    ap.add_argument("--param_dir", default="")
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
            "comma-separated amct algorithm keys built into every quant block; "
            "must match the algorithms the ptq params were trained with; omit "
            f"for {','.join(DEFAULT_ALGOS)}. Not accepted with --direct, which "
            "builds no algorithm module."
        ),
    )
    ap.add_argument("--bit_config", default=DEFAULT_BIT_CONFIG)
    ap.add_argument(
        "--quant_dtype",
        default="int",
        choices=["int", "mxfp", "hifp", "fp"],
        help=(
            "amct quantization data type, same choices as amct's own "
            "--quant_dtype; must match the dtype the ptq params were trained "
            "with. Kept in step with assemble_and_eval.py so the two entry "
            "points do not diverge."
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
            "here (results go to --output_json). Required so the two entry "
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
        "--limit",
        type=int,
        default=None,
        help="debug only; official custom_evaluator requires the full selected release",
    )
    ap.add_argument("--max_gen_toks", type=int, default=1024)
    ap.add_argument(
        "--output_json",
        required=True,
        help=(
            "official lcb_runner custom_output_file JSON. Required: the old "
            "relative default landed the generation next to whatever cwd the "
            "script was invoked from (SKILL.md #4.3)."
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
            "derive it from) and rejects --param_dir / --algos / --bf16. Kept "
            "in step with assemble_and_eval.py so the two entry points do not "
            "diverge."
        ),
    )
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument(
        "--allow_attn_quant",
        action="store_true",
        help="explicitly allow attn-linear/attn-cache quant targets",
    )
    ap.add_argument(
        "--dataset_task",
        choices=["livecodebench_release_v6"],
        default="livecodebench_release_v6",
        help=(
            "LiveCodeBench registry task (cumulative release set): smoke runs "
            "use --limit on it, the official score needs the full set."
        ),
    )
    ap.add_argument(
        "--release_version",
        default="release_v6",
        help="lcb_runner release_version to use when scoring this output",
    )
    ap.add_argument(
        "--raw_output_jsonl",
        default="",
        help="optional JSONL with prompt and raw generation for debugging",
    )
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

    check_environment()
    from transformers import AutoTokenizer

    # Same inference as assemble_and_eval.py: an omitted --device_map is derived
    # from config.json, an explicitly passed '' still means single card.
    dmap = (infer_device_map(a.model) if a.device_map is None else a.device_map) or None
    tag = "bf16" if a.bf16 else "quant"
    print(
        f"[run] tag={tag} script=livecodebench_eval dataset_task={a.dataset_task} "
        f"release_version={a.release_version} model_name={a.model_name} "
        f"param_dir={a.param_dir or '<none>'} direct={a.direct} "
        f"quant_target={a.quant_target or '<derived from param_dir>'} "
        f"algos={','.join(algo_list) or '<none: direct quant>'} "
        f"bit_config={a.bit_config} "
        f"quant_dtype={a.quant_dtype} is_per_tensor={a.is_per_tensor} "
        f"limit={a.limit} "
        f"max_gen_toks={a.max_gen_toks} seq_len={a.seq_len} "
        f"device_map={dmap or '<single npu:0>'} output_dir={a.output_dir} "
        f"output_json={a.output_json} trust_remote_code={a.trust_remote_code}"
    )
    # Before the model is loaded: an unprefetched / partially prefetched
    # LiveCodeBench cache must fail here, not after assembly (see
    # resolve_dataset_files).
    dataset_files = resolve_dataset_files(a.dataset_task)
    # Same rule for --param_dir, and the same reason to run it here rather than
    # inside assembly: a mistyped / truncated directory, or attention params
    # whose target the path does not name, must be reported before weights go on
    # a card instead of as a mixed object minutes in (constraints.md #1 / #4).
    if not a.bf16 and not a.direct:
        _preflight_param_dir(
            a.param_dir,
            num_layers=_config_num_layers(a.model),
            quant_target=a.quant_target,
        )
    warn_trust_remote_code(a.model, a.trust_remote_code)
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

    docs = load_lcb_docs(dataset_files, a.limit)
    validate_lcb_docs(docs)
    custom_outputs = []
    raw_rows = []
    if a.limit is not None:
        print(
            "[warn] --limit creates a partial output file; official custom_evaluator "
            "will reject it unless the selected release has the same number of problems."
        )
    generate_use_cache = not has_attention_quant(quant_target)
    print(
        f"[generate] {tag} livecodebench dataset_task={a.dataset_task} "
        f"release_version={a.release_version} samples={len(docs)} "
        f"max_gen_toks={a.max_gen_toks}"
    )
    for idx, doc in enumerate(docs, 1):
        prompt = build_prompt(doc)
        raw = generate_code(
            model, tok, prompt, a.max_gen_toks, use_cache=generate_use_cache
        )
        code = extract_code(raw)
        question_id = str(doc.get("question_id", ""))
        custom_outputs.append(
            {
                "question_id": question_id,
                "code_list": [code],
            }
        )
        raw_rows.append(
            {
                "idx": idx,
                "question_id": question_id,
                "prompt": prompt,
                "generation": raw,
                "code": code,
            }
        )
        print(
            f"[sample] {idx}/{len(docs)} id={question_id} generated_chars={len(code)}"
        )

    write_json(a.output_json, custom_outputs)
    if a.raw_output_jsonl:
        write_jsonl(a.raw_output_jsonl, raw_rows)
    # The official evaluator's score file carries no run parameters, so without
    # this sidecar compare_results.py has nothing to gate on for the most
    # expensive path in the skill: two sides generated with different
    # release_version / max_gen_toks / bit configs would still print a delta.
    # compare_results.py finds it by the evaluator's own filename convention.
    run_meta_path = lcb_run_meta_path(a.output_json)
    write_json(
        run_meta_path,
        {
            "tag": tag,
            "task": a.dataset_task,
            "release_version": a.release_version,
            "sample_len": len(docs),
            **sample_limit_fields(a.limit),
            "max_gen_toks": a.max_gen_toks,
            "model_name": a.model_name,
            # Same rule as assemble_and_eval.py's payload: a bf16 baseline
            # builds no quant block, so these describe nothing. None =
            # "not applicable", not "different".
            "param_dir": None if a.bf16 or a.direct else a.param_dir,
            "quant_target": None if a.bf16 else quant_target,
            "bit_config": None if a.bf16 else a.bit_config,
            "quant_dtype": None if a.bf16 else a.quant_dtype,
            # Same rule as assemble_and_eval.py's payload: a string, and
            # None where no algorithm module was built.
            "algos": None if a.bf16 or a.direct else ",".join(algo_list),
            # Same rule as assemble_and_eval.py's payload: recorded, not a
            # comparability key -- the bf16 side quantizes nothing.
            "direct": a.direct,
            "is_per_tensor": None if a.bf16 else a.is_per_tensor,
            # Which decoder layers were actually quantized, derived from
            # --param_dir. Recorded so the report can quote the span from a
            # file instead of scraping the log; deliberately not a
            # comparability key -- the bf16 side has no quantized span at all,
            # so there is nothing on the other side to compare it with
            # (constraints.md #4).
            "quant_scope": quant_scope,
            "device_map": dmap,
            "bf16": a.bf16,
        },
    )
    # Deliberately NOT a `[RESULT]` line: this log only proves generation
    # finished, it holds no score. compare_results.py parses `[RESULT]` lines,
    # and tagging `sample_len` as a result made it auto-pick that count and
    # print `delta_quant_minus_bf16: 0.0` -- i.e. "no accuracy loss" -- for two
    # logs that were never scored. pass@1 comes from the official evaluator
    # output below (`*_eval.json`); feed THAT file to compare_results.py.
    print(
        f"[lcb-gen] {tag} livecodebench_custom_output: "
        f"{{'sample_len': {len(docs)}, 'output_json': {a.output_json!r}, "
        f"'raw_output_jsonl': {a.raw_output_jsonl!r}, "
        f"'run_meta': {run_meta_path!r}}}"
    )
    print("[next] score with official hidden tests:")
    print(
        "python -m lcb_runner.runner.custom_evaluator "
        f"--scenario codegeneration --release_version {a.release_version} "
        f"--custom_output_file {a.output_json}"
    )
    print(
        "[next] it writes "
        f"{a.output_json[:-5] if a.output_json.endswith('.json') else a.output_json}"
        "_codegeneration_output_eval.json; compare the two sides with: "
        f"python {os.path.join(SKILL_DIR, 'compare_results.py')} "
        "--bf16 <bf16..._eval.json> "
        "--quant <quant..._eval.json> --metric pass@1"
    )


if __name__ == "__main__":
    main()
