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
    _load_float,
    assemble_quant_model,
    input_device,
)
from config import DEFAULT_BIT_CONFIG, local_dataset_map, load_dataset_registry  # noqa: E402
from utils import (  # noqa: E402
    check_environment,
    has_attention_quant,
    local_dataset_files,
    read_dataset_file,
    validate_lcb_docs,
    warn_trust_remote_code,
)


def _dataset_files(task="livecodebench"):
    return local_dataset_files(task, load_dataset_registry(), local_dataset_map())


def load_lcb_docs(task, limit):
    frames = []
    for fp in _dataset_files(task):
        frames.append(read_dataset_file(fp))
    df = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
    if limit and limit > 0:
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
    ap.add_argument("--model_name", default="qwen3")
    ap.add_argument("--param_dir", default="")
    ap.add_argument("--quant_target", default="")
    ap.add_argument(
        "--algos",
        default=",".join(DEFAULT_ALGOS),
        help=(
            "comma-separated amct algorithm keys built into every quant block; "
            "must match the algorithms the ptq params were trained with"
        ),
    )
    ap.add_argument("--bit_config", default=DEFAULT_BIT_CONFIG)
    ap.add_argument(
        "--output_dir", default="/tmp/qbe_out", help="amct temporary output dir"
    )
    ap.add_argument("--seq_len", type=int, default=2048)
    ap.add_argument("--device_map", default="")
    ap.add_argument(
        "--limit",
        type=int,
        default=0,
        help="debug only; official custom_evaluator requires the full selected release",
    )
    ap.add_argument("--max_gen_toks", type=int, default=1024)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument(
        "--allow_attn_quant",
        action="store_true",
        help="explicitly allow attn-linear/attn-cache quant targets",
    )
    ap.add_argument(
        "--dataset_task",
        choices=["livecodebench", "livecodebench_release_v6"],
        default="livecodebench_release_v6",
    )
    ap.add_argument(
        "--release_version",
        default="release_v6",
        help="lcb_runner release_version to use when scoring this output",
    )
    ap.add_argument(
        "--output_json",
        default="livecodebench_custom_outputs.json",
        help="official lcb_runner custom_output_file JSON",
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
            "directory you trust (constraints.md #12)."
        ),
    )
    a = ap.parse_args()

    check_environment()
    from transformers import AutoTokenizer

    dmap = a.device_map or None
    warn_trust_remote_code(a.model, a.trust_remote_code)
    tok = AutoTokenizer.from_pretrained(
        a.model, use_fast=False, trust_remote_code=a.trust_remote_code
    )
    quant_target = a.quant_target or ""
    if a.bf16:
        tag = "bf16"
        print(f"[bf16] loading original model (no quant, device_map={dmap})...")
        model = _load_float(a.model, dmap, a.trust_remote_code)
    else:
        tag = "quant"
        if not a.param_dir:
            sys.exit("--param_dir required unless --bf16")
        model, quant_target = assemble_quant_model(
            a.model,
            a.model_name,
            a.param_dir,
            a.bit_config,
            a.seq_len,
            device_map=dmap,
            quant_target=(a.quant_target or None),
            output_dir=a.output_dir,
            allow_attn_quant=a.allow_attn_quant,
            algos=a.algos,
            trust_remote_code=a.trust_remote_code,
        )

    docs = load_lcb_docs(a.dataset_task, a.limit)
    validate_lcb_docs(docs)
    custom_outputs = []
    raw_rows = []
    if a.limit and a.limit > 0:
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
    # Deliberately NOT a `[RESULT]` line: this log only proves generation
    # finished, it holds no score. compare_results.py parses `[RESULT]` lines,
    # and tagging `sample_len` as a result made it auto-pick that count and
    # print `delta_quant_minus_bf16: 0.0` -- i.e. "no accuracy loss" -- for two
    # logs that were never scored. pass@1 comes from the official evaluator
    # output below (`*_eval.json`); feed THAT file to compare_results.py.
    print(
        f"[lcb-gen] {tag} livecodebench_custom_output: "
        f"{{'sample_len': {len(docs)}, 'output_json': {a.output_json!r}, "
        f"'raw_output_jsonl': {a.raw_output_jsonl!r}}}"
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
        "python scripts/compare_results.py --bf16 <bf16..._eval.json> "
        "--quant <quant..._eval.json> --metric pass@1"
    )


if __name__ == "__main__":
    main()
