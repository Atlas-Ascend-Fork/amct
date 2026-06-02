"""MMLU 5-shot accuracy evaluation on NPU, with optional W8A8 HiFloat8 quantization.

Standard lm-evaluation-harness protocol:
  prompt = "The following are multiple choice questions (with answers) about <subject>.\\n\\n"
           + 5 dev examples ("{Q}\\nA. ..\\nB. ..\\nC. ..\\nD. ..\\nAnswer: <letter>\\n\\n") * 5
           + test "{Q}\\nA. ..\\nB. ..\\nC. ..\\nD. ..\\nAnswer:"
  → forward once, take logits at last position over [' A',' B',' C',' D'] token ids, argmax = pred.

Usage:
    # FP16 baseline only:
    python mmlu_eval.py --model_path /path/to/Qwen3-0.6B
    # FP16 + HiFloat8 W8A8 in one process (saves model reload):
    python mmlu_eval.py --model_path /path/to/Qwen3-0.6B --quant both
    # Quick smoke test on 4 subjects:
    python mmlu_eval.py --model_path /path/to/Qwen3-0.6B \
        --subjects abstract_algebra,anatomy,marketing,global_facts
"""
import argparse
import json
import os
import time

import pandas as pd
import torch
import torch.nn as nn
import torch_npu  # noqa: F401
import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# Token IDs for " A", " B", " C", " D" in Qwen tokenizer (verified for Qwen2/Qwen3).
# Resolved at runtime to support other tokenizers.
LETTERS = ["A", "B", "C", "D"]


SUBCATEGORIES = {
    "abstract_algebra": "STEM", "anatomy": "Other", "astronomy": "STEM",
    "business_ethics": "Other", "clinical_knowledge": "Other",
    "college_biology": "STEM", "college_chemistry": "STEM",
    "college_computer_science": "STEM", "college_mathematics": "STEM",
    "college_medicine": "Other", "college_physics": "STEM",
    "computer_security": "STEM", "conceptual_physics": "STEM",
    "econometrics": "Social", "electrical_engineering": "STEM",
    "elementary_mathematics": "STEM", "formal_logic": "Humanities",
    "global_facts": "Other", "high_school_biology": "STEM",
    "high_school_chemistry": "STEM", "high_school_computer_science": "STEM",
    "high_school_european_history": "Humanities", "high_school_geography": "Social",
    "high_school_government_and_politics": "Social",
    "high_school_macroeconomics": "Social", "high_school_mathematics": "STEM",
    "high_school_microeconomics": "Social", "high_school_physics": "STEM",
    "high_school_psychology": "Social", "high_school_statistics": "STEM",
    "high_school_us_history": "Humanities", "high_school_world_history": "Humanities",
    "human_aging": "Other", "human_sexuality": "Social",
    "international_law": "Humanities", "jurisprudence": "Humanities",
    "logical_fallacies": "Humanities", "machine_learning": "STEM",
    "management": "Other", "marketing": "Other", "medical_genetics": "Other",
    "miscellaneous": "Other", "moral_disputes": "Humanities",
    "moral_scenarios": "Humanities", "nutrition": "Other", "philosophy": "Humanities",
    "prehistory": "Humanities", "professional_accounting": "Other",
    "professional_law": "Humanities", "professional_medicine": "Other",
    "professional_psychology": "Social", "public_relations": "Social",
    "security_studies": "Social", "sociology": "Social",
    "us_foreign_policy": "Social", "virology": "Other", "world_religions": "Humanities",
}


def format_subject(s):
    return " ".join(s.split("_"))


def format_example(question, choices, answer_letter=None):
    s = question + "\n"
    for letter, choice in zip(LETTERS, choices):
        s += f"{letter}. {choice}\n"
    s += "Answer:"
    if answer_letter is not None:
        s += f" {answer_letter}\n\n"
    return s


def build_prompt(subject, dev_rows, test_row):
    prefix = (f"The following are multiple choice questions (with answers) "
              f"about {format_subject(subject)}.\n\n")
    for _, r in dev_rows.iterrows():
        prefix += format_example(r["question"], r["choices"],
                                 LETTERS[int(r["answer"])])
    prefix += format_example(test_row["question"], test_row["choices"])
    return prefix


@torch.no_grad()
def eval_subject(model, tokenizer, dev_rows, test_rows, letter_ids, device, label):
    correct = 0
    total = len(test_rows)
    for _, r in test_rows.iterrows():
        prompt = build_prompt(r["subject"], dev_rows, r)
        ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        logits = model(ids).logits[0, -1, :]  # [vocab]
        choice_logits = logits[letter_ids]  # [4]
        pred = int(torch.argmax(choice_logits).item())
        if pred == int(r["answer"]):
            correct += 1
    return correct, total


def run_full_eval(model, tokenizer, mmlu_root, subjects, letter_ids, device, label):
    dev_all = pd.read_parquet(f"{mmlu_root}/all/dev-00000-of-00001.parquet")
    test_all = pd.read_parquet(f"{mmlu_root}/all/test-00000-of-00001.parquet")
    per_subject = {}
    cat_correct = {"STEM": 0, "Humanities": 0, "Social": 0, "Other": 0}
    cat_total = {"STEM": 0, "Humanities": 0, "Social": 0, "Other": 0}
    overall_correct, overall_total = 0, 0
    t0 = time.time()
    for subj in tqdm.tqdm(subjects, desc=f"mmlu-{label}"):
        dev_rows = dev_all[dev_all["subject"] == subj].head(5)
        test_rows = test_all[test_all["subject"] == subj]
        c, t = eval_subject(model, tokenizer, dev_rows, test_rows,
                            letter_ids, device, label)
        per_subject[subj] = {"correct": c, "total": t, "acc": c / t}
        overall_correct += c
        overall_total += t
        cat = SUBCATEGORIES[subj]
        cat_correct[cat] += c
        cat_total[cat] += t
    dt = time.time() - t0
    return {
        "label": label,
        "per_subject": per_subject,
        "categories": {k: {"correct": cat_correct[k], "total": cat_total[k],
                           "acc": cat_correct[k] / cat_total[k] if cat_total[k] else 0}
                       for k in cat_correct},
        "overall": {"correct": overall_correct, "total": overall_total,
                    "acc": overall_correct / overall_total},
        "time_s": dt,
        "questions_per_s": overall_total / dt,
    }


def print_summary(result):
    o = result["overall"]
    print(f"\n[{result['label']}] overall: {o['correct']}/{o['total']} = "
          f"{o['acc']*100:.2f}%  time={result['time_s']:.1f}s "
          f"({result['questions_per_s']:.2f} q/s)")
    for cat, v in result["categories"].items():
        if v["total"]:
            print(f"  {cat:11s} {v['correct']:>5}/{v['total']:>5} = {v['acc']*100:.2f}%")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--mmlu_root", default="/mnt/workspace/gitCode/cann/datasets/mmlu")
    p.add_argument("--device", default="npu:0")
    p.add_argument("--quant", default="none",
                   choices=["none", "hif8_w8a8", "both"],
                   help="none = FP16 only; hif8_w8a8 = quantized only; "
                        "both = FP16 then in-place quantize then eval again")
    p.add_argument("--subjects", default=None,
                   help="Comma-separated subset for smoke testing; default = all 57")
    p.add_argument("--out", default=None,
                   help="Write per-run JSON results to this path "
                        "(suffix .fp16.json / .hif8.json appended for --quant both)")
    args = p.parse_args()

    subjects = (args.subjects.split(",") if args.subjects
                else sorted(SUBCATEGORIES.keys()))
    print(f"[load] {args.model_path}  device={args.device}")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, use_fast=False, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.float16).to(args.device).eval()
    print(f"[load] done in {time.time()-t0:.1f}s")

    letter_ids = torch.tensor(
        [tokenizer.encode(f" {L}", add_special_tokens=False)[0] for L in LETTERS],
        device=args.device,
    )
    print(f"[tokens] letter ids ' A B C D' = {letter_ids.tolist()}")
    print(f"[eval] {len(subjects)} subjects")

    results = []
    if args.quant in ("none", "both"):
        r = run_full_eval(model, tokenizer, args.mmlu_root, subjects,
                          letter_ids, args.device, "fp16")
        print_summary(r)
        results.append(("fp16", r))

    if args.quant in ("hif8_w8a8", "both"):
        from hif8_w8a8_wrap import wrap_model_hif8_w8a8
        print("[quant] applying wrap_model_hif8_w8a8")
        t0 = time.time()
        n = wrap_model_hif8_w8a8(model)
        torch.npu.empty_cache()
        print(f"[quant] wrapped {n} modules in {time.time()-t0:.1f}s")
        r = run_full_eval(model, tokenizer, args.mmlu_root, subjects,
                          letter_ids, args.device, "hif8_w8a8")
        print_summary(r)
        results.append(("hif8", r))

    if args.out:
        if len(results) == 1:
            with open(args.out, "w") as f:
                json.dump(results[0][1], f, indent=2)
            print(f"[out] wrote {args.out}")
        else:
            base, ext = os.path.splitext(args.out)
            for tag, r in results:
                path = f"{base}.{tag}{ext or '.json'}"
                with open(path, "w") as f:
                    json.dump(r, f, indent=2)
                print(f"[out] wrote {path}")

    if len(results) == 2:
        fp = results[0][1]["overall"]["acc"]
        hf = results[1][1]["overall"]["acc"]
        print(f"\n[diff] FP16={fp*100:.2f}%  HiFloat8={hf*100:.2f}%  "
              f"Δ={(hf-fp)*100:+.2f}pp ({(hf-fp)/fp*100:+.2f}%)")


if __name__ == "__main__":
    main()
