"""MATH-500 generation-based accuracy evaluation on NPU, with optional HiFloat8 W8A8.

Protocol:
  - Dataset: HuggingFaceH4/MATH-500 (test.jsonl, 500 problems, 7 subjects, 5 levels)
  - Prompt: official template from eval.yaml — "Solve...The last line of your
    response should be of the following format: 'Therefore, the final answer is:
    $\\boxed{ANSWER}$. I hope it is correct'..."
  - Apply Qwen chat template with enable_thinking=True (Qwen3 default)
  - Sampling: temp=0.6, top_p=0.95, top_k=20 (Qwen3 official thinking-mode params)
  - Max new tokens: 8192 (thinking mode generates long CoT)
  - Scorer: math_verify (sympy-based equivalence, lm-eval-harness MATH-500 standard)

Usage:
    # Smoke test 5 problems FP16:
    python math500_eval.py --model_path /path/to/Qwen3-0.6B --max_problems 5
    # Full FP16 + HiFloat8 W8A8 in one process:
    python math500_eval.py --model_path /path/to/Qwen3-0.6B --quant both \
        --out /tmp/math500.json
"""
import argparse
import json
import os
import time
from collections import defaultdict

import torch
import torch_npu  # noqa: F401
import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPT_TEMPLATE = (
    "Solve the following math problem efficiently and clearly.  The last line "
    "of your response should be of the following format: 'Therefore, the final "
    "answer is: $\\boxed{ANSWER}$. I hope it is correct' (without quotes) "
    "where ANSWER is just the final number or expression that solves the "
    "problem. Think step by step before answering.\n\n{problem}"
)


def build_chat_prompt(tokenizer, problem, enable_thinking=True):
    # Plain replace — str.format would treat the literal "{ANSWER}" inside the
    # template as a missing placeholder and raise KeyError.
    user_msg = PROMPT_TEMPLATE.replace("{problem}", problem)
    msgs = [{"role": "user", "content": user_msg}]
    try:
        return tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
            enable_thinking=enable_thinking)
    except TypeError:
        return tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True)


def score_with_math_verify(pred_text, gold_answer):
    """Returns (correct: bool, extracted: str). Uses math_verify sympy-based
    equivalence, the lm-eval-harness MATH-500 default scorer."""
    from math_verify import parse, verify
    try:
        gold = parse(f"${gold_answer}$")
    except Exception:
        gold = parse(gold_answer)
    try:
        pred = parse(pred_text)
    except Exception:
        return False, ""
    try:
        return bool(verify(gold, pred)), str(pred)
    except Exception:
        return False, str(pred)


@torch.no_grad()
def eval_one(model, tokenizer, problem, gold, device,
             max_new_tokens, temperature, top_p, top_k):
    prompt = build_chat_prompt(tokenizer, problem, enable_thinking=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    in_len = inputs.input_ids.shape[-1]
    do_sample = temperature is not None and temperature > 0
    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        pad_token_id=tokenizer.eos_token_id,
    )
    if do_sample:
        gen_kwargs.update(temperature=temperature, top_p=top_p, top_k=top_k)
    out = model.generate(**inputs, **gen_kwargs)
    new_tokens = out[0, in_len:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    correct, extracted = score_with_math_verify(text, gold)
    return correct, extracted, text, int(new_tokens.numel())


def run_full_eval(model, tokenizer, problems, device, label,
                  max_new_tokens, temperature, top_p, top_k):
    n_correct, n_total = 0, 0
    by_subject = defaultdict(lambda: [0, 0])
    by_level = defaultdict(lambda: [0, 0])
    per_problem = []
    total_new_tokens = 0
    t0 = time.time()
    pbar = tqdm.tqdm(problems, desc=f"math500-{label}")
    for r in pbar:
        ok, ext, full_text, n_new = eval_one(
            model, tokenizer, r["problem"], r["answer"], device,
            max_new_tokens, temperature, top_p, top_k)
        n_correct += int(ok)
        n_total += 1
        total_new_tokens += n_new
        s = r["subject"]
        lv = int(r["level"])
        by_subject[s][0] += int(ok); by_subject[s][1] += 1
        by_level[lv][0] += int(ok); by_level[lv][1] += 1
        per_problem.append({
            "unique_id": r.get("unique_id"),
            "subject": s, "level": lv,
            "gold": r["answer"], "extracted": ext,
            "correct": bool(ok), "new_tokens": n_new,
        })
        elapsed = time.time() - t0
        pbar.set_postfix(
            acc=f"{n_correct}/{n_total}={n_correct/n_total*100:.1f}%",
            tps=f"{total_new_tokens/elapsed:.1f}",
        )
    dt = time.time() - t0
    return {
        "label": label,
        "overall": {"correct": n_correct, "total": n_total,
                    "acc": n_correct / n_total},
        "by_subject": {k: {"correct": v[0], "total": v[1],
                           "acc": v[0] / v[1]} for k, v in by_subject.items()},
        "by_level": {k: {"correct": v[0], "total": v[1],
                         "acc": v[0] / v[1]} for k, v in sorted(by_level.items())},
        "time_s": dt,
        "tokens_total": total_new_tokens,
        "tokens_per_s": total_new_tokens / dt,
        "avg_new_tokens": total_new_tokens / n_total,
        "per_problem": per_problem,
    }


def print_summary(r):
    o = r["overall"]
    print(f"\n[{r['label']}] overall: {o['correct']}/{o['total']} = {o['acc']*100:.2f}%")
    print(f"  time={r['time_s']:.1f}s  tokens={r['tokens_total']}  "
          f"throughput={r['tokens_per_s']:.1f} tok/s  "
          f"avg_gen_len={r['avg_new_tokens']:.0f}")
    print("  by subject:")
    for k, v in sorted(r["by_subject"].items()):
        print(f"    {k:25s} {v['correct']:>3}/{v['total']:>3} = {v['acc']*100:.2f}%")
    print("  by level:")
    for k, v in r["by_level"].items():
        print(f"    L{k} {v['correct']:>3}/{v['total']:>3} = {v['acc']*100:.2f}%")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--data_path",
                   default=os.environ.get(
                       "MATH500_DATA",
                       "/mnt/workspace/gitCode/cann/datasets/math500/test.jsonl"))
    p.add_argument("--device", default="npu:0")
    p.add_argument("--quant", default="none",
                   choices=["none", "hif8_w8a8", "both"])
    p.add_argument("--max_problems", type=int, default=None,
                   help="Smoke test: only first N problems")
    p.add_argument("--stratified_sample", type=int, default=None,
                   help="Stratified random sample of N problems by "
                        "(subject, level), seeded by --seed")
    p.add_argument("--max_new_tokens", type=int, default=8192)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--top_k", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default=None,
                   help="Write per-run JSON results "
                        "(suffix .fp16.json / .hif8.json appended for --quant both)")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    torch.npu.manual_seed_all(args.seed)

    print(f"[load] {args.model_path}  device={args.device}")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, use_fast=False, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.float16).to(args.device).eval()
    print(f"[load] done in {time.time()-t0:.1f}s")

    with open(args.data_path) as f:
        problems = [json.loads(l) for l in f]
    if args.stratified_sample:
        import random
        rng = random.Random(args.seed)
        buckets = defaultdict(list)
        for r in problems:
            buckets[(r["subject"], r["level"])].append(r)
        n_total = len(problems)
        target = args.stratified_sample
        sampled = []
        for k, items in sorted(buckets.items()):
            n_cell = max(1, round(len(items) * target / n_total))
            sampled.extend(rng.sample(items, min(n_cell, len(items))))
        problems = sampled[: target] if len(sampled) > target else sampled
        rng.shuffle(problems)
    if args.max_problems:
        problems = problems[: args.max_problems]
    print(f"[data] {len(problems)} problems  thinking=True  "
          f"sample(t={args.temperature}, top_p={args.top_p}, top_k={args.top_k})  "
          f"max_new_tokens={args.max_new_tokens}")

    results = []
    if args.quant in ("none", "both"):
        r = run_full_eval(model, tokenizer, problems, args.device, "fp16",
                          args.max_new_tokens, args.temperature,
                          args.top_p, args.top_k)
        print_summary(r)
        results.append(("fp16", r))

    if args.quant in ("hif8_w8a8", "both"):
        from hif8_w8a8_wrap import wrap_model_hif8_w8a8
        print("[quant] applying wrap_model_hif8_w8a8")
        t0 = time.time()
        n = wrap_model_hif8_w8a8(model)
        torch.npu.empty_cache()
        print(f"[quant] wrapped {n} modules in {time.time()-t0:.1f}s")
        r = run_full_eval(model, tokenizer, problems, args.device, "hif8_w8a8",
                          args.max_new_tokens, args.temperature,
                          args.top_p, args.top_k)
        print_summary(r)
        results.append(("hif8", r))

    if args.out:
        if len(results) == 1:
            with open(args.out, "w") as f:
                json.dump(results[0][1], f, indent=2, ensure_ascii=False)
            print(f"[out] wrote {args.out}")
        else:
            base, ext = os.path.splitext(args.out)
            for tag, r in results:
                path = f"{base}.{tag}{ext or '.json'}"
                with open(path, "w") as f:
                    json.dump(r, f, indent=2, ensure_ascii=False)
                print(f"[out] wrote {path}")

    if len(results) == 2:
        fp = results[0][1]["overall"]["acc"]
        hf = results[1][1]["overall"]["acc"]
        print(f"\n[diff] FP16={fp*100:.2f}%  HiFloat8={hf*100:.2f}%  "
              f"Δ={(hf-fp)*100:+.2f}pp")


if __name__ == "__main__":
    main()
