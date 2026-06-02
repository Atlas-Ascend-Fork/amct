"""Smoke-test a HF causal LM on NPU with chat-template inference.

Usage:
    python infer_smoke.py --model_path /path/to/model
    python infer_smoke.py --model_path /path/to/model --device npu:0 --max_new_tokens 128

Reports per-prompt latency and throughput, prints generated text.
Confirms model files are intact and NPU pipeline works end-to-end.
"""
import argparse
import time

import torch
import torch_npu  # noqa: F401
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_PROMPTS = [
    "用一句话解释什么是大模型量化。",
    "Write a Python function that returns the n-th Fibonacci number.",
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--device", default="npu:0")
    p.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--prompts", nargs="*", default=None,
                   help="Override default prompts; omit to use built-in two-shot Chinese+English set")
    p.add_argument("--no_chat_template", action="store_true",
                   help="Pass prompt directly to tokenizer (skip apply_chat_template)")
    args = p.parse_args()

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]
    prompts = args.prompts if args.prompts else DEFAULT_PROMPTS

    print(f"[load] {args.model_path}  device={args.device}  dtype={args.dtype}")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=dtype).to(args.device).eval()
    print(f"[load] done in {time.time()-t0:.1f}s")

    for i, prompt in enumerate(prompts):
        if args.no_chat_template:
            text = prompt
        else:
            messages = [{"role": "user", "content": prompt}]
            # enable_thinking only matters for Qwen3 family; harmless elsewhere if supported
            try:
                text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            except TypeError:
                text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(args.device)

        t0 = time.time()
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        dt = time.time() - t0
        gen = out[0][inputs.input_ids.shape[1]:]
        reply = tokenizer.decode(gen, skip_special_tokens=True)
        n_new = gen.shape[0]
        print(f"\n===== Prompt {i+1} =====")
        print(f"Q: {prompt}")
        print(f"A: {reply}")
        print(f"[stats] new_tokens={n_new}, time={dt:.2f}s, throughput={n_new/dt:.1f} tok/s")


if __name__ == "__main__":
    main()
