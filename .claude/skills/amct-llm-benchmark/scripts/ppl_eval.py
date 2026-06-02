"""WikiText-2 PPL evaluation on NPU, with optional W8A8 HiFloat8 quantization.

Usage:
    # FP16 baseline:
    python ppl_eval.py --model_path /path/to/model

    # HiFloat8 W8A8 Cast (uses amct_ops NPU op):
    python ppl_eval.py --model_path /path/to/model --quant hif8_w8a8

    # Quick probe on N chunks (debug/timing):
    python ppl_eval.py --model_path /path/to/model --quant hif8_w8a8 --max_chunks 4

Standard protocol (matches amct_pytorch/experimental/hifloat8/utils.py:
    - dataset: wikitext, wikitext-2-raw-v1, test split
    - texts joined by "\\n\\n", tokenized once
    - seqlen=2048, non-overlapping chunks, ~146 chunks total

Quantization variants are layered: import a wrap function from hif8_w8a8_wrap
and pass --quant <name>. To add INT8/W4A16/AWQ/GPTQ, extend QUANT_REGISTRY.
"""
import argparse
import time

import torch
import torch_npu  # noqa: F401
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from ppl_core import eval_ppl  # canonical PPL protocol — single source of truth


def get_quant_wrapper(name):
    if name == "none":
        return None
    if name == "hif8_w8a8":
        from hif8_w8a8_wrap import wrap_model_hif8_w8a8
        return wrap_model_hif8_w8a8
    raise ValueError(f"Unknown --quant value: {name!r}. "
                     f"Built-in: 'none', 'hif8_w8a8'. "
                     f"Add INT8/W4A16/AWQ/GPTQ by extending get_quant_wrapper.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--device", default="npu:0")
    p.add_argument("--seqlen", type=int, default=2048)
    p.add_argument("--dataset", default="wikitext", help="HF datasets name")
    p.add_argument("--config", default="wikitext-2-raw-v1", help="HF dataset config")
    p.add_argument("--split", default="test")
    p.add_argument("--quant", default="none",
                   help="none | hif8_w8a8 (extend get_quant_wrapper for more)")
    p.add_argument("--start_chunk", type=int, default=0,
                   help="First chunk to evaluate (for batched runs)")
    p.add_argument("--max_chunks", type=int, default=None,
                   help="Limit eval to N chunks starting at --start_chunk. "
                        "Default = run to end. Use this + --nll_dump to split "
                        "a long eval across multiple invocations under 600s.")
    p.add_argument("--nll_dump", default=None,
                   help="Path to dump per-chunk NLLs (JSON). When set, the final "
                        "PPL printed here is *partial* — merge with scripts/merge_ppl.py.")
    args = p.parse_args()

    print(f"[load] {args.model_path}  device={args.device}  seqlen={args.seqlen}")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.float16) \
        .to(args.device).eval()
    print(f"[load] done in {time.time()-t0:.1f}s")

    wrap = get_quant_wrapper(args.quant)
    if wrap is not None:
        print(f"[quant] applying wrapper: {args.quant}")
        t0 = time.time()
        n = wrap(model)
        torch.npu.empty_cache()
        print(f"[quant] wrapped {n} modules in {time.time()-t0:.1f}s")

    print(f"[data] {args.dataset}/{args.config}/{args.split}")
    testenc = load_dataset(args.dataset, args.config, split=args.split)
    testenc = tokenizer("\n\n".join(testenc["text"]), return_tensors="pt").input_ids
    total = testenc.numel() // args.seqlen
    will_eval = (min(total, args.start_chunk + args.max_chunks) - args.start_chunk
                 if args.max_chunks else total - args.start_chunk)
    print(f"[data] total_chunks={total}, eval chunks=[{args.start_chunk}, "
          f"{args.start_chunk + will_eval}) → {will_eval} chunks")

    ppl, dt, n_used = eval_ppl(model, testenc, args.seqlen, args.device,
                               start_chunk=args.start_chunk,
                               max_chunks=args.max_chunks,
                               label=f"ppl-{args.quant}",
                               nll_dump_path=args.nll_dump)
    suffix = " (PARTIAL — use merge_ppl.py to get full PPL)" if args.nll_dump else ""
    print(f"\n[result] quant={args.quant}  PPL={ppl:.4f}{suffix}  chunks={n_used}/{total}")
    print(f"[result] time={dt:.1f}s ({dt/n_used:.2f}s/chunk)")


if __name__ == "__main__":
    main()
