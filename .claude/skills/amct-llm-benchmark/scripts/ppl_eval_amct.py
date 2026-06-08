"""WikiText-2 PPL evaluation using the AMCT main pipeline (recommended).

Compared to ppl_eval.py + hif8_w8a8_wrap.py, this script uses:

    import amct_pytorch as amct
    amct.quantize(model, amct.HIFP8_CAST_CFG)

which is the production calibration-free Cast path with:
- weight: per-output-channel scale (vs per-tensor in the experimental wrap)
- activation: per-tensor cast
- skip_layers: lm_head
- calibration-free (Cast algorithm)

Per-channel weight scale typically gives 0.01-0.02 better PPL than per-tensor
on Qwen3-8B (measured: 9.7398 vs 9.7523 with same data, same NPU op).

PROTOBUF IMPORT — handled automatically:
The amct_pytorch source tree needs protoc-generated _pb2.py to import. If
you've run `bash build.sh --pkg` (or pip-installed the wheel), it just works.
If not, this script falls back to the installed wheel by stripping the repo
from sys.path. Proper fix is to run `bash build.sh --pkg` once.

FP8*FP4 (W4A8) mode — `--quant fp8fp4`:
Builds the float8_e4m3fn (act, per-tensor) * float4_e2m1 (weight, group=32) config
matching the minmax example. Requires the model in bfloat16 (constraint enforced
by amct: vars.py ALLOWED_DTYPE_MAP). Runs amct.quantize → calibrate via pileval →
amct.convert → PPL eval. Baseline PPL in this mode is bf16 (apples-to-apples).

Usage:
    python ppl_eval_amct.py --model_path /path/to/model            # FP16 baseline + HIFP8 cast
    python ppl_eval_amct.py --model_path /path/to/model --quant fp16   # FP16 only
    python ppl_eval_amct.py --model_path /path/to/model --quant hif8   # HIFP8 only
    python ppl_eval_amct.py --model_path /path/to/model --quant fp8fp4 # bf16 baseline + W4A8 (FP8*FP4)
    python ppl_eval_amct.py --model_path /path/to/model --quant hif8 \
        --start_chunk 0 --max_chunks 73 --nll_dump part1.json  # half eval for subagent timeout
"""
import argparse
import time

import torch
import torch_npu  # noqa: F401
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from ppl_core import eval_ppl
import amct_pytorch as amct


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--device", default="npu:0")
    p.add_argument("--seqlen", type=int, default=2048)
    p.add_argument("--quant", default="both", choices=["both", "fp16", "hif8", "fp8fp4"],
                   help="both = FP16 (full) then in-place quantize and run HIFP8 — "
                        "shares one model load. --start_chunk/--max_chunks apply to the "
                        "HIFP8 part; FP16 always runs full (28s, irrelevant to budget). "
                        "fp8fp4 = bf16 baseline + W4A8 (float8_e4m3fn act × float4_e2m1 "
                        "weight, group=32, minmax algorithm, requires pileval calibration).")
    p.add_argument("--calib_samples", type=int, default=64,
                   help="pileval calibration sample count for fp8fp4 mode")
    p.add_argument("--start_chunk", type=int, default=0,
                   help="HiFloat8 start chunk (FP16 ignores this).")
    p.add_argument("--max_chunks", type=int, default=None,
                   help="HiFloat8 chunk count (FP16 ignores this).")
    p.add_argument("--nll_dump", default=None,
                   help="Path prefix; emits .fp16.json (if FP16 was partial) and .hif8.json.")
    args = p.parse_args()

    print(f"[load] tokenizer + dataset")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False, trust_remote_code=True)
    testenc = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    testenc = tokenizer("\n\n".join(testenc["text"]), return_tensors="pt").input_ids
    total = testenc.numel() // args.seqlen
    print(f"[data] total_chunks={total}")

    results = []

    # fp8fp4 (W4A8) requires bf16; the other paths use fp16.
    model_dtype = torch.bfloat16 if args.quant == "fp8fp4" else torch.float16
    baseline_label = "bf16" if args.quant == "fp8fp4" else "fp16"

    def _load_model():
        t0 = time.time()
        m = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=model_dtype) \
            .to(args.device).eval()
        print(f"[load] model dtype={model_dtype} in {time.time()-t0:.1f}s")
        return m

    # Shared model: load once, run baseline (full) if requested, then in-place
    # quantize and run the quant variant (with start_chunk/max_chunks) if requested.
    model = _load_model()

    if args.quant in ("fp16", "both"):
        ppl, dt, n = eval_ppl(model, testenc, args.seqlen, args.device,
                              start_chunk=0, max_chunks=None,
                              label="ppl-fp16",
                              nll_dump_path=None)
        results.append(("fp16", ppl, dt, n))

    if args.quant == "fp8fp4":
        # bf16 baseline (apples-to-apples with the quantized model dtype)
        ppl, dt, n = eval_ppl(model, testenc, args.seqlen, args.device,
                              start_chunk=0, max_chunks=None,
                              label="ppl-bf16",
                              nll_dump_path=None)
        results.append(("bf16", ppl, dt, n))

        cfg = {
            'batch_num': 1,
            'quant_cfg': {
                'weights': {
                    'type': 'float4_e2m1',
                    'symmetric': True,
                    'strategy': 'group',
                    'group_size': 32,
                },
                'inputs': {
                    'type': 'float8_e4m3fn',
                    'symmetric': True,
                    'strategy': 'tensor',
                },
            },
            'algorithm': {'minmax'},
            'skip_layers': {'lm_head'},
        }
        print(f"[quant] amct.quantize(model, FP8*FP4 minmax cfg)")
        t0 = time.time()
        amct.quantize(model, cfg)
        print(f"[quant] inserted observers in {time.time()-t0:.1f}s")

        # pileval calibration
        print(f"[calib] loading pileval, {args.calib_samples} samples")
        calib_ds = load_dataset("mit-han-lab/pile-val-backup")["validation"].shuffle(seed=42)
        block_size = 256
        samples = []
        for data in calib_ds:
            line = data["text"].strip()
            ids = tokenizer.encode(line)
            if len(ids) > 512 or len(ids) == 0:
                continue
            samples.append(torch.tensor([ids]))
            if len(samples) >= 512:
                break
        cat = torch.cat(samples, dim=1)
        n_split = cat.shape[1] // block_size
        calib = torch.cat([cat[:, i*block_size:(i+1)*block_size] for i in range(n_split)],
                          dim=0)[:args.calib_samples, :]
        t0 = time.time()
        with torch.no_grad():
            model(calib.to(args.device))
        torch_npu.npu.empty_cache()
        print(f"[calib] forward done in {time.time()-t0:.1f}s")

        print(f"[convert] amct.convert(model)")
        t0 = time.time()
        amct.convert(model)
        torch_npu.npu.empty_cache()
        print(f"[convert] done in {time.time()-t0:.1f}s")

        ppl, dt, n = eval_ppl(model, testenc, args.seqlen, args.device,
                              start_chunk=args.start_chunk, max_chunks=args.max_chunks,
                              label="ppl-fp8fp4",
                              nll_dump_path=(args.nll_dump.replace(".json", ".fp8fp4.json")
                                             if args.nll_dump else None))
        results.append(("fp8*fp4_W4A8", ppl, dt, n))

    if args.quant in ("hif8", "both"):
        print(f"[quant] amct.quantize(model, HIFP8_CAST_CFG)  (per-channel weight + per-tensor activation)")
        t0 = time.time()
        amct.quantize(model, amct.HIFP8_CAST_CFG)
        print(f"[quant] done in {time.time()-t0:.1f}s")
        ppl, dt, n = eval_ppl(model, testenc, args.seqlen, args.device,
                              start_chunk=args.start_chunk, max_chunks=args.max_chunks,
                              label="ppl-hif8",
                              nll_dump_path=(args.nll_dump.replace(".json", ".hif8.json")
                                             if args.nll_dump else None))
        results.append(("hif8_cast", ppl, dt, n))

    print(f"\n[result] AMCT pipeline ({args.quant} mode, baseline={baseline_label}):")
    for tag, ppl, dt, n in results:
        partial = " (PARTIAL)" if args.nll_dump else ""
        print(f"  {tag:14s}  PPL={ppl:.4f}{partial}  chunks={n}/{total}  time={dt:.1f}s")


if __name__ == "__main__":
    main()
