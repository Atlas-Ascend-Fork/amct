"""WikiText-2 PPL evaluation for Qwen3 with FlatQuant.

Modes (--mode):
- w4a4: int4 weight (channel) x int4 act (token); calls amct.convert -> NPU W4A4.
- w4a8: int4 weight (channel) x int8 act (token); fake-quant only.
        amct's flatquant NPU deploy op (dynamic_w4a4) only supports W4A4,
        so amct.convert is skipped and PPL is measured on the pre-convert
        fake-quant model. The 'int8 int4' combination is not in amct's
        whitelist by default; this script monkey-patches the whitelist in
        memory before quantize. No wheel mutation.
"""
import argparse
import os
import random
import sys
import time

import datasets
import torch
import torch_npu  # noqa: F401
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

# Import canonical PPL impl BEFORE the protobuf-fallback sys.path strip below —
# this scripts dir path contains 'gitCode/cann/amct' and would otherwise be removed.
from ppl_core import eval_ppl  # single source of truth

# Protobuf fallback identical to ppl_eval_amct.py.
try:
    import amct_pytorch as amct
except ImportError as _e:
    print(f"[amct] direct import failed ({str(_e)[:120]}); falling back to wheel")
    sys.path = [p for p in sys.path if "gitCode/cann/amct" not in p and p != ""]
    os.chdir("/tmp")
    for _m in list(sys.modules):
        if _m.startswith("amct_pytorch"):
            del sys.modules[_m]
    import amct_pytorch as amct  # noqa: E402

from amct_pytorch.experimental.flatquant.flat_quant_module.train_utils import cali_flat_quant
from amct_pytorch.experimental.flatquant.flat_quant_module.flat_utils import (
    save_flat_matrices, load_flat_matrices,
)


def patch_amct_whitelist_for_w4a8():
    """Add 'int8 int4' to amct's combination whitelist (in-memory only).

    amct rejects unknown act-weight pairs in config_parser.check_config; flatquant's
    bit-width parsing (default_model_utils.flat_config_parser) already handles arbitrary
    int*-int* combinations via str.replace('int', ''), so the only blocker is the
    whitelist. We intentionally don't touch the wheel — patches reset on process exit.
    """
    from amct_pytorch.utils import vars as _v
    combo = 'int8 int4'
    _v.ALGORITHM_SUPPORTED_QUANT_TYPE_COMB.setdefault(combo, ['flatquant'])
    if 'flatquant' not in _v.ALGORITHM_SUPPORTED_QUANT_TYPE_COMB[combo]:
        _v.ALGORITHM_SUPPORTED_QUANT_TYPE_COMB[combo].append('flatquant')
    _v.ALLOWED_WEIGHT_DTYPES.setdefault(combo, [torch.bfloat16, torch.float16])
    for lst in (_v.WTS_PER_CHANNEL_SUPPORT_COMBINATION,
                _v.ACT_PER_TOKEN_SUPPORT_COMBINATION):
        if combo not in lst:
            lst.append(combo)


def make_cfg(mode: str):
    """W4A4: int4/token act, int4/channel weight, deploy via amct.convert.
    W4A8: int8/token act, int4/channel weight, fake-quant only (no convert).
    """
    a_type = 'int8' if mode == 'w4a8' else 'int4'
    return {
        'quant_cfg': {
            'inputs':  {'type': a_type, 'symmetric': True, 'strategy': 'token'},
            'weights': {'type': 'int4', 'symmetric': True, 'strategy': 'channel'},
        },
        'algorithm': {
            'flatquant': {
                'use_kcache_quant': False, 'k_bits': 16, 'k_sym': False,
                'use_vcache_quant': False, 'v_bits': 16, 'v_sym': False,
                'use_o_quant': False,
                'use_down_quant': False,
                'lac': True, 'lwc': True, 'diag_alpha': 0.8,
                'epochs': 15, 'cali_bsz': 4, 'flat_lr': 3e-3,
                'cali_trans': True,
            },
        },
        'skip_layers': {'lm_head'},
    }


def seed_everything(seed=0):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.npu.manual_seed_all(seed)


def get_wikitext2_train(tokenizer, nsamples, seqlen, seed=0):
    train = datasets.load_dataset('wikitext', 'wikitext-2-raw-v1')['train']
    train = train.filter(lambda x: len(x) > 0)
    train = train.map(lambda x: {'text': x['text'].strip()})
    enc = tokenizer("\n\n".join(train['text']), return_tensors='pt')
    rng = random.Random(seed)
    loader = []
    for _ in range(nsamples):
        i = rng.randint(0, enc.input_ids.shape[1] - seqlen - 1)
        inp = enc.input_ids[:, i:i + seqlen]
        tar = inp.clone()
        tar[:, :-1] = -100
        loader.append((inp, tar))
    return loader


def get_wikitext2_test_ids(tokenizer):
    test = datasets.load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    return tokenizer("\n\n".join(test['text']), return_tensors='pt').input_ids



def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--device", default="npu:0")
    p.add_argument("--mode", default="w4a4", choices=["w4a4", "w4a8"],
                   help="w4a4: deploy via amct.convert (NPU dynamic_w4a4); "
                        "w4a8: int4w x int8a fake-quant only (no NPU deploy op)")
    p.add_argument("--seqlen", type=int, default=2048)
    p.add_argument("--nsamples", type=int, default=128)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--cali_bsz", type=int, default=4)
    p.add_argument("--flat_lr", type=float, default=3e-3)
    p.add_argument("--max_chunks", type=int, default=None,
                   help="PPL chunk cap (debug/smoke only — never report partial)")
    p.add_argument("--save_matrix", default=None)
    p.add_argument("--load_matrix", default=None,
                   help="Skip calibration, load flat_matrices.pth instead")
    p.add_argument("--skip_baseline", action="store_true")
    args = p.parse_args()

    if args.mode == "w4a8":
        patch_amct_whitelist_for_w4a8()
        print("[patch] amct whitelist extended with 'int8 int4'")

    seed_everything()
    print("[load] tokenizer + wikitext2 test")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, use_fast=False, trust_remote_code=True)
    test_ids = get_wikitext2_test_ids(tokenizer)
    total = test_ids.numel() // args.seqlen
    print(f"[data] total_chunks={total}")

    print("[load] model bf16 + eager attention")
    config = AutoConfig.from_pretrained(args.model_path, attn_implementation='eager')
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, config=config,
        low_cpu_mem_usage=True)
    model.seqlen = args.seqlen
    model.to(args.device).eval()
    print(f"[load] model in {time.time() - t0:.1f}s")

    results = []
    if not args.skip_baseline:
        ppl, dt, n = eval_ppl(model, test_ids, args.seqlen, args.device,
                              label="ppl-bf16", max_chunks=args.max_chunks)
        results.append(("bf16", ppl, dt, n))

    cfg = make_cfg(args.mode)
    cfg['algorithm']['flatquant']['epochs'] = args.epochs
    cfg['algorithm']['flatquant']['cali_bsz'] = args.cali_bsz
    cfg['algorithm']['flatquant']['flat_lr'] = args.flat_lr
    print(f"[quant] amct.quantize(model, mode={args.mode}) "
          f"epochs={args.epochs} bsz={args.cali_bsz} lr={args.flat_lr} "
          f"nsamples={args.nsamples}")
    t0 = time.time()
    amct.quantize(model, cfg)
    print(f"[quant] inserted modules in {time.time() - t0:.1f}s")

    if args.load_matrix:
        print(f"[cali] load_flat_matrices from {args.load_matrix}")
        model = load_flat_matrices(model, args.load_matrix)
    else:
        print(f"[cali] build wikitext2 train loader, nsamples={args.nsamples}")
        calib = get_wikitext2_train(tokenizer, args.nsamples, args.seqlen)
        t0 = time.time()
        cali_flat_quant(model, calib, args.device)
        torch_npu.npu.empty_cache()
        print(f"[cali] done in {time.time() - t0:.1f}s")
        if args.save_matrix:
            os.makedirs(os.path.dirname(args.save_matrix) or ".", exist_ok=True)
            save_flat_matrices(model, args.save_matrix)
            print(f"[cali] saved matrices -> {args.save_matrix}")

    model.to(args.device)
    if args.mode == "w4a4":
        print("[convert] amct.convert(model)")
        t0 = time.time()
        amct.convert(model)
        torch_npu.npu.empty_cache()
        print(f"[convert] done in {time.time() - t0:.1f}s")
        quant_label = "flatquant_W4A4"
        eval_label = "ppl-w4a4"
    else:
        print("[convert] skipped (W4A8 fake-quant; no NPU dynamic_w4a8 op)")
        quant_label = "flatquant_W4A8_fakequant"
        eval_label = "ppl-w4a8"

    ppl, dt, n = eval_ppl(model, test_ids, args.seqlen, args.device,
                          label=eval_label, max_chunks=args.max_chunks)
    results.append((quant_label, ppl, dt, n))

    combo = "INT4*INT4" if args.mode == "w4a4" else "INT4w x INT8a (fake-quant)"
    print(f"\n[result] FlatQuant {combo} (channel weight + token activation):")
    for tag, ppl, dt, n in results:
        partial = " (PARTIAL)" if args.max_chunks else ""
        print(f"  {tag:30s}  PPL={ppl:.4f}{partial}  chunks={n}/{total}  time={dt:.1f}s")


if __name__ == "__main__":
    main()
