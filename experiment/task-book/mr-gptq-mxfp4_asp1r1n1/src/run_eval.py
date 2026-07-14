# -*- coding: UTF-8 -*-
"""MR-GPTQ MXFP4 评测入口。

用法：
    export ASCEND_RT_VISIBLE_DEVICES=0
    python run_eval.py --model_path /path/to/Qwen3-8B --mode bf16
    python run_eval.py --model_path /path/to/Qwen3-8B --mode gptq_mxfp4
    python run_eval.py --model_path /path/to/Qwen3-8B --mode mr_gptq        # 待实现

mode 说明：
    bf16        原始网络 PPL 基准（不量化）
    gptq_mxfp4  MXFP4 + 仓内 GPTQ（baseline 对照）
    mr_gptq     MXFP4 + MR-GPTQ（本任务算法）

依赖以源码方式运行 amct（改动即时生效、天然进 git）：
    PYTHONPATH=<amct 仓库根> python run_eval.py ...
"""
import argparse

import torch

from data import get_test_dataset, get_calib_dataset
from eval_utils import get_model, calibrate, test_ppl


# ---- 量化配置 ----------------------------------------------------------------
# MXFP4 权重（micro-block=32，E2M1 元素 + E8M0 共享指数），weight-only。
_MXFP4_WEIGHTS = {
    "type": "mxfp4_e2m1",
    "symmetric": True,
    "strategy": "group",
    "group_size": 32,
}

# E1 baseline：MXFP4 + GPTQ（仓内已有算法）。仓里无现成常量，此处自建。
MXFP4_GPTQ_CFG = {
    "batch_num": 1,
    "quant_cfg": {"weights": dict(_MXFP4_WEIGHTS)},
    "algorithm": {"gptq"},
    "skip_layers": {"lm_head"},
}

# 本任务：MXFP4 + MR-GPTQ（注册名 'mr_gptq'，待 mr_gptq_module.py 实现并注册后启用）。
MXFP4_MR_GPTQ_CFG = {
    "batch_num": 1,
    "quant_cfg": {"weights": dict(_MXFP4_WEIGHTS)},
    "algorithm": {"mr_gptq"},
    "skip_layers": {"lm_head"},
}


def build_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--mode", required=True, choices=["bf16", "gptq_mxfp4", "mr_gptq"])
    ap.add_argument("--seq_len", type=int, default=2048,
                    help="PPL 分块长度；探索用 2048，最终交付用 4096")
    ap.add_argument("--n_calib", type=int, default=64, help="校准样本数（OOM 就调小）")
    ap.add_argument("--calib_block", type=int, default=512)
    ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"])
    ap.add_argument("--device", default="npu:0")
    return ap.parse_args()


def main():
    args = build_args()
    dtype = getattr(torch, args.dtype)

    model, tokenizer = get_model(args.model_path, seqlen=args.seq_len,
                                 dtype=dtype, device=args.device)

    if args.mode != "bf16":
        import amct_pytorch as amct
        cfg = MXFP4_GPTQ_CFG if args.mode == "gptq_mxfp4" else MXFP4_MR_GPTQ_CFG
        calib = get_calib_dataset(tokenizer, n_samples=args.n_calib,
                                  block_size=args.calib_block)
        amct.quantize(model, cfg)
        calibrate(model, calib)

    testenc = get_test_dataset(tokenizer).to(args.device)
    ppl = test_ppl(model, testenc)
    print(f"[RESULT] model={args.model_path} mode={args.mode} "
          f"seq_len={args.seq_len} dtype={args.dtype} PPL={ppl:.4f}")


if __name__ == "__main__":
    main()
