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
from eval_utils import (get_model, calibrate, test_ppl, report_coverage_memory,
                        report_actual_coverage, report_memory_full)


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
    ap.add_argument("--device_map", default=None,
                    help="多卡切分：'auto' 让 accelerate 跨可见卡切分（35B 单卡放不下时用）；"
                         "留空=单卡整体加载（8B）")
    ap.add_argument("--skip_router", action="store_true",
                    help="MoE：跳过 router（mlp.gate）量化，避免路由被 MXFP4 扰动改变 top-k")
    ap.add_argument("--increments", nargs="*", default=[],
                    choices=["scale_fitting", "hadamard", "act_reorder"],
                    help="mr_gptq 模式下开启的增量（消融用）")
    return ap.parse_args()


def main():
    args = build_args()
    dtype = getattr(torch, args.dtype)

    model, tokenizer = get_model(args.model_path, seqlen=args.seq_len,
                                 dtype=dtype, device=args.device,
                                 device_map=args.device_map)

    if args.mode != "bf16":
        import amct_pytorch as amct
        if args.mode == "mr_gptq":
            import mr_gptq_module  # import 即运行时注册 'mr_gptq'
            for inc in args.increments:
                mr_gptq_module._INCREMENTS[inc] = True
            print(f"[MR-GPTQ] increments = {mr_gptq_module._INCREMENTS}")
            cfg = dict(MXFP4_MR_GPTQ_CFG)
        else:
            cfg = dict(MXFP4_GPTQ_CFG)
        # MoE：可选跳过 router（mlp.gate）。amct skip_layers 为子串匹配，
        # 'mlp.gate' 命中 router（...mlp.gate）但不命中 expert（...mlp.experts.N.gate_proj）。
        if args.skip_router:
            cfg["skip_layers"] = set(cfg["skip_layers"]) | {"mlp.gate"}
        print(f"[cfg] skip_layers = {sorted(cfg['skip_layers'])}")
        calib = get_calib_dataset(tokenizer, n_samples=args.n_calib,
                                  block_size=args.calib_block)
        report_coverage_memory(model, skip_layers=cfg["skip_layers"])  # 量化前理论统计
        amct.quantize(model, cfg)
        calibrate(model, calib)
        report_actual_coverage(model)  # 校准后真实统计（MoE 冷 expert 会掉出来）
        # MoE 融合路由专家（amct 覆盖不到的大头）：手动 MXFP4，否则显存降不达标
        if args.mode == "mr_gptq":
            sf = mr_gptq_module._INCREMENTS["scale_fitting"]
            mr_gptq_module.quantize_fused_experts(model, scale_fitting=sf)
        report_memory_full(model)  # 全参数口径的真实显存降低（验收用）

    testenc = get_test_dataset(tokenizer)  # 保持 CPU，test_ppl 逐 batch 搬到输入卡
    ppl = test_ppl(model, testenc)
    print(f"[RESULT] model={args.model_path} mode={args.mode} "
          f"seq_len={args.seq_len} dtype={args.dtype} PPL={ppl:.4f}")


if __name__ == "__main__":
    main()
