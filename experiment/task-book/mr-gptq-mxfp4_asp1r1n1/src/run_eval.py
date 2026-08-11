# -*- coding: UTF-8 -*-
"""MR-GPTQ MXFP4 评测入口。

mode 说明：
    bf16        原始网络 PPL 基准（不量化）
    gptq_mxfp4  MXFP4 + 仓内 GPTQ（baseline 对照）
    mr_gptq     MXFP4 + MR-GPTQ（本任务算法）

用法（详见 README.md）：
    export ASCEND_RT_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1

    # bf16 基准
    python run_eval.py --model_path <Qwen3-8B> --mode bf16 --seq_len 4096

    # MR-GPTQ 主线（W4A16）
    python run_eval.py --model_path <Qwen3-8B> --mode mr_gptq --seq_len 4096         --increments hadamard scale_fitting

    # MR-GPTQ W4A4 达标配置（delta 0.3897）
    python run_eval.py --model_path <Qwen3-8B> --mode mr_gptq --seq_len 4096         --hadamard_k 4096 --hadamard_full --perc_damp 0.2         --act_protect down_proj o_proj         --increments hadamard scale_fitting act_quant act_scale_fit joint_comp

MR-GPTQ 由 `import mr_gptq_module` 运行时注册，**零改 amct 源码**；量化本身仍走
amct 的 `amct.quantize()` 与 `GPTQuant` 状态机。amct 按已安装包使用即可，无需设 PYTHONPATH。
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

# 本任务：MXFP4 + MR-GPTQ（注册名 'mr_gptq'，由 mr_gptq_module 于 import 时注册）。
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
    ap.add_argument("--n_calib", type=int, default=64,
                    help="校准语料【条数】（不是块数！）。pile-val 每条 ≤--max_line_tokens，"
                         "取 n_calib 条拼接后按 --calib_block 切块 → 总 token≈n_calib×平均条长。"
                         "⚠️调大 --calib_block 而不同步调大本参数会【减少】总 token 数")
    ap.add_argument("--calib_block", type=int, default=512,
                    help="校准切块长度。注意单条语料上限受 --max_line_tokens 约束，"
                         "块长超过它时块内是多条不相关短文拼接，不等于真实长上下文")
    ap.add_argument("--max_line_tokens", type=int, default=512,
                    help="校准语料单条 token 上限，超长整条丢弃。要真正的长上下文校准"
                         "（与 seq_len=4096 评测对齐）需同时调大本参数与 --calib_block")
    ap.add_argument("--calib_seed", type=int, default=42,
                    help="校准语料的抽样种子。换种子=换一批 pile-val 样本→Hessian 变→结果会漂。"
                         "用于量化「结果对校准集选取的敏感性」（评测本身确定性，无随机性）")
    ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"])
    ap.add_argument("--device", default="npu:0")
    ap.add_argument("--device_map", default=None,
                    help="多卡切分：'auto' 让 accelerate 跨可见卡切分（35B 单卡放不下时用）；"
                         "留空=单卡整体加载（8B）")
    ap.add_argument("--max_eval_samples", type=int, default=0,
                    help="仅评测前 N 块（0=全量）；消融/快速验证省时用")
    ap.add_argument("--act_protect", nargs="*", default=[],
                    help="W4A4 混合精度：层名含这些子串的层保激活高精度（如 down_proj）；空=全量 W4A4")
    ap.add_argument("--hadamard_k", type=int, default=128,
                    help="Hadamard 微旋转块大小（须为2的幂，且能整除各层 in_features）。论文默认128；"
                         "fake-quant 无在线开销，可调大(512/1024/4096)测更强旋转能否进一步压 W4A4 误差")
    ap.add_argument("--hadamard_full", action="store_true",
                    help="每层旋转块取满 in_features（全宽旋转），不可构造时退回 --hadamard_k。"
                         "Qwen3-8B 上唯一受影响的是 down_proj(in=12288)——它不是 2 的幂，"
                         "k=4096 下只能切成 3 个对角块，是全网唯一没拿到全宽旋转的层。"
                         "12288=12×1024 走 Paley H12 ⊗ Sylvester H1024（3 阶 Hadamard 不存在）")
    ap.add_argument("--hadamard_random", action="store_true",
                    help="randomized Hadamard：旋转前乘随机 ±1 对角阵（QuaRot/QuIP# 标准做法）。"
                         "归一化 Sylvester 首行全 1 → (xH)[0]=√k·mean(x)，而 RMSNorm 输出不居中，"
                         "于是旋转反而在坐标 0 制造离群值、顶高第 0 个 32 块的倍率。"
                         "乘 D 打散 DC；D²=I 故数学无损")
    ap.add_argument("--skip_router", action="store_true",
                    help="MoE：跳过 router（mlp.gate）量化，避免路由被 MXFP4 扰动改变 top-k")
    ap.add_argument("--increments", nargs="*", default=[],
                    choices=["scale_fitting", "hadamard", "act_quant",
                             "act_scale_fit", "joint_comp", "act_clip", "weight_clip",
                             "act_fit_rowwise"],
                    help="mr_gptq 增量；act_quant=W4A4；act_scale_fit/joint_comp/act_clip/"
                         "weight_clip/act_fit_rowwise=W4A4 优化①②③④⑤"
                         "（act_clip 实测为负结果，保留作消融）")
    ap.add_argument("--perc_damp", type=float, default=None,
                    help="GPTQ Hessian 阻尼系数（默认用 amct 内置值）；影响 Cholesky 稳定性与补偿强度")
    ap.add_argument("--calib_chunk", type=int, default=0,
                    help="校准前向的分块块数（0=不分块）。校准 token 一多，Hessian 累积的 [cin,N] fp32 "
                         "转置副本、以及补偿后 fake_quant_forward 里旋转/激活量化的 fp32 中间量都会 "
                         "撑爆显存。分块只降峰值、不改 Hessian 数学（父类 running-average）。"
                         "放大 --n_calib 时配它使用，建议 32")
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
            mr_gptq_module._ACT_PROTECT = list(args.act_protect)
            mr_gptq_module._HADAMARD_K = args.hadamard_k
            mr_gptq_module._HADAMARD_FULL = args.hadamard_full
            mr_gptq_module._HADAMARD_RANDOM = args.hadamard_random
            if args.perc_damp is not None:
                mr_gptq_module._PERC_DAMP = args.perc_damp
                print(f"[MR-GPTQ] GPTQ 阻尼系数 perc_damp = {args.perc_damp}")
            if args.calib_chunk:
                mr_gptq_module._CALIB_CHUNK = args.calib_chunk
                print(f"[MR-GPTQ] 校准前向分块 = 每 {args.calib_chunk} 块"
                      f"（与整块数学等价，仅降峰值显存以放大校准集）")
            print(f"[MR-GPTQ] increments = {mr_gptq_module._INCREMENTS}")
            if args.hadamard_k != 128:
                print(f"[MR-GPTQ] Hadamard 块大小 = {args.hadamard_k}（消融：>128 测更强旋转压离群值）")
            if args.hadamard_full:
                print("[MR-GPTQ] 全宽旋转：各层旋转块取满 in_features"
                      "（8B 上即 down_proj 12288 = Paley H12 ⊗ H1024）")
            if args.hadamard_random:
                print("[MR-GPTQ] randomized Hadamard：旋转前乘随机 ±1 对角阵"
                      "（消除 Sylvester 首行把 DC 堆到坐标 0 的副作用）")
            if args.act_protect:
                print(f"[MR-GPTQ] W4A4 混合精度：以下层名子串的层保激活高精度 = {args.act_protect}")
            inc = mr_gptq_module._INCREMENTS
            if inc["act_quant"]:
                mode_str = "W4A4（激活也 MXFP4）" if inc["hadamard"] else \
                    "W4A4 但【未开 hadamard】——激活离群值未压制，精度会崩（仅作消融对照）"
                print(f"[MR-GPTQ] 激活量化开启：{mode_str}")
            cfg = dict(MXFP4_MR_GPTQ_CFG)
        else:
            cfg = dict(MXFP4_GPTQ_CFG)
        # MoE：可选跳过 router（mlp.gate）。amct skip_layers 为子串匹配，
        # 'mlp.gate' 命中 router（...mlp.gate）但不命中 expert（...mlp.experts.N.gate_proj）。
        if args.skip_router:
            cfg["skip_layers"] = set(cfg["skip_layers"]) | {"mlp.gate"}
        print(f"[cfg] skip_layers = {sorted(cfg['skip_layers'])}")
        calib = get_calib_dataset(tokenizer, n_samples=args.n_calib,
                                  block_size=args.calib_block,
                                  max_line_tokens=args.max_line_tokens,
                                  seed=args.calib_seed)
        if args.calib_seed != 42:
            print(f"[calib] seed = {args.calib_seed}（非默认；用于校准集敏感性测试）")
        report_coverage_memory(model, skip_layers=cfg["skip_layers"])  # 量化前理论统计
        amct.quantize(model, cfg)
        calibrate(model, calib)
        report_actual_coverage(model)  # 校准后真实统计（MoE 冷 expert 会掉出来）
        # MoE 融合路由专家（amct 覆盖不到的大头）：手动 MXFP4，否则显存降不达标
        if args.mode == "mr_gptq":
            if mr_gptq_module._INCREMENTS["act_quant"]:
                mr_gptq_module.report_act_coverage(model)  # W4A4 验收口径：真 4-bit 激活层占比
            sf = mr_gptq_module._INCREMENTS["scale_fitting"]
            mr_gptq_module.quantize_fused_experts(model, scale_fitting=sf)
        report_memory_full(model)  # 全参数口径的真实显存降低（验收用）

    testenc = get_test_dataset(tokenizer)  # 保持 CPU，test_ppl 逐 batch 搬到输入卡
    ppl = test_ppl(model, testenc, max_samples=args.max_eval_samples)
    print(f"[RESULT] model={args.model_path} mode={args.mode} "
          f"seq_len={args.seq_len} dtype={args.dtype} PPL={ppl:.4f}")


if __name__ == "__main__":
    main()
