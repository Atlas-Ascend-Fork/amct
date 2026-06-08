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
"""Qwen2.5-3B HiFloat8 量化全流程脚本（CANN 体验官任务）。

使用 AMCT 模型压缩工具对 Qwen2.5-3B-Instruct 执行 HiFloat8 量化，并对比量化
前后在 wikitext2 上的困惑度（PPL），直观体现低比特量化的精度表现。

支持三种量化后端（--backend）：
    npu_op   —— 基于 amct_ops NPU 自定义算子的 HiFloat8 伪量化（默认，推荐）。
                算子为独立 ascendc kernel，不依赖 CANN aclnnQuantize 的 HiFloat8
                支持，在当前 CANN 9.1.0 环境可跑通，且全程 NPU、速度快。
    cpu_sim  —— 基于 experimental/hifloat8 CPU 扩展的伪量化。逐层 CPU<->NPU
                搬运，速度慢，仅作精度对照。
    amct     —— AMCT 内置 HIFP8_CAST_CFG，走 torch_npu.npu_quantize(hifloat8)。
                需要 torch_npu 注册 hifloat8 dtype 且 CANN aclnnQuantize 内核
                支持 HiFloat8；当前环境会失败，留作完整环境验证。

示例：
    # 默认 NPU 自定义算子路径，全量评估
    python3 quantize.py --model_path /home/developer/models/Qwen2.5-3B-Instruct

    # CPU 仿真路径，快速冒烟
    python3 quantize.py --model_path <path> --backend cpu_sim --max_samples 5

环境提示（详见 README）：
    - 运行需将仓库根加入 PYTHONPATH 以导入 amct_pytorch / amct_ops。
    - npu_op 后端需要 torch_npu==2.7.1.post4（注册 hifloat8 dtype）。
    - wikitext2 默认从本地 parquet 读取（WIKITEXT2_PARQUET 或 --dataset_path）。
"""

import argparse
import json
import os
import time

import torch
import torch_npu  # 注册 NPU 后端，并用于读取版本号
from eval_common import get_model, get_wikitext2_test, eval_ppl

import amct_pytorch as amct

# HiFloat8 高精度表示范围
HIF8_RANGE = 16.0


def parse_args():
    parser = argparse.ArgumentParser(
        description="Qwen2.5-3B HiFloat8 量化全流程（AMCT）"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Qwen2.5-3B-Instruct 本地模型路径",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="npu_op",
        choices=["npu_op", "cpu_sim", "amct"],
        help="量化后端，默认 npu_op（NPU 自定义算子伪量化）",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="wikitext2 测试集 parquet 本地路径（可选）",
    )
    parser.add_argument(
        "--seqlen", type=int, default=2048, help="评估序列长度，默认 2048"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="限制评估段数（冒烟用），默认全量",
    )
    parser.add_argument(
        "--device", type=str, default="npu:0", help="推理设备，默认 npu:0"
    )
    parser.add_argument(
        "--output", type=str, default="result.json", help="结果输出 JSON 路径"
    )
    parser.add_argument(
        "--skip_baseline", action="store_true", help="跳过量化前 baseline 评估"
    )
    return parser.parse_args()


def quantize_model(model, backend, device):
    """对模型应用 HiFloat8 量化，返回（量化后模型, 配置描述）。"""
    if backend == "amct":
        cfg = amct.HIFP8_CAST_CFG
        amct.quantize(model, cfg)
        return model, _jsonable(cfg)

    if backend == "npu_op":
        from npu_hifloat8_fakequant_linear import NpuHifloat8FakequantLinear

        algo, mod = "npu_hif8_fakequant", NpuHifloat8FakequantLinear
    else:  # cpu_sim
        from hifloat8_fakequant_linear import Hifloat8FakequantLinear

        algo, mod = "hifloat8_fakequant", Hifloat8FakequantLinear

    cfg = {
        "quant_cfg": {
            "weights": {
                "type": "hifloat8",
                "symmetric": True,
                "strategy": "channel",
            },
            "inputs": {
                "type": "hifloat8",
                "symmetric": True,
                "strategy": "tensor",
            },
        },
        "algorithm": {algo},
        "skip_layers": {"lm_head"},
    }
    amct.algorithm_register(algo, "Linear", mod, None)
    amct.quantize(model, cfg)
    return model, _jsonable(cfg)


def _build_result(args, device):
    """构造结果字典的初始字段。"""
    return {
        "model_path": args.model_path,
        "backend": args.backend,
        "seqlen": args.seqlen,
        "max_samples": args.max_samples,
        "device": device,
        "torch_npu": torch_npu.__version__,
    }


def _eval_baseline(model, testenc, device, args):
    """量化前 FP16 baseline PPL 评估，返回 (ppl, 耗时秒)。"""
    fp16_model = model.eval().to(device)
    t0 = time.time()
    ppl = eval_ppl(fp16_model, testenc, device, max_samples=args.max_samples)
    elapsed = round(time.time() - t0, 1)
    fp16_model = fp16_model.cpu()
    del fp16_model
    torch.npu.empty_cache()
    return ppl, elapsed


def _summarize(args, result):
    """计算 PPL 变化、打印汇总并写出结果 JSON。"""
    if "baseline_ppl" in result:
        delta = result["quant_ppl"] - result["baseline_ppl"]
        result["ppl_delta"] = round(delta, 6)
        result["ppl_delta_pct"] = round(
            delta / result["baseline_ppl"] * 100, 3
        )

    print("=" * 60)
    print("[结果汇总]")
    print(f"  模型      : {args.model_path}")
    print(f"  后端      : {args.backend}")
    if "baseline_ppl" in result:
        print(f"  量化前 PPL: {result['baseline_ppl']:.6f}")
    print(f"  量化后 PPL: {result['quant_ppl']:.6f}")
    if "ppl_delta" in result:
        print(
            f"  PPL 变化  : {result['ppl_delta']:+.6f} "
            f"({result['ppl_delta_pct']:+.3f}%)"
        )
    print("=" * 60)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"结果已写入 {args.output}")


def main():
    args = parse_args()
    if args.dataset_path:
        os.environ["WIKITEXT2_PARQUET"] = args.dataset_path

    dev_id = int(args.device.split(":")[-1]) if ":" in args.device else 0
    torch.npu.set_device(dev_id)
    device = f"npu:{dev_id}"

    result = _build_result(args, device)

    # ---- Phase 0：加载模型与数据集 ----
    print("=" * 60)
    print(f"[Phase 0] 加载模型与数据集，backend = {args.backend}")
    print("=" * 60)
    model, tokenizer = get_model(args.model_path, seqlen=args.seqlen)
    testenc = get_wikitext2_test(tokenizer)

    # ---- Phase 1：量化前 baseline PPL ----
    if not args.skip_baseline:
        print("=" * 60)
        print("[Phase 1] 量化前 baseline（FP16）PPL 评估")
        print("=" * 60)
        result["baseline_ppl"], result["baseline_eval_sec"] = _eval_baseline(
            model, testenc, device, args
        )
        model, tokenizer = get_model(args.model_path, seqlen=args.seqlen)

    # ---- Phase 2：HiFloat8 量化 ----
    print("=" * 60)
    print(f"[Phase 2] HiFloat8 量化（backend={args.backend}）")
    print("=" * 60)
    quant_model = model.eval().to(device)
    t0 = time.time()
    quant_model, cfg_desc = quantize_model(quant_model, args.backend, device)
    result["quant_config"] = cfg_desc
    result["quantize_sec"] = round(time.time() - t0, 1)
    print(f"[Phase 2] 量化完成，耗时 {result['quantize_sec']}s")

    # ---- Phase 3：量化后 PPL ----
    print("=" * 60)
    print("[Phase 3] 量化后 HiFloat8 模型 PPL 评估")
    print("=" * 60)
    t0 = time.time()
    result["quant_ppl"] = eval_ppl(
        quant_model, testenc, device, max_samples=args.max_samples
    )
    result["quant_eval_sec"] = round(time.time() - t0, 1)

    # ---- 汇总 ----
    _summarize(args, result)


def _jsonable(obj):
    """把含 set 的量化配置转成可 JSON 序列化形式。"""
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (set, frozenset)):
        return sorted(obj)
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    return obj


if __name__ == "__main__":
    main()
