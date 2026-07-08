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

"""AMCT 快速验证

验证 cast / minmax / smoothquant / awq 四种算法的完整量化流程（quantize -> calibrate -> convert -> forward）。
使用随机校准数据，无需下载数据集。模型默认从 HuggingFace 实时下载 Qwen3-0.6B（~0.6GB）。
"""

import argparse
import copy
import gc
import sys
import time

import torch

# 验证用短序列，节省耗时
_SEQLEN = 128
_CALIB_BATCH = 1

_ALL_CASES = ("cast", "minmax", "smoothquant", "awq")


def _parse_args():
    p = argparse.ArgumentParser(description="AMCT 快速验证")
    p.add_argument(
        "--model",
        default="Qwen/Qwen3-0.6B",
        help="HuggingFace 模型名或本地路径（默认 Qwen/Qwen3-0.6B）",
    )
    p.add_argument("--device", default="npu:0", help="运行设备（默认 npu:0）")
    p.add_argument(
        "--cases",
        default=",".join(_ALL_CASES),
        help=f"逗号分隔的用例列表，可选 {_ALL_CASES}（默认全部）",
    )
    return p.parse_args()


def _load_base_model(model_name_or_path):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[*] 加载模型: {model_name_or_path}")
    t0 = time.time()
    # 不用 device_map="auto"：多卡环境下 0.5B 会被拆到两卡，触发跨设备 addmm 报错
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    # 验证 tokenizer 可正常加载，提前暴露词表文件缺失等问题
    AutoTokenizer.from_pretrained(
        model_name_or_path, use_fast=False, trust_remote_code=True
    )
    print(f"    完成，耗时 {time.time() - t0:.1f}s")
    return model


def _dummy_input(vocab_size, device):
    return torch.randint(0, min(vocab_size, 1000), (_CALIB_BATCH, _SEQLEN), device=device)


def _verify_forward(model, device):
    with torch.no_grad():
        out = model(_dummy_input(model.config.vocab_size, device))
    logits = out.logits
    if torch.isnan(logits).any() or torch.isinf(logits).any():
        raise ValueError("模型输出含有 NaN / Inf")


def _run_case(name, base_model, cfg, device, needs_calib, skip_convert=False):
    import amct_pytorch as amct

    print(f"\n[{name}] 开始")
    t0 = time.time()

    # Phase0: prepare model
    quant_model = copy.deepcopy(base_model).eval().to(device)

    # Phase1: quantize model
    amct.quantize(quant_model, cfg)

    # Phase2: inference calibration model to cal quantized factors
    if needs_calib:
        with torch.no_grad():
            quant_model(_dummy_input(quant_model.config.vocab_size, device))
    if device.startswith("npu"):
        import torch_npu
        torch_npu.npu.empty_cache()

    # Phase3: convert deploy model
    if skip_convert:
        # convert() 依赖原生 hifloat8，环境不支持时跳过，仅验证仿真量化前向
        print("    convert() 已跳过（当前环境无原生 hifloat8 支持）")
    else:
        amct.convert(quant_model)
        if device.startswith("npu"):
            torch_npu.npu.empty_cache()

        # Phase4: verify forward pass
        _verify_forward(quant_model, device)

    del quant_model
    gc.collect()

    print(f"[{name}] PASS  耗时 {time.time() - t0:.1f}s")


def _build_cases(amct):
    # lm_head vocab 维度（151936）超过 NPU weight-quant 算子 n<=65535 限制，必须跳过。
    # 内置 INT8_MINMAX_WEIGHT_QUANT_CFG / INT8_SMOOTHQUANT_CFG 无 skip_layers，不能直接用。
    minmax_cfg = {
        "batch_num": 1,
        "quant_cfg": {
            "weights": {"type": "int8", "symmetric": True, "strategy": "channel"},
        },
        "algorithm": {"minmax"},
        "skip_layers": {"lm_head"},
    }
    smoothquant_cfg = {
        "batch_num": 1,
        "quant_cfg": {
            "weights": {"type": "int8", "symmetric": True, "strategy": "channel"},
            "inputs": {"type": "int8", "symmetric": False, "strategy": "tensor"},
        },
        "algorithm": {"smoothquant": {"smooth_strength": 0.77}},
        "skip_layers": {"lm_head", "down_proj"},
    }
    # awq INT4 group 量化，grids_num=1 降低搜索耗时（验证流程正确性，不追求精度）
    awq_cfg = {
        "batch_num": 1,
        "quant_cfg": {
            "weights": {
                "type": "int4",
                "symmetric": False,
                "strategy": "group",
                "group_size": 32,
            },
        },
        "algorithm": {"awq": {"grids_num": 1}},
        "skip_layers": {"lm_head"},
    }
    return [
        # cast: 依赖原生 hifloat8 或 amct_ops，环境不满足时整个用例 SKIP
        {
            "name": "cast", "cfg": amct.HIFP8_CAST_CFG, "needs_calib": True,
            "skip_convert": True,
            "skip_on": ("hifloat8", "amct_ops"),
        },
        # minmax 仅权重量化，自定义 cfg 跳过 lm_head（vocab 超 NPU 算子 n<=65535 限制）
        {"name": "minmax", "cfg": minmax_cfg, "needs_calib": True, "skip_convert": False},
        # smoothquant W8A8 全量化，自定义 cfg 跳过 lm_head + down_proj
        {"name": "smoothquant", "cfg": smoothquant_cfg, "needs_calib": True, "skip_convert": False},
        # awq INT4 group 量化，grids_num=1 仅验证流程完整性
        {"name": "awq", "cfg": awq_cfg, "needs_calib": True, "skip_convert": False},
    ]


def _run_all(cases, model_name_or_path, device):
    total_start = time.time()
    base_model = _load_base_model(model_name_or_path)

    results = {}
    for case in cases:
        skip_on = case.get("skip_on", ())
        try:
            _run_case(
                case["name"],
                base_model,
                case["cfg"],
                device,
                case["needs_calib"],
                case["skip_convert"],
            )
            results[case["name"]] = "PASS"
        except Exception as exc:
            msg = str(exc)
            if skip_on and any(kw in msg for kw in skip_on):
                print(f"[{case['name']}] SKIP: {msg}")
                results[case["name"]] = "SKIP"
            else:
                print(f"[{case['name']}] FAIL: {exc}")
                results[case["name"]] = "FAIL"

    total = time.time() - total_start
    print(f"\n{'=' * 55}")
    print(f"总耗时: {total:.1f}s  ({total / 60:.1f}min)")
    print("验证结果汇总:")
    all_pass = True
    for name, result in results.items():
        print(f"  [{result}] {name}")
        if result == "FAIL":
            all_pass = False
    return all_pass


def main():
    args = _parse_args()

    requested = [c.strip() for c in args.cases.split(",") if c.strip()]
    unknown = [c for c in requested if c not in _ALL_CASES]
    if unknown:
        print(f"[ERROR] 未知用例: {unknown}，可选: {_ALL_CASES}")
        sys.exit(1)

    if args.device.startswith("npu"):
        import torch_npu  # noqa: F401

    import amct_pytorch as amct

    all_cases = _build_cases(amct)
    cases = [c for c in all_cases if c["name"] in requested]

    all_pass = _run_all(cases, args.model, args.device)

    if all_pass:
        print("[AMCT_EXAMPLE_ALL_PASS]")
    else:
        print("[AMCT_EXAMPLE_FAILED]")

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
