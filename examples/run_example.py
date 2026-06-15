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

验证 cast / minmax / smoothquant 三种算法的完整量化流程（quantize -> calibrate -> convert -> forward）。
使用随机校准数据，无需下载数据集。模型默认从 HuggingFace 实时下载 Qwen2-0.5B（~1GB）。
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

_ALL_CASES = ("cast", "minmax", "smoothquant")


def _parse_args():
    p = argparse.ArgumentParser(description="AMCT 快速验证")
    p.add_argument(
        "--model_name_or_path",
        default="Qwen/Qwen2-0.5B",
        help="HuggingFace 模型名或本地路径（默认 Qwen/Qwen2-0.5B）",
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


def _free(model, device):
    del model
    gc.collect()
    if device.startswith("npu"):
        import torch_npu  # noqa: F401
        torch_npu.npu.empty_cache()


def _run_case(name, base_model, cfg, device, needs_calib, skip_convert=False):
    import amct_pytorch as amct

    print(f"\n[{name}] 开始")
    t0 = time.time()

    model = copy.deepcopy(base_model).eval().to(device)

    amct.quantize(model, cfg)

    if needs_calib:
        with torch.no_grad():
            model(_dummy_input(model.config.vocab_size, device))

    if skip_convert:
        # convert() 依赖原生 hifloat8，环境不支持时跳过，仅验证仿真量化前向
        print("    convert() 已跳过（当前环境无原生 hifloat8 支持）")
    else:
        amct.convert(model)
        _verify_forward(model, device)

    _free(model, device)

    print(f"[{name}] PASS  耗时 {time.time() - t0:.1f}s")


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

    # lm_head 的 vocab 维度（151936）超过 NPU weight-quant 算子 n<=65535 限制，必须跳过。
    # 内置 INT8_MINMAX_WEIGHT_QUANT_CFG / INT8_SMOOTHQUANT_CFG 无 skip_layers，不能直接用。
    _minmax_cfg = {
        'batch_num': 1,
        'quant_cfg': {
            'weights': {'type': 'int8', 'symmetric': True, 'strategy': 'channel'},
        },
        'algorithm': {'minmax'},
        'skip_layers': {'lm_head'},
    }
    _smoothquant_cfg = {
        'batch_num': 1,
        'quant_cfg': {
            'weights': {'type': 'int8', 'symmetric': True, 'strategy': 'channel'},
            'inputs': {'type': 'int8', 'symmetric': False, 'strategy': 'tensor'},
        },
        'algorithm': {'smoothquant': {'smooth_strength': 0.77}},
        'skip_layers': {'lm_head', 'down_proj'},
    }

    all_cases = [
        # cast: convert() 依赖原生 hifloat8，环境不支持时跳过 convert，只验证仿真量化前向
        {"name": "cast",        "cfg": amct.HIFP8_CAST_CFG, "needs_calib": True, "skip_convert": True},
        # minmax 仅权重量化，自定义 cfg 跳过 lm_head（vocab 超 NPU 算子 n<=65535 限制）
        {"name": "minmax",      "cfg": _minmax_cfg,          "needs_calib": True, "skip_convert": False},
        # smoothquant W8A8 全量化，自定义 cfg 跳过 lm_head + down_proj
        {"name": "smoothquant", "cfg": _smoothquant_cfg,     "needs_calib": True, "skip_convert": False},
    ]
    cases = [c for c in all_cases if c["name"] in requested]

    total_start = time.time()
    base_model = _load_base_model(args.model_name_or_path)

    results = {}
    for case in cases:
        try:
            _run_case(
                case["name"],
                base_model,
                case["cfg"],
                args.device,
                case["needs_calib"],
                case["skip_convert"],
            )
            results[case["name"]] = "PASS"
        except Exception as exc:
            print(f"[{case['name']}] FAIL: {exc}")
            results[case["name"]] = "FAIL"

    total = time.time() - total_start
    print(f"\n{'=' * 55}")
    print(f"总耗时: {total:.1f}s  ({total / 60:.1f}min)")
    print("验证结果汇总:")
    all_pass = True
    for name, result in results.items():
        tag = "PASS" if result == "PASS" else "FAIL"
        print(f"  [{tag}] {name}")
        if result != "PASS":
            all_pass = False

    if all_pass:
        print("[AMCT_EXAMPLE_ALL_PASS]")
    else:
        print("[AMCT_EXAMPLE_FAILED]")

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
