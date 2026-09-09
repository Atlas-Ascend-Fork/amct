# Copyright (c) 2025 Advanced Micro Devices, Inc.
# Modifications Copyright (c) 2026 Modelbest and Huawei
#
# This file is adapted from upstream open-source code.
# Licensed under the MIT License.

"""
Stage 3：把（OSPlus SmoothQuant 融合后的）BF16 MiniMax-M2.7 检查点用 RTN 转换为
Quark 风格 MXFP4。

与 stage1/stage2 不同，本步骤**不使用任何校准数据**，直接对可量化的 Linear 权重做
round-to-nearest MXFP4 量化，并导出张量命名/布局与 Quark 导出格式一致的 HuggingFace
safetensors 检查点：

- ``*.weight``：打包后的 uint8 FP4 载荷
- ``*.weight_scale``：uint8 e8m0 scale

不参与量化的张量（MoE 门控 / lm_head 等）保持 BF16 / 原 dtype。

输入通常是 stage2（``stage2_export_bf16.py``）导出的融合后 BF16 目录，也可直接对任意
可加载的 BF16 MiniMax-M2.7 检查点使用。
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import os
import shutil
import sys
from pathlib import Path

import torch
from tqdm import tqdm

# 将 src/ 自身加入 sys.path 以导入同级 common；再定位样例根目录以导入 mxfp4_quantizer 包。
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))
for _parent in _THIS_DIR.parents:
    if (_parent / "mxfp4_quantizer" / "__init__.py").is_file():
        if str(_parent) not in sys.path:
            sys.path.insert(0, str(_parent))
        break

from common import (  # noqa: E402
    ATTN_IMPLEMENTATION,
    EXCLUDE_PATTERNS,
    IGNORE_STATE_DICT_KEYS,
    matches_exclude,
)
from mxfp4_quantizer import mxfp4_quantize  # noqa: E402

COPY_FILES = [
    "chat_template.jinja",
    "configuration_minimax_m2.py",
    "generation_config.json",
    "modeling_minimax_m2.py",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "added_tokens.json",
    "special_tokens_map.json",
    "merges.txt",
]

GROUP_SIZE = 32
QUARK_VERSION = "0.11.1+210bbb7"


def get_module(model, path: str):
    module = model
    for part in path.split("."):
        module = getattr(module, part)
    return module


def resolve_load_device_map(
    load_device_mode: str, device_map_file: str | None
) -> None | str | dict[str, str | int]:
    if device_map_file is not None:
        from common import load_device_map

        return load_device_map(device_map_file)
    if load_device_mode == "auto":
        return "auto"
    return None


def resolve_quant_devices(device_spec: str) -> list[torch.device]:
    if device_spec == "auto":
        if hasattr(torch, "npu") and torch.npu.is_available():
            return [
                torch.device(f"npu:{idx}") for idx in range(torch.npu.device_count())
            ]
        if torch.cuda.is_available():
            return [
                torch.device(f"cuda:{idx}") for idx in range(torch.cuda.device_count())
            ]
        return [torch.device("cpu")]
    return [
        torch.device(device.strip())
        for device in device_spec.split(",")
        if device.strip()
    ]


def synchronize_device(device: torch.device) -> None:
    if device.type == "npu" and hasattr(torch, "npu"):
        torch.npu.synchronize(device)
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def clear_device_cache(device: torch.device) -> None:
    if device.type == "npu" and hasattr(torch, "npu"):
        torch.npu.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()


def quantize_linear_weight(
    name: str,
    weight: torch.Tensor,
    quant_device: torch.device,
    row_chunk_size: int,
) -> tuple[str, torch.Tensor, torch.Tensor]:
    packed_chunks = []
    scale_chunks = []
    total_rows = weight.shape[0]

    for start in range(0, total_rows, row_chunk_size):
        end = min(start + row_chunk_size, total_rows)
        quant_input = (
            weight[start:end].detach().to(device=quant_device, dtype=torch.float32)
        )
        packed, scale_e8m0 = mxfp4_quantize(quant_input, axis=-1)
        synchronize_device(quant_device)
        packed_chunks.append(packed.cpu())
        scale_chunks.append(scale_e8m0.cpu())
        del quant_input
        del packed
        del scale_e8m0
        clear_device_cache(quant_device)

    packed_all = (
        torch.cat(packed_chunks, dim=0) if len(packed_chunks) > 1 else packed_chunks[0]
    )
    scale_all = (
        torch.cat(scale_chunks, dim=0) if len(scale_chunks) > 1 else scale_chunks[0]
    )
    return name, packed_all, scale_all


def _collect_export_params_and_jobs(
    model,
    quant_devices: list[torch.device],
) -> tuple[
    dict[str, torch.Tensor],
    list[tuple[str, torch.Tensor, torch.device | None]],
]:
    state_dict: dict[str, torch.Tensor] = {}
    quant_jobs: list[tuple[str, torch.Tensor, torch.device | None]] = []
    quant_device_set = {str(device) for device in quant_devices}

    for name, param in model.named_parameters():
        if name in IGNORE_STATE_DICT_KEYS:
            continue
        if name.endswith(".weight") and param.dim() == 2:
            module_name = name.rsplit(".weight", 1)[0]
            module = get_module(model, module_name)
            if isinstance(module, torch.nn.Linear) and not matches_exclude(
                module_name, EXCLUDE_PATTERNS
            ):
                preferred_device = None
                if str(param.device) in quant_device_set:
                    preferred_device = torch.device(str(param.device))
                quant_jobs.append((name, param.data.detach(), preferred_device))
                continue
        state_dict[name] = param.data.cpu().clone()

    for name, buf in model.named_buffers():
        if name in IGNORE_STATE_DICT_KEYS:
            continue
        if name not in state_dict:
            state_dict[name] = buf.data.cpu().clone()
    return state_dict, quant_jobs


def _store_quantized_weight(
    state_dict: dict[str, torch.Tensor],
    name: str,
    packed: torch.Tensor,
    scale_e8m0: torch.Tensor,
) -> None:
    state_dict[name] = packed
    state_dict[name.replace(".weight", ".weight_scale")] = scale_e8m0


def _run_single_device_quant(
    state_dict: dict[str, torch.Tensor],
    quant_jobs: list[tuple[str, torch.Tensor, torch.device | None]],
    quant_device: torch.device,
    row_chunk_size: int,
) -> dict[str, torch.Tensor]:
    for name, weight, preferred_device in tqdm(
        quant_jobs, desc=f"RTN quant on {quant_device}"
    ):
        run_device = preferred_device or quant_device
        _, packed, scale_e8m0 = quantize_linear_weight(
            name, weight, run_device, row_chunk_size=row_chunk_size
        )
        _store_quantized_weight(state_dict, name, packed, scale_e8m0)
    print(f"[RTN] Quantized {len(quant_jobs)} linear weights on {quant_device}")
    return state_dict


def _pick_quant_device(
    preferred_device: torch.device | None,
    quant_devices: list[torch.device],
    job_idx: int,
) -> torch.device:
    return preferred_device or quant_devices[job_idx % len(quant_devices)]


def _submit_quant_job(
    executor: cf.ThreadPoolExecutor,
    quant_jobs: list[tuple[str, torch.Tensor, torch.device | None]],
    job_idx: int,
    quant_devices: list[torch.device],
    row_chunk_size: int,
) -> tuple[cf.Future, str, torch.device]:
    name, weight, preferred_device = quant_jobs[job_idx]
    quant_device = _pick_quant_device(preferred_device, quant_devices, job_idx)
    future = executor.submit(
        quantize_linear_weight,
        name,
        weight,
        quant_device,
        row_chunk_size,
    )
    return future, name, quant_device


def _prime_quant_jobs(
    executor: cf.ThreadPoolExecutor,
    quant_jobs: list[tuple[str, torch.Tensor, torch.device | None]],
    quant_devices: list[torch.device],
    row_chunk_size: int,
    max_inflight: int,
) -> tuple[int, dict[cf.Future, tuple[str, torch.device]]]:
    next_job_idx = 0
    future_to_meta: dict[cf.Future, tuple[str, torch.device]] = {}
    while next_job_idx < len(quant_jobs) and len(future_to_meta) < max_inflight:
        future, name, quant_device = _submit_quant_job(
            executor, quant_jobs, next_job_idx, quant_devices, row_chunk_size
        )
        future_to_meta[future] = (name, quant_device)
        next_job_idx += 1
    return next_job_idx, future_to_meta


def _handle_finished_quant_job(
    future: cf.Future,
    future_to_meta: dict[cf.Future, tuple[str, torch.device]],
    state_dict: dict[str, torch.Tensor],
    progress,
    executor: cf.ThreadPoolExecutor,
    quant_jobs: list[tuple[str, torch.Tensor, torch.device | None]],
    next_job_idx: int,
    quant_devices: list[torch.device],
    row_chunk_size: int,
) -> int:
    future_to_meta.pop(future)
    finished_name, packed, scale_e8m0 = future.result()
    _store_quantized_weight(state_dict, finished_name, packed, scale_e8m0)
    progress.update(1)
    if next_job_idx >= len(quant_jobs):
        return next_job_idx
    new_future, next_name, quant_device = _submit_quant_job(
        executor, quant_jobs, next_job_idx, quant_devices, row_chunk_size
    )
    future_to_meta[new_future] = (next_name, quant_device)
    return next_job_idx + 1


def _drain_quant_jobs(
    executor: cf.ThreadPoolExecutor,
    quant_jobs: list[tuple[str, torch.Tensor, torch.device | None]],
    quant_devices: list[torch.device],
    row_chunk_size: int,
    next_job_idx: int,
    future_to_meta: dict[cf.Future, tuple[str, torch.device]],
    state_dict: dict[str, torch.Tensor],
) -> None:
    progress = tqdm(total=len(quant_jobs), desc="RTN quant multi-device")
    while future_to_meta:
        done, _ = cf.wait(future_to_meta, return_when=cf.FIRST_COMPLETED)
        for future in done:
            next_job_idx = _handle_finished_quant_job(
                future,
                future_to_meta,
                state_dict,
                progress,
                executor,
                quant_jobs,
                next_job_idx,
                quant_devices,
                row_chunk_size,
            )
    progress.close()


def _run_multi_device_quant(
    state_dict: dict[str, torch.Tensor],
    quant_jobs: list[tuple[str, torch.Tensor, torch.device | None]],
    quant_devices: list[torch.device],
    max_inflight_jobs: int,
    row_chunk_size: int,
) -> dict[str, torch.Tensor]:
    max_workers = len(quant_devices)
    max_inflight = max(max_inflight_jobs, max_workers)
    print(
        f"[RTN] Quantizing {len(quant_jobs)} linear weights across "
        f"{len(quant_devices)} devices"
    )

    with cf.ThreadPoolExecutor(max_workers=max_workers) as executor:
        next_job_idx, future_to_meta = _prime_quant_jobs(
            executor, quant_jobs, quant_devices, row_chunk_size, max_inflight
        )
        _drain_quant_jobs(
            executor,
            quant_jobs,
            quant_devices,
            row_chunk_size,
            next_job_idx,
            future_to_meta,
            state_dict,
        )

    print(
        f"[RTN] Quantized {len(quant_jobs)} linear weights on "
        + ", ".join(str(device) for device in quant_devices)
    )
    return state_dict


def build_export_state_dict(
    model,
    quant_devices: list[torch.device],
    max_inflight_jobs: int,
    row_chunk_size: int,
) -> dict[str, torch.Tensor]:
    state_dict, quant_jobs = _collect_export_params_and_jobs(model, quant_devices)
    if not quant_jobs:
        return state_dict
    if len(quant_devices) == 1:
        return _run_single_device_quant(
            state_dict, quant_jobs, quant_devices[0], row_chunk_size
        )
    return _run_multi_device_quant(
        state_dict, quant_jobs, quant_devices, max_inflight_jobs, row_chunk_size
    )


def build_exclude_modules(model) -> list[str]:
    excluded = []
    for module_name, module in model.named_modules():
        if not module_name:
            continue
        if isinstance(module, torch.nn.Linear) and matches_exclude(
            module_name, EXCLUDE_PATTERNS
        ):
            excluded.append(module_name)
    return sorted(set(excluded))


def build_quantization_config(excluded_modules: list[str]) -> dict:
    tensor_quant_cfg = {
        "ch_axis": -1,
        "dtype": "fp4",
        "group_size": GROUP_SIZE,
        "is_dynamic": False,
        "is_scale_quant": False,
        "mx_element_dtype": None,
        "observer_cls": "PerBlockMXObserver",
        "qscheme": "per_group",
        "round_method": "half_even",
        "scale_calculation_mode": "even",
        "scale_format": "e8m0",
        "scale_type": "float",
        "symmetric": None,
    }
    input_quant_cfg = dict(tensor_quant_cfg)
    input_quant_cfg["is_dynamic"] = True

    return {
        "algo_config": [],
        "exclude": excluded_modules,
        "export": {
            "kv_cache_group": [],
            "min_kv_scale": 0.0,
            "pack_method": "reorder",
            "weight_format": "real_quantized",
            "weight_merge_groups": None,
        },
        "global_quant_config": {
            "bias": None,
            "input_tensors": input_quant_cfg,
            "output_tensors": None,
            "target_device": None,
            "weight": tensor_quant_cfg,
        },
        "kv_cache_post_rope": False,
        "kv_cache_quant_config": {},
        "layer_quant_config": {},
        "layer_type_quant_config": {},
        "quant_method": "quark",
        "quant_mode": "eager_mode",
        "softmax_quant_spec": None,
        "version": QUARK_VERSION,
    }


def export_model(
    model_dir: str,
    output_dir: str,
    load_device_mode: str = "cpu",
    device_map_file: str | None = None,
    quant_device_name: str = "auto",
    max_inflight_jobs: int = 32,
    row_chunk_size: int = 512,
) -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device_map = resolve_load_device_map(load_device_mode, device_map_file)
    quant_devices = resolve_quant_devices(quant_device_name)
    load_kwargs = {
        "dtype": torch.bfloat16,
        "trust_remote_code": True,
        "attn_implementation": ATTN_IMPLEMENTATION,
        "low_cpu_mem_usage": True,
    }
    if device_map is not None:
        load_kwargs["device_map"] = device_map

    print(f"[RTN] Loading model from {model_dir}...")
    model = AutoModelForCausalLM.from_pretrained(model_dir, **load_kwargs)
    model.eval()

    excluded_modules = build_exclude_modules(model)
    print(f"[RTN] Expanded {len(excluded_modules)} excluded modules")
    print(f"[RTN] Quant devices: {', '.join(str(device) for device in quant_devices)}")
    print(f"[RTN] Row chunk size: {row_chunk_size}")

    print("[RTN] Quantizing weights to MXFP4 and building state dict...")
    state_dict = build_export_state_dict(
        model,
        quant_devices=quant_devices,
        max_inflight_jobs=max_inflight_jobs,
        row_chunk_size=row_chunk_size,
    )

    os.makedirs(output_dir, exist_ok=True)
    model.config.quantization_config = build_quantization_config(excluded_modules)
    model.save_pretrained(
        output_dir, state_dict=state_dict, safe_serialization=True, max_shard_size="5GB"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_dir, padding_side="left", trust_remote_code=True
    )
    tokenizer.save_pretrained(output_dir)

    for filename in COPY_FILES:
        src = os.path.join(model_dir, filename)
        dst = os.path.join(output_dir, filename)
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy2(src, dst)

    print(f"[RTN] Done. Exported model to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Export BF16 MiniMax-M2.7 to RTN MXFP4"
    )
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--load_device_map", default="cpu", choices=["cpu", "auto"])
    parser.add_argument("--device_map_file", default=None)
    parser.add_argument("--quant_device", default="auto")
    parser.add_argument("--max_inflight_jobs", type=int, default=32)
    parser.add_argument("--row_chunk_size", type=int, default=512)
    args = parser.parse_args()

    export_model(
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        load_device_mode=args.load_device_map,
        device_map_file=args.device_map_file,
        quant_device_name=args.quant_device,
        max_inflight_jobs=args.max_inflight_jobs,
        row_chunk_size=args.row_chunk_size,
    )


if __name__ == "__main__":
    main()
