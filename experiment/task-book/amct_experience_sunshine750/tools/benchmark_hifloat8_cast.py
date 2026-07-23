# -*- coding: UTF-8 -*-
"""Reproducible HiFloat8 cast benchmark for the AMCT experience report."""

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import torch
import torch_npu

from amct_ops.hifloat8_cast import decode_from_hifloat8, encode_to_hifloat8


DEFAULT_SIZES = [
    1 << 10,
    1 << 12,
    1 << 14,
    1 << 16,
    1 << 18,
    1 << 20,
    1 << 22,
    1 << 24,
]


def positive_int(value):
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {value}")
    return parsed


def nonnegative_int(value):
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError(f"expected a nonnegative integer, got {value}")
    return parsed


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark AMCT HiFloat8 cast on NPU")
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--warmup", type=nonnegative_int, default=10)
    parser.add_argument("--repeats", type=positive_int, default=100)
    parser.add_argument("--sizes", type=positive_int, nargs="+", default=DEFAULT_SIZES)
    parser.add_argument("--output", type=Path, default=Path("benchmark_hifloat8_cast.json"))
    return parser.parse_args()


def summarize(samples_ms):
    ordered = sorted(samples_ms)
    p95_index = min(len(ordered) - 1, int(0.95 * len(ordered)))
    return {
        "mean_ms": statistics.fmean(samples_ms),
        "p50_ms": statistics.median(samples_ms),
        "p95_ms": ordered[p95_index],
        "std_ms": statistics.pstdev(samples_ms),
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
    }


def timed_call(fn, warmup, repeats):
    for _ in range(warmup):
        fn()
        torch.npu.synchronize()

    samples_ms = []
    result = None
    for _ in range(repeats):
        start = time.perf_counter()
        result = fn()
        torch.npu.synchronize()
        samples_ms.append((time.perf_counter() - start) * 1000.0)
    return result, summarize(samples_ms)


def throughput_mb_s(num_bytes, mean_ms):
    return num_bytes / (mean_ms / 1000.0) / 1_000_000.0


def benchmark_case(numel, dtype, warmup, repeats, device, lut_cache_expected_cold):
    x = torch.randn(numel, dtype=dtype, device=device)

    torch.npu.synchronize()
    cold_start = time.perf_counter()
    encoded = encode_to_hifloat8(x)
    torch.npu.synchronize()
    first_encode_call_ms = (time.perf_counter() - cold_start) * 1000.0

    encoded, encode_stats = timed_call(
        lambda: encode_to_hifloat8(x), warmup, repeats
    )
    decoded, decode_stats = timed_call(
        lambda: decode_from_hifloat8(encoded, dtype), warmup, repeats
    )
    _, roundtrip_stats = timed_call(
        lambda: decode_from_hifloat8(encode_to_hifloat8(x), dtype),
        warmup,
        repeats,
    )

    encode_stats["throughput_mb_s"] = throughput_mb_s(3 * numel, encode_stats["mean_ms"])
    decode_stats["throughput_mb_s"] = throughput_mb_s(3 * numel, decode_stats["mean_ms"])
    roundtrip_stats["effective_throughput_mb_s"] = throughput_mb_s(
        4 * numel, roundtrip_stats["mean_ms"]
    )
    roundtrip_stats["kernel_logical_throughput_mb_s"] = throughput_mb_s(
        6 * numel, roundtrip_stats["mean_ms"]
    )

    source = x.float().cpu()
    restored = decoded.float().cpu()
    abs_error = (source - restored).abs()
    finite = torch.isfinite(source) & torch.isfinite(restored)

    return {
        "numel": numel,
        "dtype": str(dtype).removeprefix("torch."),
        "first_encode_call_ms": first_encode_call_ms,
        "lut_cache_expected_cold": lut_cache_expected_cold,
        "encode": encode_stats,
        "decode": decode_stats,
        "roundtrip": roundtrip_stats,
        "accuracy": {
            "max_abs_error": float(abs_error[finite].max()) if finite.any() else None,
            "mean_abs_error": float(abs_error[finite].mean()) if finite.any() else None,
        },
    }


def benchmark_noncontiguous(dtype, warmup, repeats, device):
    contiguous = torch.randn((2048, 2048), dtype=dtype, device=device)
    noncontiguous = contiguous.transpose(0, 1)
    _, contiguous_stats = timed_call(
        lambda: encode_to_hifloat8(contiguous), warmup, repeats
    )
    _, noncontiguous_stats = timed_call(
        lambda: encode_to_hifloat8(noncontiguous), warmup, repeats
    )
    return {
        "shape": [2048, 2048],
        "dtype": str(dtype).removeprefix("torch."),
        "contiguous_mean_ms": contiguous_stats["mean_ms"],
        "noncontiguous_mean_ms": noncontiguous_stats["mean_ms"],
        "slowdown": noncontiguous_stats["mean_ms"] / contiguous_stats["mean_ms"],
    }


def main():
    args = parse_args()
    device_id = int(args.device.split(":", maxsplit=1)[1])
    torch.npu.set_device(device_id)
    torch.manual_seed(20260723)

    results = []
    for dtype in (torch.float16, torch.bfloat16):
        for size_index, size in enumerate(args.sizes):
            print(f"benchmarking dtype={dtype}, numel={size}", flush=True)
            results.append(
                benchmark_case(
                    size,
                    dtype,
                    args.warmup,
                    args.repeats,
                    args.device,
                    lut_cache_expected_cold=(size_index == 0),
                )
            )

    report = {
        "methodology": {
            "warmup": args.warmup,
            "repeats": args.repeats,
            "synchronize_each_sample": True,
            "throughput_unit": "decimal MB/s",
            "roundtrip_effective_bytes_per_element": 4,
            "roundtrip_kernel_logical_bytes_per_element": 6,
        },
        "environment": {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_npu": torch_npu.__version__,
            "device": args.device,
            "device_name": torch.npu.get_device_name(device_id),
        },
        "results": results,
        "noncontiguous": [
            benchmark_noncontiguous(dtype, args.warmup, args.repeats, args.device)
            for dtype in (torch.float16, torch.bfloat16)
        ],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
