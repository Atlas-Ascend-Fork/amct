# -*- coding: UTF-8 -*-
"""Run AMCT Cast, Quantile, and OFMR integration smoke tests on NPU."""

import argparse
import copy
import json
import time
from pathlib import Path

import torch
import torch_npu

import amct_pytorch as amct


CONFIGS = {
    "cast": amct.HIFP8_CAST_CFG,
    "quantile": amct.HIFP8_QUANTILE_CFG,
    "ofmr": amct.HIFP8_OFMR_CFG,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Smoke-test AMCT HiFloat8 algorithms")
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--output", type=Path, default=Path("smoke_amct_algorithms.json"))
    return parser.parse_args()


def make_model():
    return torch.nn.Sequential(
        torch.nn.Linear(16, 32),
        torch.nn.GELU(),
        torch.nn.Linear(32, 8),
    ).eval().to(torch.bfloat16)


def run_algorithm(name, config, state_dict, inputs, device):
    baseline = make_model()
    baseline.load_state_dict(state_dict)
    baseline = baseline.to(device)

    model = make_model()
    model.load_state_dict(state_dict)
    model = model.to(device)

    start = time.perf_counter()
    amct.quantize(model, copy.deepcopy(config))
    torch.npu.synchronize()
    quantize_ms = (time.perf_counter() - start) * 1000.0

    calibration_ms = None
    with torch.no_grad():
        baseline_output = baseline(inputs)
        if name != "cast":
            start = time.perf_counter()
            model(inputs)
            torch.npu.synchronize()
            calibration_ms = (time.perf_counter() - start) * 1000.0
        quantized_output = model(inputs)
        torch.npu.synchronize()

    error = (baseline_output.float() - quantized_output.float()).abs().cpu()
    return {
        "algorithm": name,
        "module_types": [type(module).__name__ for module in model.modules()],
        "quantize_ms": quantize_ms,
        "calibration_ms": calibration_ms,
        "output_shape": list(quantized_output.shape),
        "output_dtype": str(quantized_output.dtype),
        "finite": bool(torch.isfinite(quantized_output).all().item()),
        "max_abs_error": float(error.max()),
        "mean_abs_error": float(error.mean()),
    }


def main():
    args = parse_args()
    device_id = int(args.device.split(":", maxsplit=1)[1])
    torch.npu.set_device(device_id)
    torch.manual_seed(20260723)

    reference = make_model()
    state_dict = copy.deepcopy(reference.state_dict())
    inputs = torch.randn(4, 16, dtype=torch.bfloat16, device=args.device)

    report = {
        "environment": {
            "torch": torch.__version__,
            "torch_npu": torch_npu.__version__,
            "device": args.device,
            "device_name": torch.npu.get_device_name(device_id),
        },
        "scope": "integration smoke only; not a model-accuracy benchmark",
        "results": [
            run_algorithm(name, config, state_dict, inputs, args.device)
            for name, config in CONFIGS.items()
        ],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
