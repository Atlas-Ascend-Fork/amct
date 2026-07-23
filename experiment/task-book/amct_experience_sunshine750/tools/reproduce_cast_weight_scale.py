# -*- coding: UTF-8 -*-
"""Reproduce the HiFloat8 Cast signed weight-scale issue."""

import argparse
import json
from pathlib import Path

import torch
import torch_npu  # noqa: F401 - register the NPU backend

from amct_pytorch.classic.quantize_op.utils import (
    QUANT_SCOPE,
    calculate_hifloat8_weight_scale,
    process_scale,
)
from amct_pytorch.common.utils.quant_util import quant_dequant_weight
from amct_pytorch.common.utils.vars import HIFLOAT8


def parse_args():
    parser = argparse.ArgumentParser(description="Reproduce HiFloat8 signed scale issue")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.npu.set_device(0)
    weight = torch.tensor(
        [
            [-1.0, 0.01, 0.02, 0.03],
            [-100.0, -120.0, -80.0, -90.0],
        ],
        dtype=torch.bfloat16,
        device="npu:0",
    )

    current_scale = calculate_hifloat8_weight_scale(weight, "channel")
    expected_scale = weight.abs().amax(dim=1) / QUANT_SCOPE[HIFLOAT8]
    expected_scale, _ = process_scale(expected_scale, None, symmetric=True)

    current_dq = quant_dequant_weight(weight, HIFLOAT8, current_scale)
    expected_dq = quant_dequant_weight(weight, HIFLOAT8, expected_scale)
    torch.npu.synchronize()

    current_error = (current_dq.float() - weight.float()).abs()
    expected_error = (expected_dq.float() - weight.float()).abs()
    report = {
        "weight": weight.float().cpu().tolist(),
        "current_scale_max_div_16": current_scale.cpu().tolist(),
        "expected_scale_absmax_div_16": expected_scale.cpu().tolist(),
        "current_dequantized": current_dq.float().cpu().tolist(),
        "expected_dequantized": expected_dq.float().cpu().tolist(),
        "current_max_abs_error": float(current_error.max().cpu()),
        "expected_max_abs_error": float(expected_error.max().cpu()),
        "current_mean_abs_error": float(current_error.mean().cpu()),
        "expected_mean_abs_error": float(expected_error.mean().cpu()),
    }
    serialized = json.dumps(report, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    print(serialized)


if __name__ == "__main__":
    main()
