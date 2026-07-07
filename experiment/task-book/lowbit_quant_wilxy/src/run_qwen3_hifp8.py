# ----------------------------------------------------------------------------
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the Apache License Version 2.0.
# You may not use this file except in compliance with the License.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# Apache License for more details at
# http://www.apache.org/licenses/LICENSE-2.0
# ----------------------------------------------------------------------------
"""HiFloat8 channel OFMR on Qwen3-8B (W+A quant).

This script demonstrates per-channel OFMR for HiFloat8 weight quantization
combined with per-tensor OFMR for activations.  Channel-level scale search
gives tighter per-output-channel MSE minimisation than the existing tensor-level
preset (HIFP8_OFMR_CFG).

Expected result on wikitext2:
    BF16 baseline PPL  ~9.72
    Quantized PPL      ≤ 9.82  (Δ ≤ 0.1)
    Linear quant ratio  ≥ 85 %
"""

import argparse
import sys
import os
import torch
import torch_npu

sys.path.insert(0, os.path.dirname(__file__))
from utils import get_test_dataset, get_qwen, get_calib_dataset, infer_model, test_ppl
from utils import get_memory_footprint, count_quantized_linears

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import HIFP8_OFMR_CHANNEL_CFG

import amct_pytorch as amct


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Qwen3-8B HiFloat8 channel OFMR quantization'
    )
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the pretrained Qwen3-8B model")
    parser.add_argument("--seqlen", type=int, default=2048,
                        help="Sequence length for evaluation")
    args = parser.parse_args()

    # Phase0: Load model & calibration data
    print("=" * 60)
    print("Phase0: Loading model and calibration data")
    print("=" * 60)
    model, enc = get_qwen(args.model_path, seqlen=args.seqlen)
    quant_model = model.eval().npu()

    samples = get_calib_dataset(tokenizer=enc, n_samples=512, block_size=256)
    samples = torch.cat(samples, dim=0)[:1, :]

    mem_before = get_memory_footprint(quant_model)
    print(f"NPU memory before quant: {mem_before[0]:.1f} MiB allocated, "
          f"{mem_before[1]:.1f} MiB reserved")

    # Phase1: Quantize model
    print("\n" + "=" * 60)
    print("Phase1: Quantizing model with HIFP8_OFMR_CHANNEL_CFG")
    print("=" * 60)
    amct.quantize(quant_model, HIFP8_OFMR_CHANNEL_CFG)

    # Phase2: Calibration inference
    print("\n" + "=" * 60)
    print("Phase2: Running calibration inference")
    print("=" * 60)
    infer_model(quant_model, samples)
    torch_npu.npu.empty_cache()

    # Phase3: Convert to deploy model
    print("\n" + "=" * 60)
    print("Phase3: Converting to deploy model")
    print("=" * 60)
    amct.convert(quant_model)
    torch_npu.npu.empty_cache()

    mem_after = get_memory_footprint(quant_model)
    print(f"NPU memory after quant:  {mem_after[0]:.1f} MiB allocated, "
          f"{mem_after[1]:.1f} MiB reserved")
    if mem_before[0] > 0:
        reduction = (1 - mem_after[0] / mem_before[0]) * 100
        print(f"Memory reduction: {reduction:.1f}%")

    total, quantized = count_quantized_linears(quant_model)
    if total > 0:
        ratio = quantized / total * 100
        print(f"Linear quant ratio: {quantized}/{total} = {ratio:.1f}%")

    # Phase4: PPL evaluation
    print("\n" + "=" * 60)
    print("Phase4: Evaluating PPL on Wikitext2")
    print("=" * 60)
    testenc = get_test_dataset(enc=enc, seqlen=model.seqlen)
    testenc = testenc.input_ids.npu()
    test_ppl(quant_model, testenc)
