# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

import torch
import torch_npu
import numpy as np
import amct
import sys, os
from pathlib import Path

DEVICE_ID = 0
torch_npu.npu.set_device(int(DEVICE_ID))
torch.npu.config.allow_internal_format = False

SVD_QUANT_TEST_CASES = [
# Prefill, Rank: 32
   (32 * 1024, 10944, 2048, 32),
   (32 * 1024,  2816, 2048, 32),
   (32 * 1024,  1408, 2048, 32),
   (32 * 1024, 2048, 10944, 32),
   (32 * 1024, 2048,  2816, 32),
   (32 * 1024, 2048,  1408, 32),
   (32 * 1024, 2048, 2048, 32),
   (32 * 1024, 2048, 3072, 32),
   (32 * 1024,  512, 4096, 32),
# Decode, Rank: 32
   (1, 10944, 2048, 32),
   (1,  2816, 2048, 32),
   (1,  1408, 2048, 32),
   (1, 2048, 10944, 32),
   (1, 2048,  2816, 32),
   (1, 2048,  1408, 32),
   (1, 2048, 2048, 32),
   (1, 2048, 3072, 32),
   (1,  512, 4096, 32),
# Prefill, Rank: 64
   (32 * 1024, 10944, 2048, 64),
   (32 * 1024,  2816, 2048, 64),
   (32 * 1024,  1408, 2048, 64),
   (32 * 1024, 2048, 10944, 64),
   (32 * 1024, 2048,  2816, 64),
   (32 * 1024, 2048,  1408, 64),
   (32 * 1024, 2048, 2048, 64),
   (32 * 1024, 2048, 3072, 64),
   (32 * 1024,  512, 4096, 64),
# Decode, Rank: 64
   (1, 10944, 2048, 64),
   (1,  2816, 2048, 64),
   (1,  1408, 2048, 64),
   (1, 2048, 10944, 64),
   (1, 2048,  2816, 64),
   (1, 2048,  1408, 64),
   (1, 2048, 2048, 64),
   (1, 2048, 3072, 64),
   (1,  512, 4096, 64),
]

def generate_tensors(case):
    seq_len, n, k, rank = case
    a_shape = (seq_len, k)
    w_shape = (n, k)
    dp_shape = (k, rank)
    up_shape = (rank, n)

    np.random.seed(0)

    x = torch.tensor(np.random.uniform(-10, 10, a_shape), dtype=torch.bfloat16)
    w = torch.tensor(np.random.uniform(-10, 10, w_shape), dtype=torch.bfloat16)
    dp = torch.tensor(np.random.uniform(-10, 10, dp_shape), dtype=torch.bfloat16)
    up = torch.tensor(np.random.uniform(-10, 10, up_shape), dtype=torch.bfloat16)
    w_quant, scale = torch_npu.npu_dynamic_mx_quant(w.npu(), block_size=32, round_mode="round")

    return x, w_quant, scale, dp, up

def test_svd_quant_matmul(cases):
    torch.npu.synchronize()
    steps = 14  # skip_first + (wait + warmup + active) * repeat
    experimental_config = torch_npu.profiler._ExperimentalConfig(
        aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization, profiler_level=torch_npu.profiler.ProfilerLevel.Level1, l2_cache=False, data_simplification=False
    )
    with torch_npu.profiler.profile(
        activities=[
            torch_npu.profiler.ProfilerActivity.CPU,
            torch_npu.profiler.ProfilerActivity.NPU
            ],
        schedule=torch_npu.profiler.schedule(wait=1, warmup=1, active=2, repeat=2, skip_first=10),
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler("./perf_result/svd_quant"),

        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_flops=True,
        with_modules=False,
        experimental_config=experimental_config) as prof:
            for step in range(steps):
                for idx, case in enumerate(cases):
                    x, w_quant, scale, dp, up = generate_tensors(case)
                    svd_quant_out = torch.ops.amct.svd_quant(x.npu(), w_quant, scale,  dp.npu(), up.npu())
                prof.step()
if __name__ == '__main__':
    test_svd_quant_matmul(SVD_QUANT_TEST_CASES)
