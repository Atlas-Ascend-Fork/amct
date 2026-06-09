# ----------------------------------------------------------------------------
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------
import argparse
from collections import namedtuple
from datetime import datetime

import torch
import torch_npu


def parse_shape(value):
    parts = value.split(",")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(f"shape must be M,K,N, got {value!r}")
    try:
        return tuple(int(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"shape must contain integers, got {value!r}"
        ) from exc


def parse_args():
    parser = argparse.ArgumentParser(
        description="Lightweight validation runner for dynamic-quant + qmm estimator."
    )
    parser.add_argument(
        "--shapes",
        nargs="+",
        type=parse_shape,
        default=[
            (4096, 4096, 12800),
            (4096, 4096, 4096),
            (4096, 1024, 1024),
            (256, 1024, 1024),
            (64, 1024, 1024),
        ],
        help="Shapes in M,K,N format. Example: --shapes 4096,4096,12800 4096,1024,1024",
    )
    parser.add_argument("--iters", type=int, default=100, help="Total loop iterations.")
    parser.add_argument(
        "--warmup",
        type=int,
        default=50,
        help="Iterations skipped before profiler step.",
    )
    parser.add_argument("--device", default="npu:0", help="Torch device, default: npu:0.")
    parser.add_argument(
        "--profile-dir",
        default="./profiling",
        help="Profiler output root directory.",
    )
    return parser.parse_args()


def make_profiler(profile_dir):
    experimental_config = torch_npu.profiler._ExperimentalConfig(
        export_type=torch_npu.profiler.ExportType.Text,
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
        msprof_tx=False,
        aic_metrics=torch_npu.profiler.AiCMetrics.AiCoreNone,
        l2_cache=False,
        op_attr=False,
        data_simplification=False,
        record_op_args=False,
        gc_detect_threshold=None,
    )

    return torch_npu.profiler.profile(
        activities=[
            torch_npu.profiler.ProfilerActivity.CPU,
            torch_npu.profiler.ProfilerActivity.NPU,
        ],
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
            f"{profile_dir}/{datetime.now().strftime('%Y-%m-%d')}/quant"
        ),
        record_shapes=True,
        profile_memory=False,
        with_stack=True,
        with_modules=False,
        with_flops=False,
        experimental_config=experimental_config,
    )


def bf16_matmul(x, weight):
    return x @ weight.T


TARGET_ORDER = ("int_w8a8", "int_w4a4")


def qmm_matmul(
    x_q,
    x_scale,
    weight_q_t,
    weight_scale,
    m,
    n,
    offset=None,
    group_sizes=None,
):
    return torch_npu.npu_quant_matmul(
        x_q,
        weight_q_t,
        weight_scale,
        offset=offset,
        pertoken_scale=x_scale,
        output_dtype=torch.bfloat16,
        group_sizes=group_sizes,
    ).view(m, n)


def int_w8a8_matmul(x, weight_int8_t, weight_scale):
    m, k = x.shape
    k_t, n = weight_int8_t.shape
    if k_t != k:
        raise ValueError(f"weight K mismatch: x K={k}, weight K={k_t}")
    x_int8, x_pertoken_scale = torch_npu.npu_dynamic_quant(x)
    return qmm_matmul(x_int8, x_pertoken_scale, weight_int8_t, weight_scale, m, n)


# TODO: Re-enable int_w4a8 after the official A8W4 scale/packing path is
# confirmed. The grouped A8W4 path requires group_size=256 and grouped
# scale/offset tensors, and the previous timing path was not a reliable
# validation target for the first version of this skill.


def int_w4a4_matmul(x, weight_int4_t, weight_scale):
    m, k = x.shape
    _, n = weight_int4_t.shape
    x_int4, x_pertoken_scale = torch_npu.npu_dynamic_quant(
        x, dst_type=torch.quint4x2
    )
    return qmm_matmul(x_int4, x_pertoken_scale, weight_int4_t, weight_scale, m, n)


Case = namedtuple(
    "Case",
    [
        "x",
        "weight",
        "weight_int8_t",
        "weight_int8_scale",
        "weight_int4_t",
        "weight_int4_scale",
    ],
)


def prepare_case(shape, device):
    m, k, n = shape
    x = torch.randn(m, k, dtype=torch.bfloat16, device=device)
    weight = torch.randn(n, k, dtype=torch.bfloat16, device=device)
    weight_int8, weight_scale = torch_npu.npu_dynamic_quant(weight)
    weight_int8_t = weight_int8.T.contiguous()
    weight_int8_scale = weight_scale.view(-1)
    weight_int4, weight_scale = torch_npu.npu_dynamic_quant(
        weight, dst_type=torch.quint4x2
    )
    weight_int4_t = weight_int4.T
    weight_int4_scale = weight_scale.view(-1)

    return Case(
        x,
        weight,
        weight_int8_t,
        weight_int8_scale,
        weight_int4_t,
        weight_int4_scale,
    )


def run_case(case):
    out_gold = bf16_matmul(case.x, case.weight)
    out_int_w8a8 = int_w8a8_matmul(case.x, case.weight_int8_t, case.weight_int8_scale)
    out_int_w4a4 = int_w4a4_matmul(case.x, case.weight_int4_t, case.weight_int4_scale)
    return out_gold, out_int_w8a8, out_int_w4a4


def main():
    args = parse_args()
    device = torch.device(args.device)

    print("dynamic-quant qmm validation")
    print(f"device={device}")
    print("weight_preparation=offline_prepared")
    print("weight_quantization=offline_before_profile")
    print("target_order=" + ",".join(TARGET_ORDER))
    print("shapes=" + ", ".join(f"M={m},K={k},N={n}" for m, k, n in args.shapes))

    cases = [prepare_case(shape, device) for shape in args.shapes]
    torch.npu.synchronize()

    prof = make_profiler(args.profile_dir)
    prof.start()
    for i in range(args.iters):
        for case in cases:
            run_case(case)

        print(i)
        if i < args.warmup:
            torch.npu.synchronize()
            continue

        prof.step()
        torch.npu.synchronize()
    prof.stop()


if __name__ == "__main__":
    main()
