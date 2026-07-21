# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
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

import os

from amct_pytorch.cli.llm.args import parser_gen
from amct_pytorch.workflows.llm_ptq import LlmPtqWorkflow


def _run_hif4_convert(args):
    """HiF4 ptq: convert-only (no calibration).

    Auto-detect the input format, decode foreign FP4/FP8 to BF16 (fpx_decode), then apply HiF4
    fake-quant per weight -> float checkpoint. Output dtype: fp16/bf16 preserve the source;
    foreign and hif4 inputs default to fp16. HiF4 input is reconstructed (unpack), not re-quantized.
    """
    from amct_pytorch.common.utils.fpx_decode.detect import (
        detect_source_format, decode_to_bf16, FOREIGN_FORMATS,
    )
    from amct_pytorch.common.utils.hifloat4_quant.checkpoint_cast import main as cast_main

    model_dir = args.model
    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    fmt = detect_source_format(model_dir)
    print(f"[ptq/hif4] source format detected: {fmt}", flush=True)

    src = model_dir
    if fmt in FOREIGN_FORMATS:
        decoded = os.path.join(out_dir, "_decoded_bf16")
        print(f"[ptq/hif4] decoding {fmt} -> BF16 at {decoded}", flush=True)
        decode_to_bf16(model_dir, decoded, fmt)
        src, fmt = decoded, "bf16"

    if fmt == "hif4":
        # already HiF4: reconstruct to float (no re-quant)
        cast_main(["unpack", "--input", src, "--output", out_dir, "--dtype", "fp16"])
    else:
        dtype = "bf16" if fmt == "bf16" else "fp16"
        cast_main(["fake-quant", "--input", src, "--output", out_dir, "--dtype", dtype])


def main():
    args = parser_gen(command="ptq")
    # HiF4 ptq is convert-only (no calibration); other dtypes keep the calibration workflow.
    if args.quant_dtype == "hifloat4":
        _run_hif4_convert(args)
    else:
        workflow = LlmPtqWorkflow(args)
        workflow.run()


if __name__ == "__main__":
    main()
