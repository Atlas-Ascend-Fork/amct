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
    """HiF4 ptq: convert-only (no calibration), via the per-model route.

    - fp16/bf16/nvfp4/mxfp4/mxfp8 input -> per-model HiF4 fake-quant to a float checkpoint.
      Uses LlmDeployWorkflow in ``fake_quant_output`` mode: it auto-detects the format, decodes
      a foreign FP4/FP8 source to BF16 first, builds the model layer-by-layer, and fake-quants
      each quantizable weight (quant_target modules); non-quantized tensors are copied.
    - hif4 input -> reconstruct to float via checkpoint_cast unpack (the model pipeline cannot
      load packed HiF4 weights, so this stays model-agnostic).

    Output float dtype: fp16/bf16 preserve the source; foreign/hif4 default to fp16.
    """
    from amct_pytorch.common.utils.fpx_decode.detect import detect_source_format

    fmt = detect_source_format(args.model)
    print(f"[ptq/hif4] source format detected: {fmt}", flush=True)

    if fmt == "hif4":
        from amct_pytorch.common.utils.hifloat4_quant.checkpoint_cast import main as cast_main
        os.makedirs(args.output_dir, exist_ok=True)
        cast_main(["unpack", "--input", args.model, "--output", args.output_dir, "--dtype", "fp16"])
        return

    import torch
    from amct_pytorch.workflows.llm_deploy import LlmDeployWorkflow

    out_dtype = torch.bfloat16 if fmt == "bf16" else torch.float16
    args.granularity = "block"  # per-model fake-quant runs block-by-block
    LlmDeployWorkflow(args, fake_quant_output=True, out_dtype=out_dtype).run()


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
