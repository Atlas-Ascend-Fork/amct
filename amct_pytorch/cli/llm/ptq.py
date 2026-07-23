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


def _run_hif4_convert(args, fmt):
    """HiF4 ptq convert (no calibration): normalize a non-float input to a plain FP16 checkpoint.

    - hif4 packed input -> reconstruct to FP16 via ``decode.unpack_hif4`` (model-agnostic; the
      model pipeline cannot load packed HiF4 weights).
    - nvfp4/mxfp4/mxfp8 input -> ``LlmPtqConvertWorkflow``: auto-decode the foreign format to
      BF16, build the model layer-by-layer, and HiF4 fake-quant each quantizable weight (NPU
      kernel), emitting a plain FP16 checkpoint. Non-quantized tensors are copied.
    """
    if fmt == "hif4":
        from amct_pytorch.common.utils.decode.unpack_hif4 import unpack_hif4_checkpoint
        os.makedirs(args.output_dir, exist_ok=True)
        unpack_hif4_checkpoint(args.model, args.output_dir, out_dtype="fp16")
        return

    import torch
    from amct_pytorch.workflows.llm_ptq import LlmPtqConvertWorkflow

    args.granularity = "block"  # per-model fake-quant runs block-by-block
    LlmPtqConvertWorkflow(args, out_dtype=torch.float16).run()


def main():
    args = parser_gen(command="ptq")
    # HiFloat: route HiF4-family inputs (hif4 packed, or foreign nvfp4/mxfp4/mxfp8) through the
    # convert path -> plain FP16. A plain float input (incl. HiFloat8) keeps the standard
    # calibration workflow.
    if args.quant_dtype == "hifp":
        from amct_pytorch.common.utils.decode.detect import detect_source_format, FOREIGN_FORMATS
        fmt = detect_source_format(args.model)
        print(f"[ptq/hifp] source format detected: {fmt}", flush=True)
        if fmt == "hif4" or fmt in FOREIGN_FORMATS:
            _run_hif4_convert(args, fmt)
            return
    LlmPtqWorkflow(args).run()


if __name__ == "__main__":
    main()
