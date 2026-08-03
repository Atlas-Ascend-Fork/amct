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


def main():
    args = parser_gen(command="ptq")
    # HiFloat: a hif4-packed input is reconstructed to float (no model build). Every other input
    # goes through LlmPtqWorkflow, which either calibrates (with --algos) or converts the whole
    # model via data-free HiF4 fake-quant to a plain float checkpoint (no --algos).
    if args.quant_dtype == "hifp":
        from amct_pytorch.common.utils.decode.detect import detect_source_format
        fmt = detect_source_format(args.model)
        print(f"[ptq/hifp] source format detected: {fmt}", flush=True)
        if fmt == "hif4":
            from amct_pytorch.common.utils.decode.unpack_hif4 import unpack_hif4_checkpoint
            os.makedirs(args.output_dir, exist_ok=True)
            unpack_hif4_checkpoint(args.model, args.output_dir, out_dtype="fp16")
            return
    LlmPtqWorkflow(args).run()


if __name__ == "__main__":
    main()
