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
"""CLI entry point for FP4/FP8 → HiFloat4 checkpoint conversion.

Supports --source-format nvfp4 | mxfp4 | mxfp8.
"""


def main():
    import argparse

    ap = argparse.ArgumentParser(
        description="FP4/FP8 → BF16 → HiFloat4 checkpoint converter"
    )
    ap.add_argument("--source-format", type=str, required=True,
                    choices=["nvfp4", "mxfp4", "mxfp8"],
                    help="Source quant format.")
    args, remainder = ap.parse_known_args()

    if args.source_format == "nvfp4":
        from amct_pytorch.common.utils.fpx_decode.convert_nvfp4_to_hif4 import main as _run
    elif args.source_format == "mxfp4":
        from amct_pytorch.common.utils.fpx_decode.convert_mxfp4_to_hif4 import main as _run
    elif args.source_format == "mxfp8":
        from amct_pytorch.common.utils.fpx_decode.convert_mxfp8_to_hif4 import main as _run
    else:
        ap.error(f"unsupported source format: {args.source_format}")

    _run(remainder if remainder else None)


if __name__ == "__main__":
    main()
