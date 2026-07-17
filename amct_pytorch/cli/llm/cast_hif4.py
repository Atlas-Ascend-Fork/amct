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
"""CLI entry point for FP16/BF16 <-> HiFloat4 checkpoint conversion.

Thin front door -- the implementation (fake-quant / pack / unpack) lives in
``amct_pytorch.common.utils.hifx4_quant.checkpoint_cast``. Usage::

    python -m amct_pytorch.cli.llm.cast_hif4 fake-quant --input <model> --output <out> --dtype bf16
    python -m amct_pytorch.cli.llm.cast_hif4 pack       --input <model> --output <out>
    python -m amct_pytorch.cli.llm.cast_hif4 unpack     --input <packed> --output <out>
"""
from amct_pytorch.common.utils.hifx4_quant.checkpoint_cast import main


if __name__ == "__main__":
    main()
