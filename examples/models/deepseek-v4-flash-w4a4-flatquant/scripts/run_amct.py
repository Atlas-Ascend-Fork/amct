#!/usr/bin/env python3
# coding=utf-8
# ----------------------------------------------------------------------------
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------
"""Sample-local runner (same pattern as the w4a8 sample's expert-offload
wrapper): installs the load-time dequant patch, then dispatches to the amct
LLM CLI. Usage: run_amct.py {ptq|eval|extract_ptq_data} [original args...]
"""

import importlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dequant_load_patch import apply_dequant_patch  # noqa: E402

_MODULES = {
    "ptq": "amct_pytorch.ptq",
    "eval": "amct_pytorch.eval",
    "extract_ptq_data": "amct_pytorch.extract_ptq_data",
}


def main() -> None:
    apply_dequant_patch()
    step = sys.argv[1]
    sys.argv = [f"amct_pytorch.{step}"] + sys.argv[2:]
    module = importlib.import_module(_MODULES[step])
    module.main()


if __name__ == "__main__":
    main()
