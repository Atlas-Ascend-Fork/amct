# -*- coding: UTF-8 -*-
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
"""Minimal MXFP4 quantisation-aware training (QAT) building blocks.

Usage:
    import sys
    sys.path.insert(0, ".../amct_pytorch/experimental/fakequant")

    from mxfp4_qat import MXFP4QATConfig, convert_to_mxfp4_qat

    convert_to_mxfp4_qat(model, MXFP4QATConfig(quantize_input=True))
    # ... train as usual; nothing else changes ...
"""

from __future__ import annotations

__all__ = [
    "BACKEND_AUTO",
    "BACKEND_NPU",
    "BACKEND_TORCH",
    "BLOCK_SIZE",
    "MXFP4_E2M1_MAX",
    "MXFP4FakeQuantizer",
    "MXFP4QATConfig",
    "MXFP4QATLinear",
    "SCALE_FACTOR",
    "convert_to_mxfp4_qat",
    "mxfp4_fake_quant",
    "mxfp4_quant_dequant",
    "mxfp4_saturation_mask",
]

from .fake_quant import (
    BACKEND_AUTO,
    BACKEND_NPU,
    BACKEND_TORCH,
    BLOCK_SIZE,
    MXFP4_E2M1_MAX,
    SCALE_FACTOR,
    MXFP4FakeQuantizer,
    MXFP4QATConfig,
    mxfp4_fake_quant,
    mxfp4_quant_dequant,
    mxfp4_saturation_mask,
)
from .linear import MXFP4QATLinear, convert_to_mxfp4_qat
