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
"""Vendored HiFloat4 (hifx4) fake-quantization simulation.

This is the pure-PyTorch ("force_py") path of the HiFloat4 reference project, vendored
into amct so that the HiFloat4 *cast* algorithm can simulate 4-bit quant-dequant without
compiling any CUDA / Ascend CCE kernels. The math is identical to the compiled kernel
(the upstream project asserts the two paths match to zero difference); the compiled kernel
is only a speed optimization and is intentionally NOT vendored here.

Public API mirrors the upstream `quant_cy` package:
    from amct_pytorch.common.utils.hifx4_quant import QType, quant_dequant_float
    w_sim = quant_dequant_float(weight, QType('hifx4').dim(-1), force_py=True, force_fp32=True)
"""

from .qtype import QType
from .quant_dequant import quant_dequant_float, quant_hifx

__all__ = ['QType', 'quant_dequant_float', 'quant_hifx']
