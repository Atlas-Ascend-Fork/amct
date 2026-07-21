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
"""HiFloat4 fake-quant dispatcher: NPU kernel when available, else the pure-torch reference.

Unlike hifloat8 there is no native torch_npu hifloat4 dtype. On NPU (float16/bfloat16) this
prefers the ``amct_ops.hifloat4_cast`` AscendC operator; everywhere else -- and if amct_ops is
not installed -- it falls back to the pure-python ``quant_dequant_float`` simulation, which runs
on any device. Both paths are numerically equivalent (kernel validated to ~0 error). The pure
reference lives in ``quant_dequant.py`` and stays kernel-free; the optional amct_ops import is
isolated to this file.
"""

import torch

from .qtype import QType
from .quant_dequant import quant_dequant_float


@torch.no_grad()
def hifloat4_fake_quant(fp_tensor, qdim=-1):
    """FP -> HiFloat4 -> FP fake-quant round trip.

    The hifx4 format does its own 64-element block scaling along ``qdim`` -- so, unlike
    hifloat8_fake_quant, no external scale factor is computed or applied here. ``qdim`` defaults
    to the last (reduction) dim, matching how HiFloat4's reference QLinear quantizes both weights
    ([Cout, Cin] along Cin) and activations ([..., Cin] along Cin).

    Only ``float16``/``bfloat16`` tensors resident on NPU use the AscendC kernel; ``float32`` and
    any off-NPU tensor use the pure-torch reference (``quant_dequant_float``).
    """
    if fp_tensor.device.type == "npu" and fp_tensor.dtype in (torch.float16, torch.bfloat16):
        try:
            from amct_ops.hifloat4_cast import hifloat4_fake_quant as _npu_hifloat4_fake_quant
        except ImportError:
            _npu_hifloat4_fake_quant = None
        if _npu_hifloat4_fake_quant is not None:
            return _npu_hifloat4_fake_quant(fp_tensor, qdim=qdim)

    qtype = QType('hifx4').dim(qdim)
    return quant_dequant_float(fp_tensor, qtype, force_py=True, force_fp32=True).to(fp_tensor.dtype)
