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
"""Checkpoint format decoders: quantized checkpoint -> plain FP16/BF16 float.

One decoder per source format, each reconstructing a packed/quantized checkpoint back to a
plain float checkpoint (via the shared ``common/utils/checkpoint_io`` shard machinery):

    unpack_hif4    HiFloat4 packed (our own deploy pack) -> FP16/BF16
    unpack_nvfp4   NVFP4  -> BF16      (added downstream)
    unpack_mxfp4   MXFP4  -> BF16      (added downstream)
    unpack_mxfp8   MXFP8  -> BF16      (added downstream)

Decode is *only* the format -> float step. Any subsequent HiF4 fake-quant (the NPU kernel /
``hifloat4_quant``) is applied afterward by the ptq workflow -- it is not part of this layer.
Each decoder imports its format's reconstruct primitive from that format's package
(e.g. ``unpack_hif4`` uses ``hifloat4_quant.hif4_bit_reconstruct``); this folder owns only the
checkpoint-walk orchestration, not the per-format bit math.
"""
