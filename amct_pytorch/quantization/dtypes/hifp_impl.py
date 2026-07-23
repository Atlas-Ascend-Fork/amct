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

import functools

import torch


_HIFLOAT8_BACKEND_ERRORS = (RuntimeError, OSError)
_HIFLOAT8_BACKEND_REQUIRED = (
    "HiFloat8 fake quantization is unavailable. To use HiFloat8, enable "
    "native HiFloat8 cast support in torch_npu or install amct_ops."
)


@functools.lru_cache(maxsize=1)
def is_native_hifloat8_cast_available():
    """Check native HiFloat8 cast support with a minimal round trip."""
    try:
        import torch_npu
    except ImportError:
        return False

    if not hasattr(torch_npu, "hifloat8") or not hasattr(
        torch_npu, "npu_dtype_cast"
    ):
        return False

    try:
        fp_tensor = torch.zeros(1, dtype=torch.float16).npu()
        hifloat8_tensor = torch_npu.npu_dtype_cast(
            fp_tensor, torch_npu.hifloat8
        )
        torch_npu.npu_dtype_cast(
            hifloat8_tensor,
            torch.float16,
            input_dtype=torch_npu.hifloat8,
        )
    except Exception:
        return False
    return True


def _native_hifloat8_fake_quant(fp_tensor):
    """Run a native FP-to-HiFloat8-to-FP cast without fallback."""
    import torch_npu

    hifloat8_tensor = torch_npu.npu_dtype_cast(
        fp_tensor, torch_npu.hifloat8
    )
    return torch_npu.npu_dtype_cast(
        hifloat8_tensor,
        fp_tensor.dtype,
        input_dtype=torch_npu.hifloat8,
    )


def _load_amct_ops_cast():
    """Load amct_ops HiFloat8 cast functions only when requested."""
    try:
        from amct_ops.hifloat8_cast import (
            decode_from_hifloat8,
            encode_to_hifloat8,
        )
    except ImportError:
        return None
    return encode_to_hifloat8, decode_from_hifloat8


def _amct_ops_hifloat8_fake_quant(fp_tensor, encode, decode):
    """Run an amct_ops HiFloat8 round trip and restore input metadata."""
    work_dtype = (
        fp_tensor.dtype
        if fp_tensor.dtype in (torch.float16, torch.bfloat16)
        else torch.bfloat16
    )
    work_tensor = fp_tensor.to(work_dtype)
    if work_tensor.device.type != "npu":
        work_tensor = work_tensor.npu()
    codes = encode(work_tensor)
    return decode(codes, work_dtype).to(
        device=fp_tensor.device, dtype=fp_tensor.dtype
    )


@torch.no_grad()
def hifloat8_fake_quant(fp_tensor):
    """Run HiFloat8 fake quant with native cast and amct_ops fallback."""
    if is_native_hifloat8_cast_available():
        try:
            return _native_hifloat8_fake_quant(fp_tensor)
        except _HIFLOAT8_BACKEND_ERRORS:
            pass

    try:
        ops = _load_amct_ops_cast()
    except _HIFLOAT8_BACKEND_ERRORS:
        ops = None
    if ops is not None:
        try:
            return _amct_ops_hifloat8_fake_quant(fp_tensor, *ops)
        except _HIFLOAT8_BACKEND_ERRORS:
            pass

    raise RuntimeError(_HIFLOAT8_BACKEND_REQUIRED)
