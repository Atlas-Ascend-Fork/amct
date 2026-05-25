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
import torch


def encode_to_hifloat8(x: torch.Tensor) -> torch.Tensor:
    """Encode a FP16/BF16 tensor to HiFloat8 (uint8).

    Args:
        x: Input tensor on NPU, dtype must be float16 or bfloat16.

    Returns:
        uint8 tensor of the same shape containing HiFloat8 codes.

    Raises:
        RuntimeError: If input dtype is not float16 or bfloat16.

    Example:
        >>> x = torch.randn(1024, dtype=torch.float16, device="npu")
        >>> y = encode_to_hifloat8(x)          # uint8, same shape
        >>> y = torch.ops.amct.encode_to_hifloat8(x)  # equivalent
    """
    return torch.ops.amct.encode_to_hifloat8(x)


def decode_from_hifloat8(
    x: torch.Tensor,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Decode a HiFloat8 (uint8) tensor back to FP16 or BF16.

    Args:
        x: uint8 tensor on NPU containing HiFloat8 codes.
        dtype: Output dtype, float16 or bfloat16 (default: bfloat16).

    Returns:
        Decoded tensor with the requested dtype and same shape.

    Raises:
        RuntimeError: If input dtype is not uint8, or output dtype is unsupported.

    Example:
        >>> z = decode_from_hifloat8(y)                    # → bfloat16
        >>> z = decode_from_hifloat8(y, torch.float16)     # → float16
        >>> z = torch.ops.amct.decode_from_hifloat8(y)      # equivalent
    """
    return torch.ops.amct.decode_from_hifloat8(x, dtype)
