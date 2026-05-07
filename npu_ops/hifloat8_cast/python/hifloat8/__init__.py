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
"""
HiFloat8 encode/decode interface

Provides simple Python interface with automatic dtype inference.

Usage:
    from hifloat8 import encode_to_hifloat8, decode_from_hifloat8
    
    y = encode_to_hifloat8(x_fp32)  # Auto-detect FP32->HiF8
    z = decode_from_hifloat8(y)     # Default output FP32
"""
__all__ = [
    'encode_to_hifloat8',
    'decode_from_hifloat8',
]

import os
import torch
import torch_npu

_lib_path = os.path.join(os.path.dirname(__file__), "../../build/libhifloat8_cast_ops.so")
if not os.path.exists(_lib_path):
    _lib_path = os.path.join(os.getcwd(), "build/libhifloat8_cast_ops.so")

torch.ops.load_library(_lib_path)


def encode_to_hifloat8(x: torch.Tensor) -> torch.Tensor:
    """
    Encode FP32/FP16/BF16 tensor to HiFloat8 format
    
    Automatically selects encoding mode based on input dtype:
    - torch.float32 -> castMode=0
    - torch.float16 -> castMode=1
    - torch.bfloat16 -> castMode=2
    
    Args:
        x: Input tensor, supports dtype=float32/float16/bfloat16
        
    Returns:
        uint8 tensor, each element is HiFloat8 encoded value
        
    Raises:
        ValueError: Input dtype not supported
        
    Example:
        >>> x = torch.randn(1000, dtype=torch.float32, device='npu')
        >>> y = encode_to_hifloat8(x)
        >>> y.dtype
        torch.uint8
    """
    dtype_to_castmode = {
        torch.float32: 0,
        torch.float16: 1,
        torch.bfloat16: 2,
    }
    
    if x.dtype not in dtype_to_castmode:
        supported = [str(d) for d in dtype_to_castmode.keys()]
        raise ValueError(
            f"encode_to_hifloat8 does not support input dtype {x.dtype}, "
            f"supported dtypes: {supported}"
        )
    
    cast_mode = dtype_to_castmode[x.dtype]
    return torch.ops.npu.hifloat8_cast(x, cast_mode)


def decode_from_hifloat8(
    x: torch.Tensor, 
    dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """
    Decode HiFloat8 format to FP32/FP16/BF16 tensor
    
    Args:
        x: HiFloat8 encoded tensor (uint8)
        dtype: Output dtype, default torch.float32
            Supports: torch.float32, torch.float16, torch.bfloat16
        
    Returns:
        Decoded tensor with specified dtype
        
    Raises:
        ValueError: Input dtype is not uint8 or output dtype not supported
        
    Example:
        >>> y = torch.tensor([0, 8, 136], dtype=torch.uint8, device='npu')
        >>> z = decode_from_hifloat8(y)
        >>> z.dtype
        torch.float32
    """
    if x.dtype != torch.uint8:
        raise ValueError(
            f"decode_from_hifloat8 input must be uint8 dtype, "
            f"actual dtype: {x.dtype}"
        )
    
    dtype_to_castmode = {
        torch.float32: 3,
        torch.float16: 4,
        torch.bfloat16: 5,
    }
    
    if dtype not in dtype_to_castmode:
        supported = [str(d) for d in dtype_to_castmode.keys()]
        raise ValueError(
            f"decode_from_hifloat8 does not support output dtype {dtype}, "
            f"supported dtypes: {supported}"
        )
    
    cast_mode = dtype_to_castmode[dtype]
    return torch.ops.npu.hifloat8_cast(x, cast_mode)
