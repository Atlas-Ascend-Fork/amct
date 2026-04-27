# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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

import os
import torch
import torch_npu

_lib_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))), \
    "build", "libhifloat8_cast_ops.so")
if not os.path.exists(_lib_path):
    _lib_path = os.path.join(os.getcwd(), "build", "libhifloat8_cast_ops.so")

torch.ops.load_library(_lib_path)


def encode_to_hifloat8(x: torch.Tensor) -> torch.Tensor:
    """
    将 FP32/FP16/BF16 张量编码为 HiFloat8 格式
    
    自动根据输入数据类型选择编码模式：
    - torch.float32 → castMode=0
    - torch.float16 → castMode=1
    - torch.bfloat16 → castMode=2
    
    Args:
        x: 输入张量，支持 dtype=float32/float16/bfloat16
        
    Returns:
        uint8 张量，每个元素为 HiFloat8 编码值
        
    Raises:
        ValueError: 输入类型不支持
        
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
            f"encode_to_hifloat8 不支持输入类型 {x.dtype}，"
            f"支持类型: {supported}"
        )
    
    cast_mode = dtype_to_castmode[x.dtype]
    return torch.ops.npu.hifloat8_cast(x, cast_mode)


def decode_from_hifloat8(
    x: torch.Tensor, 
    dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """
    将 HiFloat8 格式解码为 FP32/FP16/BF16 张量
    
    Args:
        x: HiFloat8 编码的张量 (uint8)
        dtype: 输出数据类型，默认 torch.float32
            支持: torch.float32, torch.float16, torch.bfloat16
        
    Returns:
        解码后的张量，dtype 由参数指定
        
    Raises:
        ValueError: 输入类型不是 uint8 或输出类型不支持
        
    Example:
        >>> y = torch.tensor([0, 8, 136], dtype=torch.uint8, device='npu')
        >>> z = decode_from_hifloat8(y)
        >>> z.dtype
        torch.float32
    """
    if x.dtype != torch.uint8:
        raise ValueError(
            f"decode_from_hifloat8 输入必须是 uint8 类型，"
            f"实际类型: {x.dtype}"
        )
    
    dtype_to_castmode = {
        torch.float32: 3,
        torch.float16: 4,
        torch.bfloat16: 5,
    }
    
    if dtype not in dtype_to_castmode:
        supported = [str(d) for d in dtype_to_castmode.keys()]
        raise ValueError(
            f"decode_from_hifloat8 不支持输出类型 {dtype}，"
            f"支持类型: {supported}"
        )
    
    cast_mode = dtype_to_castmode[dtype]
    return torch.ops.npu.hifloat8_cast(x, cast_mode)


__all__ = [
    'encode_to_hifloat8',
    'decode_from_hifloat8',
]