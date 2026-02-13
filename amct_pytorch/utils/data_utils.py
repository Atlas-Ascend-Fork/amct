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

import torch

from amct_pytorch.utils.quant_util import pad_zero_by_group

FP4E2M1_MAX_EXP = 2

def check_linear_input_dim(input_tensor):
    """
    Function: Check if the input dimension for linear operation is between 2 and 6
    Args:
        input_tensor: Input tensor to check dimension
    """
    input_dim = len(input_tensor.shape)
    if input_dim < 2 or input_dim > 6:
        raise RuntimeError("Linear quant only support dim from 2 to 6")


def float_to_fp4e2m1(tensor):
    """
    Function: convert float16/bfloat16 tensor to float4 by pytorch
    Args:
        tensor: torch.tensor. float16/bfloat16
    Returns:
        torch.tensor. float4_e2m1
    """
    sign = torch.sign(tensor)
    absvalues = torch.abs(tensor)

    fp4e2m1_tensor = torch.zeros_like(tensor)
    fp4e2m1_tensor[absvalues <= 0.25] = 0
    fp4e2m1_tensor[(absvalues > 0.25) & (absvalues < 0.75)] = 0.5
    fp4e2m1_tensor[(absvalues >= 0.75) & (absvalues <= 1.25)] = 1.0
    fp4e2m1_tensor[(absvalues > 1.25) & (absvalues < 1.75)] = 1.5
    fp4e2m1_tensor[(absvalues >= 1.75) & (absvalues <= 2.5)] = 2.0
    fp4e2m1_tensor[(absvalues > 2.5) & (absvalues < 3.5)] = 3.0
    fp4e2m1_tensor[(absvalues >= 3.5) & (absvalues <= 5.0)] = 4.0
    fp4e2m1_tensor[absvalues > 5.0] = 6.0
    
    return (fp4e2m1_tensor * sign)


def cal_shared_exponent(input_tensor, block_size=32):
    """
    Function: cal shared exponent for MXFP4
    Args:
        input_tensor: weight tensor
        mx_dtype: MXFP4_E2M1
        block_size: block size of calculate shared exponent
    Returns:
        shared_exponent: torch.tensor
    """
    ori_shape = input_tensor.shape
    reshape_input_tensor = input_tensor.reshape(-1, ori_shape[-1])
    first_dim, _ = reshape_input_tensor.shape
    # fill 0 if the number of data is not divisible by 32
    reshape_input_tensor = pad_zero_by_group(reshape_input_tensor, block_size)
    # reshape to [first_dim, block_size]
    reshaped_tensor = reshape_input_tensor.view(first_dim, -1, block_size)
    max_values = torch.max(torch.abs(reshaped_tensor), dim=-1).values
    zero_mask = (max_values == 0)
    invalid_mask = ~torch.isfinite(max_values)

    non_zero_max_vals = torch.where(zero_mask, torch.ones_like(max_values), max_values)

    exponents = torch.floor(torch.log2(non_zero_max_vals))
    mantissas = non_zero_max_vals / torch.pow(2, exponents)
    # exp add 1 if mantissas larger than 1.75;
    shared_exponents = torch.where(mantissas > 1.75, exponents + 1, exponents) - FP4E2M1_MAX_EXP

    shared_exponents[zero_mask] = 0
    shared_exponents[invalid_mask] = torch.nan
    shared_exponents = shared_exponents.reshape(ori_shape[:-1] + ((ori_shape[-1] + block_size - 1) // block_size,))
    return shared_exponents


def scale_input_by_shared_exponents(input_tensor, shared_exponents, block_size=32):
    """
    Function: scale input by shared exponents
    Args:
        input_tensor: torch.tensor
        shared_exponents: torch.tensor
        block_size: block size of calculate shared exponent
    Return:
        torch.tensor, shape is same with input_tensor
    """
    n = input_tensor.shape[-1]
    expanded_tensor = torch.repeat_interleave(torch.pow(2, shared_exponents), repeats=block_size, dim=-1)[..., :n]
    result = input_tensor * expanded_tensor
    return result