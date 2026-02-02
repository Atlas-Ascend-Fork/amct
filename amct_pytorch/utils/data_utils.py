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
