#!/usr/bin/env python3
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
import os
import torch
import numpy as np
import math

np.random.seed(42)
torch.manual_seed(42)

# 获取根日志记录器
import logging
root_logger = logging.getLogger()

# 移除所有的处理器
for handler in root_logger.handlers[:]:
    root_logger.removeHandler(handler)

DEVICE = torch.device("cpu")
DATA_TYPE = torch.bfloat16


SCOP_MAP = {
    "FLOAT4_E2M1": 6,
    "FLOAT4_E1M2": 1.75,
}

FLT_EPSILON = 1.192092896e-7
def golden_float4(weight, quant_type, group_size):
    """
    Scales weights to a quantization type scope based on specified group size.

    Parameters:
    weight: Original weights to be scaled
    quant_type: Quantization type for scaling reference
    group_size: Size of each group for quantization calculations

    Returns:
    Returns tuple of (scaled weights, corresponding scaling factors)
    """
    device = weight.device
    ori_type = weight.dtype
    ori_shape = weight.shape
    weight = weight.reshape(weight.shape[0], -1)
    scale_group = calculate_scale_by_group_size(weight, quant_type, group_size)
    scale_group = torch.where(
            scale_group < FLT_EPSILON,
            torch.tensor(1.0, device=device), scale_group)
    # scaled_weights shape: (-1, group_size)
    scaled_weights = convert_to_per_group_shape(weight, group_size) / scale_group

    if quant_type == "FLOAT4_E1M2":
        quant_weight = float_cast_to_float4e1m2(scaled_weights).to(device)
    else:
        quant_weight = float_cast_to_float4e2m1(scaled_weights).to(device)
    quant_weight = (quant_weight * scale_group).to(ori_type)

    quant_weight = convert_to_ori_shape(quant_weight, ori_shape)
    return quant_weight.to(device)

def convert_to_per_group_shape(input_tensor, group_size):
    """
    Converts the input 2D tensor into a shape grouped by the specified group size. 
    If the total number of elements isn't divisible by the group size, zero-padding is applied.
 
    Parameters:
        input_tensor (Tensor): Input 2D tensor to be processed
        group_size (int): Target size for each group
 
    Returns:
        Tensor: Reshaped tensor with grouped dimensions
    """
    total_elements = input_tensor.shape[0] * input_tensor.shape[1]
    padding_needed = math.ceil(total_elements / group_size) * group_size - total_elements
    if padding_needed > 0:
        padding = torch.zeros(padding_needed, dtype=input_tensor.dtype, device=input_tensor.device)
        flattened = torch.cat([input_tensor.flatten(), padding])
    else:
        flattened = input_tensor.flatten()
 
    return flattened.reshape(-1, group_size)

def calculate_scale_by_group_size(tensor, wts_type, group_size, pad=False):
    """
    Calculates and returns the scale factor for each group based on the input tensor, weight type, and group size.

    Parameters:
    tensor: The input tensor
    wts_type: The weight type (e.g., quantization type)
    group_size: The size of each group for per-group processing

    Returns:
    A tensor of shape (-1, 1) containing the scale factor for each group.

    Exceptions:
    May throw exceptions if input parameters are invalid.
    """
    if not pad:
        tensor = convert_to_per_group_shape(tensor, group_size)
    weight_max = tensor.max(dim=-1, keepdim=True).values  # [group_num, 1]
    weight_min = tensor.min(dim=-1, keepdim=True).values  # [group_num, 1]
    boundary = torch.where(abs(weight_max) > abs(weight_min), abs(weight_max), abs(weight_min))
    scale = boundary / SCOP_MAP[wts_type]

    return scale.reshape(weight_max.shape)

def convert_to_ori_shape(input_tensor, ori_shape):
    """
    Converts a flattened tensor back to its original 2D shape by truncating excess elements.
 
    Parameters:
        input_tensor (Tensor): Input tensor to be reshaped (can be padded or extended)
        ori_shape (tuple): Original 2D shape (rows, columns) to restore
 
    Returns:
        Tensor: Reshaped tensor matching the original dimensions
    """
    total_elements = ori_shape[0] * ori_shape[1]
 
    flattened = input_tensor.flatten()[:total_elements]
    return flattened.reshape(ori_shape)

def convert_to_ori_shape(input_tensor, ori_shape):
    """
    Converts a flattened tensor back to its original 2D shape by truncating excess elements.
 
    Parameters:
        input_tensor (Tensor): Input tensor to be reshaped (can be padded or extended)
        ori_shape (tuple): Original 2D shape (rows, columns) to restore
 
    Returns:
        Tensor: Reshaped tensor matching the original dimensions
    """
    total_elements = 1
    for shape in ori_shape:
        total_elements = total_elements * shape
 
    flattened = input_tensor.flatten()[:total_elements]
    return flattened.reshape(ori_shape)


#amct fp4 trans
def float_cast_to_float4e1m2(values: torch.Tensor) -> torch.Tensor:
    """
    Function: convert float to float4_e1m2 using PyTorch
    Args:
        values: torch.Tensor, support dtype: torch.float16 or torch.bfloat16
    Returns:
        output_tensor: torch.Tensor, dtype is same with input
    """
    with torch.no_grad():
        # 创建原始值的副本用于处理
        res = torch.zeros_like(values)
        sign = torch.sign(values)
        absvalues = torch.abs(values)
        
        # 处理特殊值（NaN, +Inf, -Inf）
        nan_mask = torch.isnan(values)
        inf_mask = torch.isposinf(values)
        neg_inf_mask = torch.isneginf(values)
        
        # 应用量化规则
        res[absvalues <= 0.125] = 0
        res[(absvalues > 0.125) & (absvalues < 0.375)] = 0.25
        res[(absvalues >= 0.375) & (absvalues <= 0.625)] = 0.5
        res[(absvalues > 0.625) & (absvalues < 0.875)] = 0.75
        res[(absvalues >= 0.875) & (absvalues <= 1.125)] = 1.0
        res[(absvalues > 1.125) & (absvalues < 1.375)] = 1.25
        res[(absvalues >= 1.375) & (absvalues <= 1.625)] = 1.5
        res[absvalues > 1.625] = 1.75
        
        # 恢复符号
        res *= sign
        
        # 恢复特殊值
        res[nan_mask] = float('nan')
        res[inf_mask] = float('inf')
        res[neg_inf_mask] = float('-inf')
    
    return res.to(values.device)

def float_cast_to_float4e2m1(values: torch.Tensor) -> torch.Tensor:
    """
    Function: convert float to float4_e2m1 using PyTorch
    Args:
        values: torch.Tensor, support dtype: torch.float16 or torch.bfloat16
    Returns:
        output_tensor: torch.Tensor, dtype is same with input
    """
    with torch.no_grad():
        # 创建原始值的副本用于处理
        res = torch.zeros_like(values)
        sign = torch.sign(values)
        absvalues = torch.abs(values)
        
        # 处理特殊值（NaN, +Inf, -Inf）
        nan_mask = torch.isnan(values)
        inf_mask = torch.isposinf(values)
        neg_inf_mask = torch.isneginf(values)
        
        # 应用量化规则
        res[absvalues <= 0.25] = 0
        res[(absvalues > 0.25) & (absvalues < 0.75)] = 0.5
        res[(absvalues >= 0.75) & (absvalues <= 1.25)] = 1.0
        res[(absvalues > 1.25) & (absvalues < 1.75)] = 1.5
        res[(absvalues >= 1.75) & (absvalues <= 2.5)] = 2.0
        res[(absvalues > 2.5) & (absvalues < 3.5)] = 3.0
        res[(absvalues >= 3.5) & (absvalues <= 5.0)] = 4.0
        res[absvalues > 5.0] = 6.0
        
        # 恢复符号
        res *= sign
        
        # 恢复特殊值
        res[nan_mask] = float('nan')
        res[inf_mask] = float('inf')
        res[neg_inf_mask] = float('-inf')
    
    return res

def golden_hifloat8(in_tensor):
    hifloat8 = trans_np_float_tensor_to_hifuint8(in_tensor.to('cpu').numpy())
    float32 = trans_np_hifuint8_tensor_to_float32(hifloat8)
    return torch.from_numpy(float32)


def trans_np_float_tensor_to_hifuint8(in_tensor, round_mode="round", over_mode=True):
    shape_tensor = in_tensor.shape
    if 0 in shape_tensor:
        return np.array([]).astype(np.uint8).reshape(shape_tensor)
    multi_shape = np.prod(shape_tensor)
    in_tensor = in_tensor.reshape(multi_shape)
    if in_tensor.dtype == np.float32:
        out_tensor = _float32_to_hifuint8(in_tensor, round_mode, over_mode)
    else:
        out_tensor = _float16_to_hifuint8(in_tensor, round_mode, over_mode)
    out_tensor = out_tensor.astype(np.uint8)
    out_tensor = out_tensor.reshape(shape_tensor)
    return out_tensor

def trans_np_hifuint8_tensor_to_float32(in_tensor):
    shape_tensor = in_tensor.shape
    if 0 in shape_tensor:
        return np.array([]).astype(np.float32).reshape(shape_tensor)
    multi_shape = np.prod(shape_tensor)
    in_tensor = in_tensor.reshape(multi_shape)
    out_tensor = _hifuint8_to_float(in_tensor)
    out_tensor = out_tensor.reshape(shape_tensor).astype(np.float32)
    return out_tensor

def cvt_float16_to_hifuint8(x, round_mode="round", over_mode=True):
    Ec = 0
    over_value = 1.25 * pow(2.0, 15 + Ec)
    sign = False
    sign_int_value = 0
    if x < 0.0:
        sign = True
        sign_int_value = 128
    x_abs = math.fabs(x)
    if np.isinf(x) or x_abs >= over_value:
        #备注 2^15 = 32768
        if sign:
            if over_mode:
                #备注 b11101111 = 239
                return 239
            else:
                #备注 b11101110 = 238
                return 238
        else:
            if over_mode:
                #备注 b01101111 = 111
                return 111
            else:
                #备注 b01101110 = 110
                return 110
    if np.isnan(x):
        if over_mode:
            #b10000000
            return 128
        else:
            return 0
    if x_abs == 0.0:
        return 0
    exponent = math.floor(math.log2(x_abs))

    if round_mode == "hybrid":
        if abs(exponent) < 4:
            cut_bit_type = "TA"
        else:
            cut_bit_type = "SSR"
    elif round_mode == "round":
        cut_bit_type = "TA"
    elif round_mode == "storound":
        cut_bit_type = "SSR"
    else:
        cut_bit_type = "TA"

    #precheck
    fraction_int = int(x_abs * pow(2, 10)*pow(2, -exponent) - pow(2, 10))
    dot_hif8_value, exponent_hif8_bits, fraction_hif8_bits = _get_hif8_fraction_bits_number(exponent)
    if cut_bit_type == "TA":
        carry_exp_status, hif8_frac_value = _fp16_ta_round_to_hif8(fraction_int, fraction_hif8_bits, exponent)
    elif cut_bit_type == "SSR":
        carry_exp_status, hif8_frac_value = _fp16_ssr_round_to_hif8(fraction_int, fraction_hif8_bits, exponent)
    else:
        print(f"unknow round type")
        return 0
    if carry_exp_status:
        exponent += 1
        dot_hif8_value, exponent_hif8_bits, fraction_hif8_bits_new = _get_hif8_fraction_bits_number(exponent)
        fraction_hif8_bits = fraction_hif8_bits_new
    if fraction_hif8_bits == -1:
        #over flow
        if sign:
            if over_mode:
                return 239
            else:
                return 238
        else:
            if over_mode:
                return 111
            else:
                return 110
    if exponent < -23:
        #zero b00000000
        return 0
    if exponent < 0:
        sig_exp = 1
    else:
        sig_exp = 0
    if dot_hif8_value == 0:
        if exponent <= -23:
            return 0
        else:
            return sign_int_value + exponent + 23
    elif dot_hif8_value == 1:
        #d0
        dot_int_value = dot_hif8_value << 3
        hif8_int_value = sign_int_value + dot_int_value + hif8_frac_value
    else:
        abs_exponent = abs(exponent)
        abs_exponent = abs_exponent - pow(2, exponent_hif8_bits - 1)
        exponent_int_value = abs_exponent << fraction_hif8_bits
        sig_exp = sig_exp << (exponent_hif8_bits - 1 + fraction_hif8_bits)
        dot_int_value = dot_hif8_value << 3
        hif8_int_value = sign_int_value + dot_int_value + sig_exp + exponent_int_value + hif8_frac_value
    return hif8_int_value

def cvt_float32_to_hifuint8(x, round_mode="round", over_mode=True):
    sign = False
    sign_int_value = 0
    x_abs = math.fabs(x)
    Ec = 0
    over_value = 1.25 * pow(2.0, 15 + Ec)
    if x < 0.0:
        sign = True
        sign_int_value = 128
    if np.isinf(x) or x_abs >= over_value:
        #备注 2^15 = 32768
        if sign:
            if over_mode:
                #备注 b11101111 = 239
                return 239
            else:
                #备注 b11101110 = 238
                return 238
        else:
            if over_mode:
                #备注 b01101111 = 111
                return 111
            else:
                #备注 b01101110 = 110
                return 110
    if np.isnan(x):
        if over_mode:
            #b10000000
            return 128
        else:
            return 0
    if x_abs == 0.0:
        return 0
    exponent = math.floor(math.log2(x_abs))
    if round_mode == "hybrid":
        if abs(exponent) < 4:
            cut_bit_type = "TA"
        else:
            cut_bit_type = "SSR"
    elif round_mode == "round":
        cut_bit_type = "TA"
    elif round_mode == "storound":
        cut_bit_type = "SSR"
    else:
        cut_bit_type = "TA"
    #precheck
    fraction_int = int(x_abs * pow(2, 23)*pow(2, -exponent) - pow(2, 23))
    dot_hif8_value, exponent_hif8_bits, fraction_hif8_bits = _get_hif8_fraction_bits_number(exponent)
    if cut_bit_type == "TA":
        carry_exp_status, hif8_frac_value = _fp32_ta_round_to_hif8(fraction_int, fraction_hif8_bits, exponent)
    elif cut_bit_type == "SSR":
        carry_exp_status, hif8_frac_value = _fp32_ssr_round_to_hif8(fraction_int, fraction_hif8_bits, exponent)
    else:
        print(f"unknow round type")
        return 0
    if carry_exp_status:
        exponent += 1
        dot_hif8_value, exponent_hif8_bits, fraction_hif8_bits_new = _get_hif8_fraction_bits_number(exponent)
        fraction_hif8_bits = fraction_hif8_bits_new
    if exponent < -23:
        #zero b00000000
        return 0
    if exponent < 0:
        sig_exp = 1
    else:
        sig_exp = 0
    if dot_hif8_value <= 0:
        #dml
        if exponent <= -23:
            return 0
        else:
            return sign_int_value + exponent + 23
    elif dot_hif8_value == 1:
        #d0
        dot_int_value = dot_hif8_value << 3
        hif8_int_value = sign_int_value + dot_int_value + hif8_frac_value
    else:
        abs_exponent = abs(exponent)
        abs_exponent = abs_exponent - pow(2, exponent_hif8_bits - 1)
        exponent_int_value = abs_exponent << fraction_hif8_bits
        sig_exp = sig_exp << (exponent_hif8_bits - 1 + fraction_hif8_bits)
        dot_int_value = dot_hif8_value << 3
        hif8_int_value = sign_int_value + dot_int_value + sig_exp + exponent_int_value + hif8_frac_value
    return hif8_int_value

def cvt_hifuint8_to_float(x, over_mode=True):
    if x == 0:
        return float(0)
    elif x == 128:
        if over_mode:
            return np.nan
        else:
            return float(0)
    elif x == 239:
        if over_mode:
            return -np.inf
        else:
            return -32768
    elif x == 111:
        if over_mode:
            return np.inf
        else:
            return 32768
    else:
        if x >= 128:
            sign = -1.0
        else:
            sign = 1.0
        dot_4_bits = x & 120 #b01111000 = 120
        dot_4_value = dot_4_bits >> 3
        if dot_4_value >= 12:
            #备注 b1100 =12 D4
            exponet = x & 30 #b00011110 = 30
            exponet_int = exponet >> 1
            if exponet_int >= 8:
                #备注 b1000 = 8
                exponet_value = -exponet_int
            else:
                exponet_value = exponet_int + 8

            fra_int = x & 1 #b00000001
            m_value = 1.0 + fra_int * 0.5
        elif dot_4_value >= 8:
            #备注 b1000 =8 D3
            exponet = x & 28 #b00011100 = 28
            exponet_int = exponet >> 2
            if exponet_int >= 4:
                #备注 b100 = 4
                exponet_value = -exponet_int
            else:
                exponet_value = exponet_int + 4
            fra_int = x & 3 #b00000011
            m_value = 1.0 + fra_int * 0.25
        elif dot_4_value >= 4:
            #备注 b0100 =8 D2
            exponet = x & 24  # b00011000 = 24
            exponet_int = exponet >> 3
            if exponet_int >= 2:
                #备注 b10 = 2
                exponet_value = -exponet_int
            else:
                exponet_value = exponet_int + 2
            fra_int = x & 7  # b00000111
            m_value = 1.0 + fra_int * 0.125
        elif dot_4_value >= 2:
            #备注 b0010 =2 D1
            exponet = x & 8 # b00001000 = 8
            exponet_sign = exponet >> 3
            if exponet_sign >= 1:
                #备注 b10 = 2
                exponet_value = -1
            else:
                exponet_value = 1
            fra_int = x & 7  # b00000111
            m_value = 1.0 + fra_int * 0.125
        elif dot_4_value == 1:
            #d0
            exponet_value = 0
            fra_int = x & 7  # b00000111
            m_value = 1.0 + fra_int * 0.125
        elif dot_4_value == 0:
            #dml
            m_value = 1
            exponet_value = (x & 7) - 23  # b00000111 = 7
        else:
            print("error,dot error")
            m_value = 0.0
            exponet_value = 0
        return sign*pow(2.0, exponet_value)*m_value

_float16_to_hifuint8 = np.vectorize(
    cvt_float16_to_hifuint8, excluded=["round_mode", "over_mode"]
)

_float32_to_hifuint8 = np.vectorize(
    cvt_float32_to_hifuint8, excluded=["round_mode", "over_mode"]
)

_hifuint8_to_float = np.vectorize(
    cvt_hifuint8_to_float, excluded=["over_mode"]
)
def _get_hif8_fraction_bits_number(exponent):
    #备注 return dot value(4bits), exponent size, fraction size
    if exponent < -22:
        #zero
        return -1, 3, 0
    if -22 <= exponent < -15:
        #dml
        return 0, 3, 0
    if exponent == 0:
        #d0
        return 1, 0, 3
    if abs(exponent) == 1:
        #d1
        return 2, 1, 3
    if 2 <= abs(exponent) <= 3:
        #d2
        return 4, 2, 3
    if 4 <= abs(exponent) <= 7:
        #d3
        return 8, 3, 2
    if 8 <= abs(exponent) <= 15:
        #d4
        return 12, 4, 1
    if exponent > 15:
        #over flow
        return 12, 4, -1

def _fp32_ta_round_to_hif8(fraction32_int, hif8_bits_num, exponent):
    if exponent == -23:
        return True, 0
    #fp32 fraction is 23,keep hif8_bits_num + 1 bits
    hif8_value_tmp = fraction32_int >> (23 - (hif8_bits_num+1))
    if hif8_value_tmp == pow(2, hif8_bits_num + 1) - 1:
        #carry exponent
        return True, 0
    elif hif8_value_tmp == 0:
        #zero
        return False, 0
    elif hif8_value_tmp % 2 == 1:
        #carrys bits
        hif8_value_tmp += 1
        return False, hif8_value_tmp >> 1
    else:
        return False, hif8_value_tmp >> 1

def _fp32_ssr_round_to_hif8(fraction32_int, hif8_bits_num, exponent):
    t14_mask = 16383  # b11111111111111
    if exponent == -23:
        f14_values = (fraction32_int >> 10) + 8192 #10 0000 0000 0000
        t14_values = fraction32_int & t14_mask
        hif8_value = 0

    else:
        hif8_value = fraction32_int >> (23 - hif8_bits_num)
        f14_t14 = fraction32_int - (hif8_value << (23 - hif8_bits_num))
        f14_values = f14_t14 >> (23 - hif8_bits_num - 14)
        t14_values = f14_t14 & t14_mask
    if f14_values >= t14_values:
        #carry bits
        if hif8_value == pow(2, hif8_bits_num) - 1:
            #carry exponent:
            return True, 0
        else:
            hif8_value += 1
            return False, hif8_value
    else:
        return False, hif8_value


def _fp16_ta_round_to_hif8(fraction16_int, hif8_bits_num, exponent):
    if exponent == -23:
        return True, 0
    #fp16 fraction is 10,keep hif8_bits_num + 1 bits
    hif8_value_tmp = fraction16_int >> (10 - (hif8_bits_num+1))
    if hif8_value_tmp == pow(2, hif8_bits_num+1) - 1:
        #carry exponent
        return True, 0
    elif hif8_value_tmp == 0:
        #zero
        return False, 0
    elif hif8_value_tmp % 2 == 1:
        #carrys bits
        hif8_value_tmp += 1
        return False, hif8_value_tmp >> 1
    else:
        return False, hif8_value_tmp >> 1

def _fp16_ssr_round_to_hif8(fraction16_int, hif8_bits_num, exponent):
    t2_mask = 1 #b1
    t2_values = (fraction16_int & t2_mask) * 2 + 1
    if exponent == -23:
        f2_values = 2 + fraction16_int >> 9
        hif8_value = 0
    else:
        hif8_value = fraction16_int >> (10 - hif8_bits_num)
        f2_t2 = fraction16_int - (hif8_value << (10 - hif8_bits_num))
        f2_values = f2_t2 >> (10 - hif8_bits_num - 2)
    if f2_values >= t2_values:
        #carry bits
        if hif8_value == pow(2, hif8_bits_num):
            #carry exponent:
            return True, 0
        else:
            hif8_value += 1
            return False, hif8_value
    else:
        return False, hif8_value

def float_to_float4e2m1(values: np.ndarray) -> np.ndarray:
    """
    Function: convert float to mxfp4
    Args:
        values: np.ndarray, support dtype:np.float16 or ml_dtypes.bfloat16
    Returns:
        output_tensor: np.ndarray, dtype is same with values (np.float16 or ml_dtypes.bfloat16)
    """
    shared_exponents = cal_shared_exponent(values)
    actual_exponents = shared_exponents.astype(np.int8) - 127
    fp4e2m1_values = scale_input_by_shared_exponents(values, 2**(-actual_exponents.astype(values.dtype)))
    # convert to fp4e2m1 and convert back
    fp4e2m1_values = fp4e2m1_values.astype(values.dtype)
    float_values = scale_input_by_shared_exponents(fp4e2m1_values, 2**(actual_exponents.astype(values.dtype)))
    return float_values


def cal_shared_exponent(input_array, block_size=32):
    """
    Function: cal shared exponent for MXFP4
    Args:
        input_array: weight tensor
        block_size: block size of calculate shared exponent
    Returns:
        shared_exponent: numpy array
    """
    ori_shape = input_array.shape
    reshape_input_array = input_array.reshape(-1, ori_shape[-1])
    first_dim, last_dim = reshape_input_array.shape
    # fill 0 if the number of data is not divisible by 32
    pad = (block_size - (last_dim % block_size)) % block_size
    reshape_input_array = np.pad(reshape_input_array, ((0, 0), (0, pad)), 'constant')
    # reshape to [first_dim, block_size]
    reshaped_array = reshape_input_array.reshape(first_dim, -1, block_size)
    max_values = np.abs(reshaped_array).max(axis=-1)
    zero_mask = (max_values == 0)
    non_zero_max_vals = np.where(zero_mask, np.ones_like(max_values), max_values)
    exponents = np.floor(np.log2(non_zero_max_vals))
    mantissas = non_zero_max_vals / (2 ** exponents)
    shared_exponents = np.where(mantissas > 1.75, exponents + 1, exponents).astype(np.uint8)

    shared_exponents[zero_mask] = 0
    shared_exponents = shared_exponents + 127
    return shared_exponents.reshape(ori_shape[:-1] + ((ori_shape[-1] + block_size - 1) // block_size,))


def scale_input_by_shared_exponents(input_tensor, shared_exponents, block_size=32):
    """
    Function: scale input by shared exponents
    Args:
        input_tensor: numpy array
        shared_exponents: numpy array
        block_size: block size of calculate shared exponent
    Return:
        numpy array, shape is same with input_tensor
    """
    n = input_tensor.shape[-1]
    expanded_tensor = np.repeat(shared_exponents, block_size, axis=-1)[..., :n]
    result = input_tensor * expanded_tensor
    return result

def convert_golden_inverse_conversion(input_tensor, quant_type):
    #return input_tensor
    dtype = input_tensor.dtype

    # if quant_type == 'MXFP4_E2M1':
    #     input_tensor = torch.from_numpy(float_to_float4e2m1(input_tensor.to('cpu').numpy()))
    if quant_type in ('FLOAT4_E2M1', 'FLOAT4_E1M2'):
        input_tensor = golden_float4(input_tensor, quant_type, 128)
    # if quant_type == 'HIFLOAT8':
    #     input_tensor = golden_hifloat8(input_tensor)

    return input_tensor.to(dtype)

def set_model_data_distribution(model, distribution_type='ones', quant_type='MXFP4_E2M1'):
    for name, param in model.named_parameters():
        param.data = param.data.to(DATA_TYPE).to(DEVICE)
        if "weight" not in name: continue
        if distribution_type == 'ones':
            param.data = torch.ones_like(param.data).to(DATA_TYPE)
            param.data = convert_golden_inverse_conversion(param.data, quant_type).to(DATA_TYPE).to(DEVICE)
        elif distribution_type == 'zeros':
            param.data = torch.zeros_like(param.data).to(DATA_TYPE)
            param.data = convert_golden_inverse_conversion(param.data, quant_type).to(DATA_TYPE).to(DEVICE)
        elif distribution_type == 'normal':
            shape = param.data.shape
            param.data = torch.normal(mean=0, std=1, size=shape).to(DATA_TYPE)
            param.data = convert_golden_inverse_conversion(param.data, quant_type).to(DATA_TYPE).to(DEVICE)
        elif distribution_type == 'rand':
            param.data = torch.rand_like(param.data).to(DATA_TYPE)
            param.data = convert_golden_inverse_conversion(param.data, quant_type).to(DATA_TYPE).to(DEVICE)

    for _, buffer in model.named_buffers():
        buffer.data = buffer.data.to(DATA_TYPE).to(DEVICE)
        if "weight" not in name: continue
        if distribution_type == 'ones':
            buffer.data = torch.ones_like(buffer.data).to(DATA_TYPE)
            buffer.data = convert_golden_inverse_conversion(param.data, quant_type).to(DATA_TYPE).to(DEVICE)
        elif distribution_type == 'zeros':
            buffer.data = torch.zeros_like(buffer.data).to(DATA_TYPE)
            buffer.data = convert_golden_inverse_conversion(param.data, quant_type).to(DATA_TYPE).to(DEVICE)
        elif distribution_type == 'normal':
            shape = buffer.data.shape
            buffer.data = torch.normal(mean=0, std=1, size=shape).to(DATA_TYPE)
            buffer.data = convert_golden_inverse_conversion(param.data, quant_type).to(DATA_TYPE).to(DEVICE)
        elif distribution_type == 'rand':
            buffer.data = torch.rand_like(buffer.data).to(DATA_TYPE)
            buffer.data = convert_golden_inverse_conversion(param.data, quant_type).to(DATA_TYPE).to(DEVICE)

def similarity(data0, data1):
    data0_nan = np.isnan(data0)
    data0[data0_nan] = 1
    data1_nan = np.isnan(data1)
    data1[data1_nan] = 1
    similar = similarity_1 = np.sum(np.multiply(data0.astype(np.float64), data1.astype(np.float64)).astype(np.float64))\
                    /(np.sqrt(np.sum(data0.astype(np.float64)**2))\
                    *np.sqrt(np.sum(data1.astype(np.float64)**2)))*100
    if (data0 == data1).all():
        similar = 100
    if np.isnan(similar) or np.isinf(similar):
        data0 = np.divide(data0,np.power(10,38))
        data1 = np.divide(data1,np.power(10,38))
        similar = similarity_1 = np.sum(np.multiply(data0, data1).astype(np.float64))\
                    /(np.sqrt(np.sum(data0.astype(np.float64)**2))\
                    *np.sqrt(np.sum(data1.astype(np.float64)**2)))*100
        if np.isnan(similar) or np.isinf(similar):
            data0 = np.divide(data0,np.power(10,38))
            data1 = np.divide(data1,np.power(10,38))
            similar = similarity_1 = np.sum(np.multiply(data0, data1).astype(np.float64))\
                    /(np.sqrt(np.sum(data0.astype(np.float64)**2))\
                    *np.sqrt(np.sum(data1.astype(np.float64)**2)))*100
    if np.isnan(similar):
        similar = 0

    return similar


def set_data_distribution(data_shape, distribution_type='ones', quant_type='MXFP4_E2M1'):
    if distribution_type == 'ones':
        data = torch.ones(data_shape).to(DATA_TYPE).to(DEVICE)
    elif distribution_type == 'zeros':
        data = torch.zeros(data_shape).to(DATA_TYPE).to(DEVICE)
    elif distribution_type == 'normal':
        data = torch.normal(mean=0, std=1, size=data_shape).to(DATA_TYPE).to(DEVICE)
    elif distribution_type == 'rand':
        data = torch.rand(data_shape).to(DATA_TYPE).to(DEVICE)
    data = convert_golden_inverse_conversion(data, quant_type).to(DATA_TYPE).to(DEVICE)
    return data

