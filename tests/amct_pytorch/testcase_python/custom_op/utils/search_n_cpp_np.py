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
import torch

accum_inputs = None

def kernel_func(inputs, do_cali=False, deq_scale=None):
    """
    device is same;
    shape is []
    """
    # print('*-'*20, 'search_n_cpp', '*-'*20)
    global accum_inputs
    # data in [c, *]
    channle_num = inputs.shape[1]
    inputs = inputs.transpose(0,1).reshape([channle_num, -1])
    if accum_inputs is None:
        accum_inputs = inputs
    else:
        # print('inputs', inputs.shape, inputs.dim())
        # print('accum_inputs', accum_inputs.shape, accum_inputs.dim())
        accum_inputs = torch.cat((inputs, accum_inputs), 1)

    if not do_cali:
        return torch.tensor(0.0), torch.tensor(0)
    # print('inputs', accum_inputs.shape, accum_inputs.dim())

    channel_wise = False
    if deq_scale.shape[0] > 1:
        channel_wise = True
        deq_scale = deq_scale.reshape([-1, 1])
        # deq_scale = deq_scale.reshape([1, -1, 1, 1])
    # print('deq_scale', deq_scale.shape)


    # c++
    quanti_bit=32
    clamp_min = -pow(2.0, quanti_bit-1)
    clamp_max = pow(2.0, quanti_bit-1) - 1
    s32_data = torch.round(torch.div(accum_inputs, deq_scale))
    s32_data = torch.clamp(s32_data, clamp_min, clamp_max)
    # print('s32_data', s32_data)

    shift_bits = 16
    s16_clamp_min = -pow(2.0, shift_bits-1)
    s16_clamp_max = pow(2.0, shift_bits-1) - 1
    errs_list = [0] *16
    for shift_bit in range(1, shift_bits + 1):
        shift_factor = pow(2.0, shift_bit)
        s16_data = torch.floor(torch.div(s32_data, shift_factor))
        s16_data = torch.clamp(s16_data, s16_clamp_min, s16_clamp_max)
        s32_quant_data = torch.mul(s16_data, shift_factor)
        if channel_wise:
            # err = torch.sum(torch.pow(s32_data - s32_quant_data, 2), dim=1)
            err = torch.norm(s32_data - s32_quant_data, dim=1)
        else:
            # err = torch.sum(torch.pow(s32_data - s32_quant_data, 2))
            err = torch.norm(s32_data - s32_quant_data)
        # print('err', err)
        errs_list[-shift_bit] = err.reshape([1, -1])
        # errs_list[shift_bit-1] = err.reshape([1, -1])

    errs = torch.cat(errs_list, 0)
    # print('errs', errs.shape, errs)
    best_shift_bit = 16 - torch.argmin(errs, 0) # argmin, 同等大小取值后面的
    # best_shift_bit = torch.argmin(errs, 0) + 1
    # print('best_shift_bit', best_shift_bit)
    best_shift_bit = best_shift_bit.to(torch.int8)
    accum_inputs = None
    return torch.tensor(1), best_shift_bit

def test_conv_channel_wise():
    inputs = torch.rand([2, 3, 4, 5])
    scale_weight = torch.tensor([0.0005, 0.005, 0.05])
    scale_data = torch.tensor(0.001)

    flag, shift_bit = kernel_func(inputs, False, scale_weight*scale_data)
    flag, shift_bit = kernel_func(inputs, True, scale_weight*scale_data)


def test_conv():
    inputs = torch.rand([20, 30, 40, 50])
    scale_weight = torch.tensor([0.005])
    scale_data = torch.tensor(0.001)

    flag, shift_bit = kernel_func(inputs, False, scale_weight*scale_data)
    flag, shift_bit = kernel_func(inputs, True, scale_weight*scale_data)

def test_fc():
    inputs = torch.rand([20, 30])
    scale_weight = torch.tensor([0.0004])
    scale_data = torch.tensor(0.001)

    flag, shift_bit = kernel_func(inputs, False, scale_weight*scale_data)
    flag, shift_bit = kernel_func(inputs, True, scale_weight*scale_data)


if __name__ == "__main__":
    # test_conv_channel_wise()
    # test_conv()
    test_fc()


