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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


class ParamsHelper:
    """ help to get node's param info like flops, bitops. """
    @staticmethod
    def calc_conv_flops(in_channel, out_channel, k_h, k_w, out_h, out_w, group, has_bias):
        """ Calculate flops for conv node.

        Args:
            in_channel (int): channle for input fearuremap.
            out_channel (int): channle for output fearuremap.
            k_h (int): height for kernel.
            k_w (int): weight for kernel.
            out_h (int): height for output fearuremap.
            out_w (int): weight for output fearuremap.
            group (int): group for node.
            has_bias (bool): with bias or not.

        Returns:
            int: flops value.
        """
        # x2 is for add
        flops_per_pixel = 2 * (in_channel // group) * k_h * k_w - 1
        if has_bias:
            flops_per_pixel += 1
        flops = flops_per_pixel * out_h * out_w * out_channel
        return flops

    @staticmethod
    def calc_deconv_flops(in_channel, out_channel, k_h, k_w, in_h, in_w, group, has_bias):
        """Calculate flops for deconv node.

        Args:
            in_channel (int): channle for input fearuremap.
            out_channel (int): channle for output fearuremap.
            k_h (int): height for kernel.
            k_w (int): weight for kernel.
            in_h ([type]): height for input fearuremap.
            in_w ([type]): weight for input fearuremap.
            out_h (int): height for output fearuremap.
            out_w (int): weight for output fearuremap.
            group (int): group for node.
            has_bias (bool): with bias or not.

        Returns:
            int: flops value.
        """
        flops = 2 * (in_channel // group) * k_h * k_w - 1
        if has_bias:
            flops += 1
        flops = out_channel * in_h * in_w * flops
        return flops

    @staticmethod
    def calc_matmul_flops(input_shape, output_shape, has_bias):
        """ Calculate flops for matmul node, including matmul, gemm, batch_matmul and so on.

        Args:
            input_shape (list of int): shape of input fearuremap.
            output_shape (list of int): shape of output fearuremap.
            has_bias (bool): with bias or not.

        Returns:
            int: flops value.
        """
        flops_per_pixel = 2 * input_shape[-1] - 1
        if has_bias:
            flops_per_pixel += 1
        flops = flops_per_pixel * int(np.prod(output_shape)) / input_shape[0]

        return flops

    @staticmethod
    def cal_bitops(flops, wts_bits, act_bits):
        """ Calculate bitops from flops.

        Args:
            flops (float/int): flops value.
            wts_bits (int): bit of weight.
            act_bits (int): bit of activation.

        Returns:
            float/int: bitops value.
        """
        bitops = flops * wts_bits * act_bits
        return bitops
