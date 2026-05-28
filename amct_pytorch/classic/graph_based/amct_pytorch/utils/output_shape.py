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


import math
from functools import wraps
import torch
import torch.nn as nn

COMPUTE_OP_TYPES = []


def compute_op_type(op_type):
    @wraps(op_type)
    def wrapper(func):
        COMPUTE_OP_TYPES.append((op_type, func))
        return func
    return wrapper


class OutputShape(object):

    math = math

    def __init__(self, module):
        self.module = module
        self._func = getattr(module, 'output_shape', None)
        if self._func is None:
            for op_type, func in COMPUTE_OP_TYPES:
                try:
                    if module is op_type or isinstance(module, op_type):
                        self._func = func
                except TypeError as error:
                    error_info = str(error.args[0]) + str(module) + 'is unknow type'
                    raise TypeError(error_info) from None
            if self._func is None:
                error_info = str(module) + 'is unknow type'
                raise TypeError(error_info) from None

    def __call__(self, *args, **kwargs):
        if not isinstance(self.module, nn.Module):
            return self._func(*args, **kwargs)

        is_bound_method = hasattr(self._func, '__func__') and getattr(self._func, '__func__', None) is not None
        is_bound_method |= hasattr(self._func, 'im_func') and getattr(self._func, 'im_func', None) is not None
        if is_bound_method:
            return self._func(*args, **kwargs)
        else:
            return self._func(self.module, *args, **kwargs)


    @staticmethod
    @compute_op_type(nn.Conv2d)
    def conv2d(module, input_shape):
        math_module = OutputShape.math
        n, _, h_in, w_in = input_shape
        c_out = module.out_channels
        padding = module.padding
        stride = module.stride
        dilation = module.dilation
        kernel_size = module.kernel_size
        h_out = math_module.floor((h_in + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0] + 1)
        w_out = math_module.floor((w_in + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1] + 1)
        output_shape = (n, c_out, h_out, w_out)
        return output_shape

    @staticmethod
    @compute_op_type(nn.Linear)
    def linear(module, input_shape):
        n, *other, _ = input_shape
        output_shape = [n] + other + [module.out_features]
        return output_shape