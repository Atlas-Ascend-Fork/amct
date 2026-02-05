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
__all__ = ['DecomposeMode', 'tensor_decomposition']

import os
import ctypes
from enum import Enum
from enum import unique
import numpy as np
from ... import lib # pylint: disable=E0402

FIRST = 'first'
LAST = 'last'


@unique
class DecomposeMode(Enum):
    """Enumeration class indicating different decomposition mode"""
    UNCHANGE = 0  # do not decompose
    FCFK = 1      # decompose First Channel and First Kernel
    FCSK = 2      # decompose First Channel and Second Kernel
    SCFK = 3      # decompose Second Channel and First Kernel
    SCSK = 4      # decompose Second Channel and Second Kernel


class ConvInfoCtypes(ctypes.Structure): # pylint: disable=R0903
    """Class for Python and C++ interaction"""
    _fields_ = [('in_channel', ctypes.c_int),
                ('out_channel', ctypes.c_int),
                ('kernel_size_h', ctypes.c_int),
                ('kernel_size_w', ctypes.c_int),
                ('stride_h', ctypes.c_int),
                ('stride_w', ctypes.c_int),
                ('group', ctypes.c_int),
                ('dilation_h', ctypes.c_int),
                ('dilation_w', ctypes.c_int)]


class ConvInfo: # pylint: disable=R0902, R0903
    """attributes of conv2d layer"""
    def __init__(self, in_channel, out_channel, # pylint: disable=R0913
                 kernel_size, stride, group, dilation):
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.group = group

        if isinstance(kernel_size, tuple):
            self.kernel_size_h = kernel_size[0]
            self.kernel_size_w = kernel_size[1]
        else:
            self.kernel_size_h = self.kernel_size_w = kernel_size

        if isinstance(stride, tuple):
            self.stride_h = stride[0]
            self.stride_w = stride[1]
        else:
            self.stride_h = self.stride_w = stride

        if isinstance(dilation, tuple):
            self.dilation_h = dilation[0]
            self.dilation_w = dilation[1]
        else:
            self.dilation_h = self.dilation_w = dilation

        self.conv_info_ctype = ConvInfoCtypes(self.in_channel,
                                              self.out_channel,
                                              self.kernel_size_h,
                                              self.kernel_size_w,
                                              self.stride_h,
                                              self.stride_w,
                                              self.group,
                                              self.dilation_h,
                                              self.dilation_w)


def fast_filter_conv(info):
    """fast filter by ConvInfo

    param info: Class ConvInfo

    return: bool
            if true, do not decompose
    """
    lib_dir = os.path.dirname(lib.__file__)
    cpp_lib = ctypes.cdll.LoadLibrary(
        os.path.join(lib_dir, "libtensor_decompose.so")
    )
    is_filter = cpp_lib.FastFilterConv(info.conv_info_ctype)
    return is_filter


def get_rank(singular_value, info):
    """get rank based on singular value

    param singular_value: 1darray
    param info: Class ConvInfo correspond to singular value

    return: int
            in range [0, len(singular_value)]
    """
    if len(singular_value.shape) != 1:
        raise ValueError('singular value must be 1darray')

    c_double_p = ctypes.POINTER(ctypes.c_double)
    data_p = singular_value.ctypes.data_as(c_double_p)
    carray_length = ctypes.c_int(len(singular_value))
    lib_dir = os.path.dirname(lib.__file__)
    cpp_lib = ctypes.cdll.LoadLibrary(
        os.path.join(lib_dir, "libtensor_decompose.so")
    )
    rank = cpp_lib.GetRank(info.conv_info_ctype, data_p, carray_length)
    return rank


def unfold(tensor, dim):
    """unfold tensor alone the dim, dimension of the tensor must be more than 2

    param tensor: ndarray
    param dim: int

    return: 2darray
    """
    if len(tensor.shape) < 2:
        raise ValueError('dimension of input tensor must be more than 1')
    if dim >= len(tensor.shape):
        raise ValueError(
            'input dim must be less than dimension of input tensor'
        )

    axis = list(range(len(tensor.shape)))
    axis[0], axis[dim] = axis[dim], axis[0]

    return tensor.transpose(*axis).reshape(tensor.shape[dim], -1)


def tensor_dim_dot(tensor, matrix, dim):
    """product of a tensor and a matrix at the specified dim

    param tensor: ndarray
    param matrix: 2darray
    param dim: int

    return: ndarray with the same dimension of tensor
    """
    if len(tensor.shape) < 2:
        raise ValueError('dimension of input tensor must be more than 1')
    if dim >= len(tensor.shape):
        raise ValueError(
            'input dim must be less than dimension of input tensor'
        )
    if len(matrix.shape) != 2:
        raise ValueError('dimension of input matrix must be 2')
    if matrix.shape[1] != tensor.shape[dim]:
        raise ValueError(
            'shapes {0} and {1} not aligned in dim-{2} '.format(
                matrix.shape, tensor.shape, dim
            ))

    res = matrix.dot(unfold(tensor, dim))

    res_shape = list(tensor.shape)
    res_shape[0], res_shape[dim] = res_shape[dim], res_shape[0]
    res_shape = res_shape[1:]

    axis = list(range(len(tensor.shape)))
    axis[0], axis[dim] = axis[dim], axis[0]

    return res.reshape(-1, *res_shape).transpose(*axis)


def tensor_decomposition(tensor, stride=1, group=1, dilation=1): # pylint: disable=R0914
    """decompose 4d tensor to 2 samll tensors,
       if tensor can not decompose, return DecomposeMode.UNCHANGE

    param tensor: ndarray
                  conv2d weight has shape of (Cout, Cin, KH, KW)
    param stride: int or tuple(int, int)
                  stride of conv2d
    param group: int
                 group of cov2d
    param dilation: int or tuple(int, int)
                    dilation of cov2d

    return: dict of ndarray
    """
    if len(tensor.shape) != 4:
        raise ValueError(
            'weight tensor must have shape (Cout, Cin, KH, KW)'
        )

    if np.isinf(np.sum(tensor)) or np.isnan(np.sum(tensor)):
        raise ValueError(
            'weight tensor contains invalid number(inf or nan)'
        )

    in_channel = tensor.shape[1] * group
    out_channel = tensor.shape[0]
    kernel_size = (tensor.shape[2], tensor.shape[3])

    info = ConvInfo(in_channel, out_channel,
                    kernel_size, stride, group, dilation)

    mode = DecomposeMode(fast_filter_conv(info))

    res = {
        'mode': mode,
        FIRST: None,
        LAST: None
    }

    if mode == DecomposeMode.UNCHANGE:
        return res

    if mode in [DecomposeMode.FCFK, DecomposeMode.FCSK]:
        tensor = tensor.transpose(1, 0, 2, 3)
    if mode in [DecomposeMode.FCSK, DecomposeMode.SCSK]:
        tensor = tensor.transpose(0, 1, 3, 2)

    shape = tensor.shape
    tensor = tensor.reshape(shape[0], shape[1] * shape[2], shape[3])
    _, singular_value, _ = np.linalg.svd(unfold(tensor, 1))
    singular_value = singular_value.astype(np.float64)
    rank = get_rank(singular_value, info)
    matrix_u, _, _ = np.linalg.svd(unfold(tensor, 1))
    portion = matrix_u[:, :rank]
    core = tensor_dim_dot(tensor, portion.transpose(), 1)

    if mode == DecomposeMode.FCFK:
        res[FIRST] = np.expand_dims(core.transpose(1, 0, 2), axis=2)
        res[LAST] = np.expand_dims(portion.reshape(out_channel, -1, rank)
                                     .transpose(0, 2, 1), axis=3)
    if mode == DecomposeMode.FCSK:
        res[FIRST] = np.expand_dims(core.transpose(1, 0, 2), axis=3)
        res[LAST] = np.expand_dims(portion.reshape(out_channel, -1, rank)
                                     .transpose(0, 2, 1), axis=2)
    if mode == DecomposeMode.SCFK:
        res[FIRST] = np.expand_dims(portion.transpose(1, 0)
                                      .reshape(rank, in_channel, -1), axis=3)
        res[LAST] = np.expand_dims(core, axis=2)
    if mode == DecomposeMode.SCSK:
        res[FIRST] = np.expand_dims(portion.transpose(1, 0)
                                      .reshape(rank, in_channel, -1), axis=2)
        res[LAST] = np.expand_dims(core, axis=3)

    return res
