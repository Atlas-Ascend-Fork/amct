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
import ctypes
import collections

from .base_quant_algo import BaseQuantizeAlgorithm
from .base_quant_algo import FloatData
from .base_quant_algo import IntData


class NuqParam(ctypes.Structure):# pylint: disable=too-few-public-methods
    """
    Function: Data structure for c++ 'NuqParam'
    """
    _fields_ = [("numSteps", ctypes.c_int),
                ("withOffset", ctypes.c_bool),
                ("numIter", ctypes.c_int)]


NquProcessedParams = collections.namedtuple('NquProcessedParams',
                                            ['data_p', 'nuq_params', 'scale_offset_size', 'scale_offset_data'])


class NuqAlgoBase(BaseQuantizeAlgorithm):
    """Function: The implement of NUQ quantize algorithm
       APIs: preprocess_params, quantize_data
    """
    _instance = None
    _lib_loader = None
    _enable_gpu = True

    def __new__(cls, *args, **kw): # pylint: disable=W0613
        if cls._instance is None:
            cls._instance = object.__new__(cls)
        return cls._instance

    def __init__(self, lib_loader, enable_gpu=True):
        NuqAlgoBase._lib_loader = lib_loader
        NuqAlgoBase._enable_gpu = enable_gpu

    @classmethod
    def quantize_data(cls, data, shape, args, mode='CPU'):
        """Do arq quantize.
           Parameters: data(array): data to be quantized
                       shape(list/tuple): shape of data, should be 2 dims(fc)
                                          or 4dims(conv, deconv, lstm...)
                       args: quantize algorithm parameters, such as
                             'num_steps', 'num_of_iteration', 'with_offset'
           Return: scale, offset: quantize factor
        """
        data_p, nuq_params, scale_offset_size, scale_offset_data = \
            NuqAlgoBase.preprocess_params(data, shape, args)

        if mode == 'CPU':
            ret = cls._lib_loader.lib.NuqQuantPython(
                data_p,
                nuq_params.get('data_length'),
                NuqParam(nuq_params.get('num_steps'),
                         nuq_params.get('with_offset'),
                         nuq_params.get('num_of_iteration')),
                scale_offset_data[0],
                scale_offset_data[1])
        else:
            if not cls._enable_gpu:
                raise RuntimeError(
                    "mode {} is not supported!".format(mode))
            ret = cls._lib_loader.lib.NuqQuantPythonGPU(
                data_p,
                nuq_params.get('data_length'),
                NuqParam(nuq_params.get('num_steps'),
                         nuq_params.get('with_offset'),
                         nuq_params.get('num_of_iteration')),
                scale_offset_data[0],
                scale_offset_data[1])
        if ret != 0:
            raise RuntimeError(
                "Do NuqQuant failed, error code: {}".format(ret))

        scale_size = nuq_params.get('scale_size')
        scale = [scale_offset_size[0][i] for i in range(scale_size)]
        offset = [scale_offset_size[1][i] for i in range(scale_size)]

        return scale, offset

    @staticmethod
    def preprocess_params(data, shape, args):
        """Extract and check input parameters, then convert them to cTypes's
           data type.
        """
        NuqAlgoBase.check_paramters(data, shape, args)
        nuq_params = {}
        nuq_params['data_length'] = \
            NuqAlgoBase.get_data_length(shape)
        nuq_params['num_steps'] = NuqAlgoBase.extract_quant_param(
            args, 'num_steps', int)
        nuq_params['with_offset'] = NuqAlgoBase.extract_quant_param(
            args, 'with_offset', bool)
        nuq_params['num_of_iteration'] = \
            NuqAlgoBase.extract_quant_param(
                args, 'num_of_iteration', int)
        if len(shape) == 4:
            scale_size = shape[0]
        else:
            scale_size = 1
        nuq_params['scale_size'] = scale_size
        scale_offset_size = [(ctypes.c_float * scale_size)(),
                             (ctypes.c_int * scale_size)()]
        scale_offset_data = \
            [FloatData(scale_size, scale_offset_size[0]),
             IntData(scale_size, scale_offset_size[1])]

        data_p = ctypes.cast(data.buffer_info()[0],
                             ctypes.POINTER(ctypes.c_double))
        return NquProcessedParams(data_p, nuq_params, scale_offset_size, scale_offset_data)
