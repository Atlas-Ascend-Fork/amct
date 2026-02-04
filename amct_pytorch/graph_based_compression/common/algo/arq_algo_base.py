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

CHANNEL_WISE = 'channel_wise'


class ArqParam(ctypes.Structure):# pylint: disable=too-few-public-methods
    """
    Function: Data structure for c++ 'ArqParam'
    """
    _fields_ = [("numBits", ctypes.c_uint),
                ("channelWise", ctypes.c_bool),
                ("withOffset", ctypes.c_bool)]


ArqProcessedParams = collections.namedtuple('ArqProcessedParams',
                                            ['data_p', 'arq_params', 'scale_offset_size', 'scale_offset_data'])


class ArqAlgoBase(BaseQuantizeAlgorithm):
    """Function: The implement of ARQ quantize algorithm
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
        ArqAlgoBase._lib_loader = lib_loader
        ArqAlgoBase._enable_gpu = enable_gpu

    @staticmethod
    def preprocess_params(data, shape, args):
        """Extract and check input parameters, then convert them to cTypes's
           data type.
        """
        ArqAlgoBase.check_paramters(data, shape, args)
        arq_params = {}
        arq_params['data_length'] = ArqAlgoBase.get_data_length(shape)
        arq_params[CHANNEL_WISE] = ArqAlgoBase.extract_quant_param(
            args, CHANNEL_WISE, bool)
        arq_params['with_offset'] = ArqAlgoBase.extract_quant_param(
            args, 'with_offset', bool)
        arq_params['num_bits'] = ArqAlgoBase.extract_quant_param(
            args, 'num_bits', int)

        if arq_params.get(CHANNEL_WISE):
            scale_size = shape[0]
        else:
            scale_size = 1
        arq_params['scale_size'] = scale_size
        scale_offset_size = [(ctypes.c_float * scale_size)(),
                             (ctypes.c_int * scale_size)()]
        scale_offset_data = \
            [FloatData(scale_size, scale_offset_size[0]),
             IntData(scale_size, scale_offset_size[1])]

        data_p = ctypes.cast(data.buffer_info()[0],
                             ctypes.POINTER(ctypes.c_double))
        return ArqProcessedParams(data_p, arq_params, scale_offset_size, scale_offset_data)
    
    @classmethod
    def quantize_data(cls, data, shape, args, mode='CPU'):
        """Do arq quantize.
           Parameters: data(array): data to be quantized
                       shape(list/tuple): shape of data, should be 2 dims(fc)
                                          or 4dims(conv, deconv, lstm...)
                       args: quantize algorithm parameters, such as 'num_bits',
                             'channel_wise', 'with_offset'
           Return: scale, offset: quantize factor
        """
        if cls._lib_loader is None:
            raise RuntimeError(" The algorithm lib is None!")

        data_p, arq_params, scale_offset_size, scale_offset_data = \
            ArqAlgoBase.preprocess_params(data, shape, args)

        if mode == 'CPU':
            ret = cls._lib_loader.lib.ArqQuantDoublePython(
                data_p, arq_params.get('data_length'),
                ArqParam(arq_params.get('num_bits'),
                         arq_params.get(CHANNEL_WISE),
                         arq_params.get('with_offset')), scale_offset_data[0],
                scale_offset_data[1])
            if ret != 0:
                raise RuntimeError(
                    "Do ArqQuant failed, error code: {}".format(ret))

        scale_size = arq_params.get('scale_size')
        scale = [scale_offset_size[0][i] for i in range(scale_size)]
        offset = [scale_offset_size[1][i] for i in range(scale_size)]

        return scale, offset
