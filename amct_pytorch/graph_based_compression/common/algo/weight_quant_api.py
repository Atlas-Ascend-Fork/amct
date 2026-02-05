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
from array import array

import numpy as np
from .base_quant_algo import FloatData
from .base_quant_algo import IntData

DEFAULT_WITH_OFFSET = False


class WtsPtqBase():
    """Function: The implement of WTS_CALI quantize algorithm
       APIs: preprocess_params, quantize_data
    """
    def __init__(self, device_mode, lib_loader):
        self.set_device_mode(device_mode)
        self.lib_loader = lib_loader

    def set_device_mode(self, device_mode):
        """ set_device_mode"""
        if device_mode not in ['CPU', 'GPU']:
            raise ValueError('device mode only support ["CPU", "GPU"], ' \
                'but get {}'.format(device_mode))
        self.device_mode = device_mode


    def weight_calibration_np(self, numpy_data, wts_param, wts_ptq_algo):
        """
        Function: warpper of arq_quant_np
        Parameter: numpy_data: data to be quantized as numpy array
                wts_param: weight quantize algorithm parameter
        Return: scale: scale of input data
                offset: offset of input data
        """
        weights_shape = numpy_data.shape
        array_data = array('d', numpy_data.astype(np.float32).flatten())
        scale, offset = self.weights_calibration_kernel(
            array_data, weights_shape, wts_param, wts_ptq_algo)
        numpy_data[:] = np.array(array_data, numpy_data.dtype).reshape(
            weights_shape)

        return scale, offset

    def weights_calibration_kernel(self, data, shape, wts_param, wts_ptq_algo):
        """Do weights quantize
        Parameters: data(array) weights to quantize
                    shape: shape of data
                    wts_param: weights quantize algorithm parameters
        Return: scale: scale of input data
                offset: offset of input data
        """
        wts_algo = wts_param['wts_algo']
        if wts_algo not in wts_ptq_algo:
            raise RuntimeError('Not support quantize algorithm "{}" yet.' \
                .format(wts_algo))
        if self.device_mode == 'CPU':
            scale, offset = wts_ptq_algo[wts_algo].quantize_data(
                data, shape, wts_param, 'CPU')
        else:
            scale, offset = wts_ptq_algo[wts_algo].quantize_data(
                data, shape, wts_param, 'GPU')
        return scale, offset

    def weights_quantize_np(self, numpy_data, scale, offset, num_bits=8):
        """
        Function: warpper of weights quantize of numpy
        Parameter: numpy_array: data to be quantized as numpy array
                scale: scale of input data
                offset: offset of input data
        Return: int8_data: quantized int8 data
        """
        weights_shape = numpy_data.shape
        array_data = array('d', numpy_data.flatten())
        int8_data = self.weights_quantize_kernel(
            array_data, scale, offset, num_bits)
        int8_array = np.frombuffer(int8_data, np.int8).reshape(weights_shape)
        return int8_array

    def weights_quantize_kernel(self, array_data, scale, offset, num_bits=8):
        """
        Function: Do weights quantize.
        Parameter: array_data: python array type data to quantize
                scale: scale of input data
                offset: offset of input data
                num_bits: bits number to quantize
        Return: int8_data: quantized int8 data in bytes
        """
        if len(scale) != len(offset):
            raise RuntimeError('length of scale:{} must equal to offset:{}'
                               .format(len(scale), len(offset)))
        data_p = ctypes.cast(
            array_data.buffer_info()[0], ctypes.POINTER(ctypes.c_double))
        data_length = len(array_data)

        int8_data = (ctypes.c_byte * data_length)()
        scale_offset_size = [(ctypes.c_float * len(scale))(*scale),
                             (ctypes.c_int * len(offset))(*offset)]
        scale_offset_data = [FloatData(len(scale), scale_offset_size[0]),
                             IntData(len(scale), scale_offset_size[1])]

        if self.device_mode == 'CPU':
            ret = self.lib_loader.lib.QuantRealDoublePython(
                data_p,
                data_length,
                scale_offset_data[0],
                scale_offset_data[1],
                ctypes.c_uint(num_bits),
                int8_data)
            if ret != 0:
                raise RuntimeError(
                    "Do QuantReal failed, error code: {}".format(ret))
        elif self.device_mode == 'GPU':
            ret = self.lib_loader.lib.QuantRealDoublePythonGPU(
                data_p,
                data_length,
                scale_offset_data[0],
                scale_offset_data[1],
                ctypes.c_uint(num_bits),
                int8_data)
            if ret != 0:
                raise RuntimeError(
                    "Do QuantReal GPU failed, error code: {}".format(ret))
        else:
            raise RuntimeError(
                'Not supported device mode "{}"'.format(self.device_mode))

        return bytes(int8_data)
