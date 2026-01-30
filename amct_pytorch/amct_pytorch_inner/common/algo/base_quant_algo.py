#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use
this file except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

NUQ quantize algorithm
"""
import ctypes
from array import array

WEIGHTS_QUANT_ALGO = {}


class FloatData(ctypes.Structure):# pylint: disable=too-few-public-methods
    """
    Function: Data structure for c++ 'FloatData'
    """
    _fields_ = [("length", ctypes.c_uint),
                ("data", ctypes.POINTER(ctypes.c_float))]


class IntData(ctypes.Structure):# pylint: disable=too-few-public-methods
    """
    Function: Data structure for c++ 'IntData'
    """
    _fields_ = [("length", ctypes.c_uint),
                ("data", ctypes.POINTER(ctypes.c_int))]


class BaseQuantizeAlgorithm():
    """Function: Base define of quantize algorithm
       APIs: check_paramters, extract_quant_param, get_data_length
    """
    @staticmethod
    def check_paramters(data, shape, args):
        """Check input parameters types"""
        if not isinstance(data, array):
            raise TypeError('Input "data" must be "array", actual is {}.' \
                ''.format(type(data)))
        if not isinstance(shape, list) and not isinstance(shape, tuple):
            raise TypeError('Input "shape" must be "list"/"tuple", ' \
                'actual is {}.'.format(type(shape)))
        if not isinstance(args, dict):
            raise TypeError('Input "args" must be "dict"m actual is ' \
                '{}.'.format(type(args)))

    @staticmethod
    def extract_quant_param(args, key, value_type):
        """Extract quantize parameter from input dict with specific dtype"""
        value = args.get(key)
        if value is None:
            raise KeyError('Cannot find {} in {}'.format(key, args))
        if not isinstance(value, value_type):
            raise TypeError('value of {} should be {}, actual is {}'.format(
                key, value_type, type(value)))
        return value

    @staticmethod
    def get_data_length(shape):
        """Calculate data length from shape."""
        length = 1
        for dim in shape:
            if dim == 0:
                raise ValueError('There is 0 in shape')
            length *= dim
        return length
