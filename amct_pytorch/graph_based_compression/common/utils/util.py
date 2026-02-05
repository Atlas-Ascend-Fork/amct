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

import sys
import stat
import re
import struct
import numpy as np


# mode is 640
FILE_MODE = stat.S_IRUSR + stat.S_IWUSR + stat.S_IRGRP
# mode is 750
DIR_MODE = stat.S_IRWXU + stat.S_IRGRP + stat.S_IXGRP

MIN_FP16 = 2**-14
MAX_FP16 = 65504.0


def find_repeated_items(items):
    '''find repeated items in a list '''
    repeat_items = set()
    for item in items:
        count = items.count(item)
        if count > 1:
            repeat_items.add(item)

    return list(repeat_items)


def check_no_repeated(items, name):
    '''raise error when item is not empty'''
    if items:
        raise ValueError("Please delete repeated items in %s, "
                         "repeated items are %s " % (name, items))


def proto_float_to_python_float(value):
    '''transform proto float to python float'''
    val_fp32 = np.float32(value)
    return float(str(val_fp32))


def is_invalid(value_array):
    ''' whether there's inf or nan in value_array'''
    is_array_invalid = np.isnan(value_array) | np.isinf(value_array)
    return is_array_invalid.any()


def check_scale(scale, layer_name):
    ''' check whether the scale is valid '''
    if scale < sys.float_info.epsilon:
        raise ValueError(
            'layer {} has invalid scale {}'.format(layer_name, scale))

    if np.isnan(scale) | np.isinf(scale):
        raise ValueError(
            'layer {} has invalid scale {}'.format(layer_name, scale))


def version_higher_than(left_version, right_version):
    """
    Function: check if the left_version is higher than right_version
    """
    def _get_version_list(version_string):
        """
        Function: wrap version string for comparing
        """
        version = re.findall(r"\d{1,2}\.+\d{1,2}\.+\d{1,2}", version_string)
        if len(version) == 0:
            raise RuntimeError('Get invalid version {}, \
                please make sure to use the released version'.format(version_string))
        version_list = list(map(int, version[0].split('.')))
        return version_list
    left_ints = _get_version_list(left_version)
    right_ints = _get_version_list(right_version)

    version_length = 3
    for i in range(version_length):
        if left_ints[i] == right_ints[i]:
            continue
        elif left_ints[i] > right_ints[i]:
            return True
        else:
            return False

    return True


def cast_to_s19(data):
    """
    Function: cast float32 to s19
    Inputs:
        data: float32 data
    Return: precision is s19, type is float32
    """
    bytes_data = struct.pack('>f', data)
    int_data = struct.unpack('>I', bytes_data)[0]
    int_data &= 0xFFFFE000
    bytes_data = struct.pack('>I', int_data)
    return struct.unpack('>f', bytes_data)[0]


def cast_fp16_precision(data):
    """cast fp32 to fp16 precision, return fp32"""
    data = np.array(data)
    if data > MAX_FP16 or data < MIN_FP16:
        data = np.sqrt(data)
        data = data.astype(np.half).astype(np.float32)
        ret = data * data
    else:
        ret = data.astype(np.half).astype(np.float32)
    return float(ret)