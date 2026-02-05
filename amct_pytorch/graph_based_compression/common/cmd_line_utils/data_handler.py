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

Data handle functions for amct command line tools.

"""
import os
from datetime import datetime
from datetime import timezone
from datetime import timedelta
import secrets
import numpy as np
from ..utils.files import delete_dir


def load_data(inputs, data_paths, data_types, num=None):
    '''
    Function: Load binary data in the given data path to numpy array and create data mapping
    Arguments:
    inputs: dict, {'input_node_name': dims(list)} the input tensors
    data_paths: list, the pathes to the processed binry data to be feeded to the inputs
    data_types: list, the dtypes of the input tensors
    num: batch num for yielding data. if None, yield all data in data_paths.
    Return:
    data_map: dict, the mapping from input_names to numpy arrays, which will be pass to tf.Session feed_dict
    '''

    data_shapes = list(inputs.values())
    input_names = list(inputs.keys())

    data_file_lists = []
    for index, _ in enumerate(input_names):
        data_file_list = os.listdir(data_paths[index])
        data_file_list.sort()
        data_file_lists.append(data_file_list)
        if num is None:
            num = len(data_file_list)
        if len(data_file_list) < num:
            raise RuntimeError("The number of given data batches is not enough "
                    "to run calibration with the given batch_num. Total batches"
                    " found: {}, but trying to load: {}.".format(len(data_file_list), num))

    for i in range(num):
        data_map = {}
        for index, input_node_name in enumerate(input_names):
            data_file_name = os.path.join(data_paths[index], data_file_lists[index][i])
            data_input = load_bin_data(data_file_name, data_types[index], data_shapes[index])
            data_map[input_node_name] = data_input
        yield data_map


def load_bin_data(data_file_name, data_types, data_shapes):
    '''
    Function: Load binary data in the given data path to numpy array format.
    Arguments:
    data_file_name: str, path to the binary file to be load.
    data_types: str, dtypes to be parsed to.
    data_shapes: list, the shape of the array to be parsed to
    Return:
    data_input: numpy array
    '''
    np_type_mapping = {
            'float64': np.float64,
            'float32': np.float32,
            'float16': np.float16,
            'int8': np.int8,
            'int16': np.int16,
            'int64': np.int64,
            'int32': np.int32,
            'uint8': np.uint8,
            'uint16': np.uint16,
            'uint32': np.uint32,
            'uint64': np.uint64,
            'bool': np.bool_
            }

    if data_types not in np_type_mapping.keys():
        raise TypeError('Unsupported data type[{}]!'.format(data_types))

    return np.fromfile(data_file_name, np_type_mapping.get(data_types)).reshape(data_shapes)


def tmp_dir():
    '''
    Function: get tmp dir name
    Return: name
    '''
    now = datetime.now(tz=timezone(offset=timedelta(hours=8)))
    secret_generator = secrets.SystemRandom()
    return str(secret_generator.randint(10000, 99999)) + '_' + now.strftime("%Y%m%d%H%M%S")


def delete_tmp_dir(tmp):
    '''
    Function: delete tmp dir
    Return: name
    '''
    level = os.environ.get('AMCT_LOG_LEVEL')
    if level is None or level.upper() != 'DEBUG':
        delete_dir(tmp)
