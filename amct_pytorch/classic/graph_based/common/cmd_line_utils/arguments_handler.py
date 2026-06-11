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

Functions and classes to handle amct command line arguments.

"""

import os


def process_data_shape(values):
    """ Process data_shape to several shape info.
    For example, "input_name1:n1,c1,h1,w1;input_name2:n1,c1,h1,w1" to
    {"input_name1": [n1,c1,h1,w1], "input_name2": [n1,c1,h1,w1]},

    or "input_name1:0:n1,c1,h1,w1;input_name2:0:0:n1,c1,h1,w1" to
    {"input_name1:0": [n1,c1,h1,w1], "input_name2:0:0": [n1,c1,h1,w1]}

    Args:
        values (string): data_shape like "input_name1:n1,c1,h1,w1;input_name2:n1,c1,h1,w1"
                                or "input_name1:0:n1,c1,h1,w1;input_name2:0:0:n1,c1,h1,w1"

    Raises:
        ValueError: format is invalid

    Returns:
        dict: new shape info, key is input_name and value is shape(list)
    """
    data_shapes_str = values.split(';')
    input_dict = {}
    for data_item in data_shapes_str:
        data_list = data_item.split(':')
        data_name = ':'.join(data_list[:-1])
        data_shape_str = data_list[-1]
        dims = data_shape_str.split(',')
        if len(data_list) < 2 or len(data_name) == 0 or len(dims) == 0:
            raise ValueError(
                "Invalid input_shape. Input name and shapes of each input node"
                " should be ':' split. E.g.: input_name1:n1,c1,h1,w1"
                " or input_name2:0:n1,c1,h1,w1")
        data_shape = [int(shape_dim) for shape_dim in dims]
        input_dict[data_name] = data_shape

    return input_dict


def process_multi_data_path(values):
    """Process data_path to several path info.

    Args:
        values (string): data_path like "input_name1:n1,c1,h1,w1;input_name2:n1,c1,h1,w1"

    Returns:
        list of string: new path info.
    """
    data_paths = values.split(';')
    data_paths = [os.path.realpath(data_path) for data_path in data_paths]
    return data_paths
