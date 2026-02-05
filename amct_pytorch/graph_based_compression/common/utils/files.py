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
__all__ = ['create_path', 'create_empty_file', 'is_valid_name', 'split_save_path', 'concat_name',
           'is_valid_save_prefix', 'create_file_path', 'delete_dir', 'check_files_exist']

import os
import shutil
import json
import re
import numpy as np
from .util import DIR_MODE
from .util import FILE_MODE
from ...utils.log import LOGGER

FILE_NAME_PATTERN = "^[^/]{1,254}$"
SAVE_PREFIX_PATTERN = "^[^/]{0,241}$"


def create_path(file_path, mode=DIR_MODE):
    ''' Create path '''
    file_dir = os.path.realpath(file_path)
    try:
        os.makedirs(file_dir, mode, exist_ok=True)
    except FileExistsError:
        pass


def create_file_path(file_name, mode=DIR_MODE, check_exist=False):
    ''' Create path '''
    file_name = os.path.realpath(file_name)
    file_dir = os.path.split(file_name)[0]
    if check_exist:
        check_files_exist([file_name])
    os.makedirs(file_dir, mode, exist_ok=True)


def create_empty_file(file_name, check_exist=False):
    '''Create empty file '''
    file_realpath = os.path.realpath(file_name)
    # set path's permission 750
    create_file_path(file_realpath, check_exist=check_exist)

    with open(file_realpath, 'w') as record_file:
        record_file.write('')
    # set file's permission 640
    os.chmod(file_realpath, FILE_MODE)
    return file_realpath


def is_valid_name(input_file, name):
    ''' check input_file as inputs'''
    if input_file == '':
        raise ValueError('empty string is an invalid value for %s' % (name))

    if input_file[-1] == '/':
        raise ValueError("%s{%s} should be a file's name but a path."
                         % (name, input_file))

    # check input_file without path
    file_name = os.path.split(os.path.realpath(input_file))[1]
    if not re.match(FILE_NAME_PATTERN, file_name):
        raise ValueError(
            "%s's name{%s} is invalid, '/' is "
            "reserved characters, length should be less than 255." %
            (name, file_name))


def is_valid_save_prefix(save_prefix):
    ''' check save_prefix. '''
    if not re.match(SAVE_PREFIX_PATTERN, save_prefix):
        raise ValueError(
            "prefix{%s} is invalid. '/' is reserved "
            "characters, length should be less than 242." % (save_prefix))


def check_file_path(file_path, name):
    """ check whether file exits. """
    if not os.path.exists(os.path.realpath(file_path)):
        raise RuntimeError(
            'Can not find the {} at {}'.format(name, file_path))


def delete_dir(path):
    ''' Function: delete tmp dir '''
    shutil.rmtree(path, True)


def check_files_exist(file_names):
    '''check if file_name already exists'''
    for file_name in file_names:
        if os.path.exists(file_name):
            LOGGER.logw('The file already exists at {}, and '
                'it will be overwritten by AMCT'.format(file_name))


def split_save_path(save_path):
    """ split save_path to save_dir and save_prefix

    Args:
        save_path (string): the path where to store model and model's name.

    Returns:
        string: the folder path
        string: the prefix
    """
    if save_path == '':
        save_prefix = ''
        save_dir = os.path.realpath(save_path)
    elif save_path != '' and save_path[-1] == '/':
        save_prefix = ''
        save_dir = os.path.realpath(save_path)
    else:
        save_dir, save_prefix = os.path.split(os.path.realpath(save_path))
    is_valid_save_prefix(save_prefix)

    return save_dir, save_prefix


def concat_name(save_path, prefix, tail):
    """ Concat file's name.

    Args:
        save_path (string): the folder's path to save file
        prefix (string): the prefix of name
        tail (string): the tail of name

    Returns:
        [type]: [description]
    """
    if prefix != '':
        name = os.path.join(save_path, '_'.join([prefix, tail]))
    else:
        name = os.path.join(save_path, tail)

    return name


def save_to_json(file_name, content):
    """ Save content to a json file.

    Args:
        file_name (string): the flile to save content.
        content (dict): the content to save.
    """
    file_name = create_empty_file(file_name, check_exist=True)
    with open(file_name, 'w') as fid:
        json.dump(content, fid, indent=4, separators=(',', ':'))


def find_dump_file(data_dir, name_prefix):
    """ Find dump file with data_dir and name_prefix.

    Args:
        data_dir (string): the path to find file.
        name_prefix (string): the name_prefix of file.

    Raises:
        RuntimeError: Not find any file.

    Returns:
        list of string: the files.
    """
    fm_file_list = []
    data_dir = os.path.realpath(data_dir)
    file_list = os.listdir(data_dir)
    for file_name in file_list:
        if not os.path.isfile(os.path.join(data_dir, file_name)):
            continue
        if file_name.startswith(name_prefix):
            fm_file_list.append(os.path.join(data_dir, file_name))
    if not fm_file_list:
        raise RuntimeError('Cannot find file begin with {}'.format(name_prefix))
    return fm_file_list


def parse_dump_data(file_path, with_type=False):
    """ Parse the dump file, with rule as follows:
    | (type), dim, shape[0], shape[1] ... | data[0], data[1], ... |
    |   float32                           |      type             |

    Args:
        file_path (string): the file's path.
        with_type (bool, optional): with type info in data or not. Defaults to False.

    Returns:
        np.array: the data with shape
    """
    real_file_path = os.path.realpath(file_path)
    check_file_path(real_file_path, 'feature_map_dump_file')
    dump_data = np.fromfile(real_file_path, np.byte)
    if with_type:
        type_info = int(np.frombuffer(dump_data[0:4], np.float32)[0])
        dim_info = int(np.frombuffer(dump_data[4:8], np.float32)[0])
        shape_info = list(np.frombuffer(dump_data[8:8 + dim_info * 4], np.float32))
        shape = list(map(int, shape_info))
        info_length = 8 + dim_info * 4
    else:
        type_info = 0
        dim_info = int(np.frombuffer(dump_data[0:4], np.float32)[0])
        shape_info = list(np.frombuffer(dump_data[4: 4 + dim_info * 4], np.float32))
        shape = list(map(int, shape_info))
        info_length = 4 + dim_info * 4
    # 1) keep same with c++; 2) each amct shoule be same if use this function
    type_mappint = {'0': np.float32, '1': np.double, '2': np.int32, '3': np.float16}
    data_info = np.frombuffer(dump_data[info_length:], type_mappint.get(str(type_info)))
    shape = list(map(int, shape))
    np_feature_map_data = data_info.reshape(shape)
    return np_feature_map_data

