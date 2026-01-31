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
import csv

DEFAULT_CAPACITY_TYPE = 'default'
ASCEND_CAPACITY_TYPE = 'ascend'
SUPPORTED_CAPACITY_TYPES = [DEFAULT_CAPACITY_TYPE, ASCEND_CAPACITY_TYPE]


class Capacity():
    """
    Function: Query capacity from csv config file
    APIs: is_enable, get_all_items
    """
    def __init__(self, config, ascend_config=None):
        self._capacity = {}
        self._ascend_capacity = {}

        self.capacity_type = DEFAULT_CAPACITY_TYPE
        self._parser_entry(config)
        if ascend_config is not None:
            self.set_capacity_type(ASCEND_CAPACITY_TYPE)
            self._parser_entry(ascend_config)
            self.set_capacity_type(DEFAULT_CAPACITY_TYPE)

    @property
    def capacity(self):
        ''' get capacity'''
        type2capacity = {
            DEFAULT_CAPACITY_TYPE: self._capacity,
            ASCEND_CAPACITY_TYPE: self._ascend_capacity
        }
        return type2capacity.get(self.capacity_type)

    def set_capacity_type(self, capacity_type):
        ''' get current capacity's type'''
        if capacity_type not in SUPPORTED_CAPACITY_TYPES:
            raise ValueError('capacity_type {} is unsupported, only support types {}'.format(capacity_type,
                                                                                             SUPPORTED_CAPACITY_TYPES))
        self.capacity_type = capacity_type

    def get_capacity_type(self):
        ''' get current capacity's type'''
        return self.capacity_type

    def is_enable(self, item):
        """query whether a feature is supported"""
        return self.capacity.get(item, False)

    def get_value(self, item):
        """Get value of a capacity"""
        return self.capacity.get(item, None)

    def show_capacities(self):
        """show all capacities and values"""
        print(self.capacity)

    def _parser_entry(self, config):
        with open(config) as fid:
            lines = csv.reader(fid)
            for line in lines:
                # skip empty line
                if len(line) == 0:
                    continue
                if len(line) < 3:
                    raise ValueError('Invalid line {}'.format(','.join(line)))

                item = line[0].strip()
                item_type = line[1].strip()

                type2func = {
                    'bool': self._parser_bool,
                    'list': self._parser_list,
                    'string': self._parser_string
                }
                if item_type not in type2func:
                    raise ValueError('Unknown type {}'.format(item_type))
                type2func[item_type](item, line[2:])

    def _parser_bool(self, item, values):
        value = values[0]
        value = value.strip()
        if value.upper() == 'True'.upper():
            self.capacity[item] = True
        elif value.upper() == 'False'.upper():
            self.capacity[item] = False
        else:
            raise ValueError('Unknown value {} for {}'.format(value, item))

    def _parser_list(self, item, values):
        self.capacity[item] = [value.strip() for value in values]

    def _parser_string(self, item, values):
        value = values[0]
        self.capacity[item] = value.strip()


def switch_capacity(capacity, capacity_type):
    """ decorate function to switch capacity to not default type"""
    def decorate(func):
        def wrapper(*args, **kw):
            try:
                capacity.set_capacity_type(capacity_type)
                ret = func(*args, **kw)
                return ret
            finally:
                capacity.set_capacity_type(DEFAULT_CAPACITY_TYPE)
        return wrapper
    return decorate
