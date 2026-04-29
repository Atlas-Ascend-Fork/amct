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
import logging
import io

from amct_pytorch.graph_based_compression.amct_pytorch.utils.log import LOGGER

def log_check_deco(func):
    def wrapper(*args, **kwargs):
        func_name = func.__name__
        logger = LOGGER.logger
        old_level = logger.level
        logger.setLevel(logging.DEBUG)

        log_stream = io.StringIO()
        handler = logging.StreamHandler(log_stream)
        handler.setLevel(logging.DEBUG)
        logger.addHandler(handler)
        func(*args, **kwargs)

        log_info = log_stream.getvalue()
        logger.removeHandler(handler)
        logger.setLevel(old_level)

        log_check(log_info, func_name)
    return wrapper

def log_check(log_info, func_name):
    if '  ' in log_info:
        raise RuntimeError(
            'Testcase {}\'s log have more than 2 space. Please check your log'.format(func_name))

    if '\n\n' in log_info:
        raise RuntimeError(
            'Testcase {}\'s log have more than 2 newline character. Please check your log'.format(func_name))

    log_info_list = log_info.split('\n')
    overlong_info = list(filter(lambda x: len(x) >= 1024, log_info_list))
    if overlong_info:
        raise RuntimeError(
            'Testcase {}\'s log have lines more than 1024 character. Please check your log'.format(func_name))
