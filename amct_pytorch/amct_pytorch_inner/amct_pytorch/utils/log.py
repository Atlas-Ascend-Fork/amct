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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from ...amct_pytorch.common.utils.log_base import LoggerBase
from ...amct_pytorch.common.utils.log_base import LOG_FILE_DIR
from ...amct_pytorch.common.utils.log_base import check_level

LOG_SET_ENV = 'AMCT_LOG_LEVEL'
LOG_FILE_SET_ENV = 'AMCT_LOG_FILE_LEVEL'


class Logger(LoggerBase):
    """
    Function：Record debug，info，warning，error level log
    API：logd, logi, logw, loge
    """
    def __init__(self, log_dir, log_name):
        """
        Function: Create logger, console handler and file handler
        Parameter: log_dir: directory of log
                   log_name: name of log
        Return:None
        """
        LoggerBase.__init__(self, log_dir, log_name)

        # Get loging level from env
        console_level_pytorch = 'info'
        env_dist = os.environ
        if LOG_SET_ENV in env_dist:
            console_level_pytorch = env_dist[LOG_SET_ENV]
            console_level_pytorch = console_level_pytorch.upper()
            check_level(console_level_pytorch, LOG_SET_ENV)

        file_level_pytorch = 'info'
        if LOG_FILE_SET_ENV in env_dist:
            file_level_pytorch = env_dist[LOG_FILE_SET_ENV]
            file_level_pytorch = file_level_pytorch.upper()
            check_level(file_level_pytorch, LOG_FILE_SET_ENV)

        self.set_debug_level(console_level_pytorch, file_level_pytorch)


LOGGER = Logger(os.path.join(os.getcwd(), LOG_FILE_DIR), 'amct_pytorch.amct_pytorch_inner.amct_pytorch.log')
