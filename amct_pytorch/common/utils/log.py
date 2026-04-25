# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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

import logging
import os
import stat
from functools import wraps


# mode is 640
FILE_MODE = stat.S_IRUSR + stat.S_IWUSR + stat.S_IRGRP
# mode is 750
DIR_MODE = stat.S_IRWXU + stat.S_IRGRP + stat.S_IXGRP

LOG_FILE_DIR = 'amct_log'

LOG_SET_ENV = 'AMCT_LOG_LEVEL'
LOG_FILE_SET_ENV = 'AMCT_LOG_FILE_LEVEL'

LOGGING_LEVEL_MAP = {
    'DEBUG': logging.DEBUG,
    'INFO': logging.INFO,
    'WARNING': logging.WARNING,
    'ERROR': logging.ERROR
}


def check_level(level, name):
    """
    Function: Check level
    Parameter: level: log's level to be check
               name: level's name
    """
    if level.upper() not in LOGGING_LEVEL_MAP:
        raise ValueError("%s{'%s'} is invalid, only support %s"
                         % (name, level, list(LOGGING_LEVEL_MAP)))


def split_str_by_length(string, length):
    ''' split string on given length '''
    return [string[i: i + length].strip() for i in range(0, len(string), length)]


def log_split_deco(length=500):
    ''' check whether input string length and split it to smaller strings '''
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if len(args) > 1:
                args = list(args)
                args[1] = split_str_by_length(str(args[1]), length)
            else:
                input_key = tuple(kwargs.keys())[0]
                kwargs[input_key] = split_str_by_length(str(kwargs[input_key]), length)
            return func(*args, **kwargs)
        return wrapper
    return decorator


class LoggerBase():
    """
    Function:Record debug,info,warning,error level log
    API:logd, logi, logw, loge, set_debug_level
    """
    def __init__(self, log_dir, log_name):
        """
        Function:Create logger, console handler and file handler
        Parameter: log_dir: directory of log
                   log_name: name of log
        Return:None
        """
        # create logger
        self.logger = logging.getLogger("Log")
        self.logger.setLevel(logging.DEBUG)
        print_format = "%(asctime)s - %(levelname)s - [AMCT]:%(message)s"
        self.formatter = logging.Formatter(print_format)

        # add console handler
        self.console_handler = logging.StreamHandler()
        self.console_handler.setLevel(logging.INFO)
        self.console_handler.setFormatter(self.formatter)
        self.logger.addHandler(self.console_handler)

        # create log dir and empty log file
        log_dir = os.path.realpath(log_dir)
        self.debug_log_file = os.path.join(log_dir, log_name)
        # set path's permission 750
        os.makedirs(log_dir, DIR_MODE, exist_ok=True)
        with open(self.debug_log_file, 'w') as log_file:
            # set file's permission 640
            os.chmod(self.debug_log_file, FILE_MODE)
            log_file.write('')

        # add file handler
        self.file_handler = logging.FileHandler(self.debug_log_file, mode='w')
        self.file_handler.setLevel(logging.INFO)
        self.file_handler.setFormatter(self.formatter)
        self.logger.addHandler(self.file_handler)

    def set_debug_level(self, print_level='info', save_level='info'):
        """
        Function:Set debug level
        Parameter: print_level: 'debug', 'info', 'warning', 'error'
                   save_level: 'debug', 'info', 'warning', 'error'
        Return:None
        """
        # entry check
        print_level = print_level.upper()
        save_level = save_level.upper()

        check_level(print_level, 'print_level')
        check_level(save_level, 'save_level')

        self.console_handler.setLevel(LOGGING_LEVEL_MAP.get(print_level))
        self.file_handler.setLevel(LOGGING_LEVEL_MAP.get(save_level))

    @log_split_deco()
    def logd(self, debug_message, module_name="AMCT"):
        """
        Function:Record debug log
        Parameter: debug_message: debug log
                   module_name: name of module
        Return:None
        """
        for msg in debug_message:
            self.logger.debug("[%s]: %s", module_name, msg)

    @log_split_deco()
    def logi(self, info_message, module_name="AMCT"):
        """
        Function:Record info log
        Parameter: info_message: info log
                   module_name: name of module
        Return:None
        """
        for msg in info_message:
            self.logger.info("[%s]: %s", module_name, msg)

    @log_split_deco()
    def logw(self, warning_message, module_name="AMCT"):
        """
        Function:Record warning log
        Parameter: warning_message: warning log
              module_name: name of module
        Return:None
        """
        for msg in warning_message:
            self.logger.warning("[%s]: %s", module_name, msg)

    @log_split_deco()
    def loge(self, error_message, module_name="AMCT"):
        """
        Function:Record error log
        Parameter: error_message: error log
              module_name: name of module
        Return:None
        """
        for msg in error_message:
            self.logger.error("[%s]: %s", module_name, msg)

    def is_file_debug_level(self):
        """ is file_handler in debug mode. """
        if self.file_handler.level == logging.DEBUG:
            return True
        return False


class Logger(LoggerBase):
    """
    Function:Record debug,info,warning,error level log
    API:logd, logi, logw, loge
    """
    def __init__(self, log_dir, log_name):
        """
        Function: Create logger, console handler and file handler
        Parameter: log_dir: directory of log
                   log_name: name of log
        Return:None
        """
        super().__init__(log_dir, log_name)

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


LOGGER = Logger(os.path.join(os.getcwd(), LOG_FILE_DIR), 'amct_pytorch.log')
