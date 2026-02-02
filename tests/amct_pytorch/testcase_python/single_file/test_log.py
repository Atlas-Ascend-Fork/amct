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
import os
import unittest
import json
import numpy as np
import torch

from amct_pytorch.amct_pytorch_inner.amct_pytorch.utils.log import LOG_SET_ENV
from amct_pytorch.amct_pytorch_inner.amct_pytorch.utils.log import LOG_FILE_SET_ENV
from amct_pytorch.amct_pytorch_inner.amct_pytorch.utils.log import Logger

CUR_DIR = os.path.split(os.path.realpath(__file__))[0]

class TestLog(unittest.TestCase):
    """
    The UT for QuantizeTool
    """
    @classmethod
    def setUpClass(cls):
        cls.temp_folder = os.path.join(CUR_DIR, 'test_log')
        if not os.path.isdir(cls.temp_folder):
            os.makedirs(cls.temp_folder)
        cls.log_file = "amct_log.txt"

    @classmethod
    def tearDownClass(cls):
        os.popen('rm -r ' + cls.temp_folder)
        pass

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_info_level(self):
        os.environ[LOG_SET_ENV ] = 'info'
        logger = Logger(self.temp_folder, self.log_file)
        self.assertIsNone(logger.logi("this is info test"))

    def test_debug_level(self):
        os.environ[LOG_SET_ENV] = 'debug'
        logger = Logger(self.temp_folder, self.log_file)
        self.assertIsNone(logger.logd("this is debug test"))

    def test_file_info_level(self):
        os.environ[LOG_FILE_SET_ENV] = 'info'
        logger = Logger(self.temp_folder, self.log_file)
        self.assertIsNone(logger.logi("this is info test"))

    def test_debug_debug_level(self):
        os.environ[LOG_FILE_SET_ENV] = 'debug'
        logger = Logger(self.temp_folder, self.log_file)
        self.assertIsNone(logger.logi("this is debug test"))
