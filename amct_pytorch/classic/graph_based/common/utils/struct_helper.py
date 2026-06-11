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

from ..auto_calibration.sensitivity_base import SensitivityBase
from ..auto_calibration.sensitivity_base import MseSensitivity
from . import files as files_util
from . import vars_util


class DumpConfig:
    """ Config info for dump data. """
    def __init__(self, dump_dir, batch_num):
        """ Init func.

        Args:
            dump_dir (string): the path to save dump file.
            batch_num (int): batch to dump data, if -1, will dump every batch.
        """
        self.dump_dir = dump_dir
        self.batch_num = batch_num

        self.check_params()

    def check_params(self):
        """ check params"""
        self.dump_dir = os.path.realpath(self.dump_dir)


class GraphInfoBase:
    """ Config info for graph. """
    def __init__(self, graph):
        """ Init func """
        self.graph = graph

        self.check_params()

    def check_params(self):
        """ check params"""
        pass


class CalibrationConfigInfo:
    """ Config info for Calibration. """
    def __init__(self, config_defination):
        """ Init func """
        self.config_defination = config_defination
        self.batch_num = None
        self.check_params()

    def check_params(self):
        """ check params"""
        if self.config_defination is not None and not os.path.exists(self.config_defination):
            raise RuntimeError("The config_defination {} does not exist, please check the file path.".format(
                self.config_defination))