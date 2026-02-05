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


import copy
import collections
from .auto_calibration_strategy_base import AutoCalibrationStrategyBase

STOP_FLAG = 'stop_flag'
ROLL_BACK_CONFIG = 'roll_back_config'


class BinarySearchStrategy(AutoCalibrationStrategyBase):
    """ the binary search class for auto fallback"""
    def __init__(self):
        """
        Function:
                __init__ function of class
        Parameter:
                cos_dict: cosine similarity dict
                quant_config: quant config file
        """
        super(BinarySearchStrategy, self).__init__()
        self.cos_dict = {}
        self.record = ""
        self.sorted_cos_dict_list = []
        self.sorted_cos_quant_dict = collections.OrderedDict()
        self.left = 0
        self.mid = 0
        self.right = 0
        self.set_flag = True
        self.last_acc_result = {}

    def initialize(self, ranking_info):
        """
        Function:
                binary search strategy init
        Parameter:
                ranking_info: cosine similarity dict
        """
        self.cos_dict = ranking_info
        self.record = 'init'
        self.sorted_cos_dict_list = sorted(
            self.cos_dict.items(), key=lambda d: d[1])
        self.sorted_cos_quant_dict = self.init_sorted_cos_quant_dict()
        self.left = 0
        self.mid = 0
        self.right = len(self.cos_dict) - 1

    def init_sorted_cos_quant_dict(self):
        """ init sorted cosine similarity dict"""
        sorted_dict = collections.OrderedDict()
        for item in self.sorted_cos_dict_list:
            sorted_dict[item[0]] = True
        return sorted_dict

    def reset_sorted_cos_dict(self, stop_layer):
        """ update sorted cosine similarity dict"""
        self.set_flag = True

        for item in self.sorted_cos_dict_list:
            if self.set_flag:
                self.sorted_cos_quant_dict[item[0]] = False
            else:
                self.sorted_cos_quant_dict[item[0]] = True
            if item[0] == stop_layer:
                self.set_flag = False

    def binary_search(self):
        """ binary search function"""
        layer_name = self.sorted_cos_dict_list[self.mid][0]
        return layer_name

    def update_quant_config(self, metric_eval):
        """
        Function:
                update quant config depend on metric eval result
        Parameter:
                metric_eval (Tuple[bool, float]): A tuple of whether the
                metric is satisfied and loss function value.
        """
        stop_flag = False
        accuracy, _ = metric_eval
        if accuracy:
            self.last_acc_result = copy.deepcopy(self.result)

        # if first accuracy is not enough
        # set half false and return
        if self.record == 'init' and accuracy is False:
            self.mid = int((self.left + self.right) / 2)
            stop_layer = self.binary_search()
            self.reset_sorted_cos_dict(stop_layer)
            self.result[STOP_FLAG] = False
            self.result[ROLL_BACK_CONFIG] = self.sorted_cos_quant_dict
            self.record = 'noinit'
            return self.result

        if not accuracy:
            self.left = self.mid + 1
        else:
            self.right = self.mid - 1
        self.mid = int((self.left + self.right) / 2)

        if self.left > self.right:
            stop_flag = True
            if not accuracy:
                self.last_acc_result[STOP_FLAG] = stop_flag
                return self.last_acc_result
            self.result[STOP_FLAG] = stop_flag
            self.result[ROLL_BACK_CONFIG] = self.sorted_cos_quant_dict
            return self.result

        stop_layer = self.binary_search()
        self.reset_sorted_cos_dict(stop_layer)
        self.result[STOP_FLAG] = stop_flag
        self.result[ROLL_BACK_CONFIG] = self.sorted_cos_quant_dict

        return self.result
