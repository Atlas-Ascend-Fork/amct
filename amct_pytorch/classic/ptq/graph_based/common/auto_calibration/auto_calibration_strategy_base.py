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


from typing import Tuple
from abc import abstractmethod


class AutoCalibrationStrategyBase:
    """ the base class for auto fallback"""
    def __init__(self):
        """
        Function:
                __init__ function of class
        Parameter:
        """
        self.result = {}
        self.set_flag = True
        self.last_acc_result = {}

    def initialize(self, ranking_info: dict):
        """need to implement with different strategy

        Args:
            ranking_info (dict): the sensitivity ranking information of
            quantable layers, key is layer name and value is the
            sensitivity metric
        """
        raise NotImplementedError

    @abstractmethod
    def update_quant_config(self, metric_eval: Tuple[bool, float]):
        """need to implement with different strategy

        Args:
            metric_eval (Tuple[bool, float]): A tuple of whether the
            metric is satisfied and loss function value.
        """
        raise NotImplementedError
