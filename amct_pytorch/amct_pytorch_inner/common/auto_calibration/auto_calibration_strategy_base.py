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

automatic calibration strategy base class.

"""


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
