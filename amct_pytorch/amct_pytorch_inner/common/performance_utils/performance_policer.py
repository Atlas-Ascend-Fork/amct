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

the performance fallback policy class file

"""


from abc import abstractmethod

PERFORMANCE_BENEFITS = 'performance_benefits'


class PerfStrategyController():
    """The performance stategy controller class."""
    def __init__(self, strategy):
        self.iteration_num = 0
        self.stop_iteration = 3
        self.strategy = strategy
        self.fp_times = []
        self.int8_times = []
        self.main_layers = []

    def is_iter_stop(self, iteration_num):
        """Judgment of termination condition of fallback strategy."""
        self.iteration_num = iteration_num
        if self.iteration_num >= self.stop_iteration:
            return True
        return False

    def analysis(self, performance_info, quant_info):
        """Generate quantitative fallback strategy."""
        self.fp_times = performance_info.get_fp_times()[self.iteration_num]
        self.int8_times = performance_info.get_int8_times()[self.iteration_num]
        self.main_layers = performance_info.get_main_layers()
        self.strategy.analysis(self.main_layers, self.fp_times, self.int8_times, quant_info)


class PerfStrategyBase(): # pylint: disable=R0903
    """The performance based quantitative fallback policy base class."""
    def __init__(self):
        pass

    @abstractmethod
    def analysis(self, main_layers, fp_times, int8_times, quant_info):
        """Generate quantitative fallback strategy."""
        raise NotImplementedError


class BatchRollBackStrategy(PerfStrategyBase): # pylint: disable=R0903
    """The performance based quantitative fallback policy sub class"""
    def __init__(self):
        PerfStrategyBase.__init__(self)
        self.main_layers = []
        self.fp_times = []
        self.int8_times = []

    def analysis(self, main_layers, fp_times, int8_times, quant_info):
        """Generate quantitative fallback strategy."""
        self.main_layers = main_layers
        self.fp_times = fp_times
        self.int8_times = int8_times

        for group_layer, config in quant_info.get_search_layers().items():
            origin_time = 0
            current_time = 0
            for layer_name in group_layer:
                index = self.main_layers.index(layer_name)
                origin_time += self.fp_times[index]
                current_time += self.int8_times[index]

            performance_benefits = origin_time - current_time
            if config[PERFORMANCE_BENEFITS] == 0 \
                and performance_benefits < 0:
                quant_info.change_search_layers_status(group_layer)
            if config[PERFORMANCE_BENEFITS] != 0 \
                and performance_benefits < config[PERFORMANCE_BENEFITS] \
                and config['status'] == 'float':
                quant_info.change_search_layers_status(group_layer)
            quant_info.change_search_layers_benefits(group_layer,
                                                     performance_benefits)
