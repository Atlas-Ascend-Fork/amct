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

the class of matching operators

"""

import os
import csv
import json
from collections import namedtuple

from ..performance_utils.data_structurer.quantize_info import PerformanceInfo
from ...utils.log import LOGGER # pylint: disable=E0402

OP_NAME = 'Op Name'
OP_TIME = 'Task Duration(us)'
TOTAL_TIME = 'Total Time(us)'
RECORD_TOTOAL_TIME = 'total_time'

PerfInfo = namedtuple('PerfInfo', ['fp_names', 'fp_times', 'int8_names', 'int8_times'])


class OperatorMatcher():  # pylint: disable=R0902,R0903
    '''
    Functions: the class of matching operators.
    APIs: match
    '''
    def __init__(self):
        self.performance_info = PerformanceInfo()
        self.fp_om_map_pb = {}
        self.int8_om_map_pb = {}
        self.main_pb_map_fp_om = {}
        self.main_pb_map_int8_om = {}
        self.recorded_list = []
        self.fp_profiling_file = None
        self.int8_profiling_file = None
        self.fp_profiling = {}
        self.int8_profiling = {}

        self.fp_om_file = None
        self.int8_om_file = None
        self.main_layers = []
        self.main_types = []
        self.auxi_layers = []

    @staticmethod
    def _parse_profiling_file(csv_file):
        profiling = {}
        total_time = 0
        csv_file = os.path.realpath(csv_file)
        with open(csv_file) as file_tmp:
            reader = csv.DictReader(file_tmp)
            for index, row in enumerate(reader):
                if index == 0:
                    total_time = float(row[TOTAL_TIME])
                profiling[row[OP_NAME]] = float(row[OP_TIME])
        return profiling, total_time

    def match(self, quant_info, iter_recorder, iter_num):
        """Operator mapping before and after matching quantization."""
        self.main_layers = quant_info.get_main_layers()
        self.auxi_layers = quant_info.get_auxi_layers()
        self.main_types = quant_info.get_main_types()
        recorder1 = iter_recorder.recorders[0]
        recorder2 = iter_recorder.recorders[iter_num]
        self.fp_om_file = os.path.realpath(recorder1['parser_json_file'])
        self.int8_om_file = os.path.realpath(recorder2['parser_json_file'])
        self.fp_profiling_file = os.path.realpath(recorder1['parser_csv_file'])
        self.int8_profiling_file = os.path.realpath(
            recorder2['parser_csv_file'])

        self.main_pb_map_fp_om, self.main_pb_map_int8_om = \
            self._fusion_op_mapping()
        self.fp_profiling, recorder1[RECORD_TOTOAL_TIME] = \
            self._parse_profiling_file(self.fp_profiling_file)
        self.int8_profiling, recorder2[RECORD_TOTOAL_TIME] = \
            self._parse_profiling_file(self.int8_profiling_file)

        if recorder1[RECORD_TOTOAL_TIME] == 0 or recorder2[RECORD_TOTOAL_TIME] == 0:
            LOGGER.loge(
                'Faied to get profiling data. Total_time is zero!')
            raise RuntimeError(
                'Faied to get profiling data. Total_time is zero!')

        perf_info = \
            self._get_performance_info()
        self.performance_info.set_main_layers(self.main_layers)
        self.performance_info.set_auxi_layers(self.auxi_layers)
        self.performance_info.set_main_types(self.main_types)
        self.performance_info.set_fp_names(iter_num, perf_info.fp_names)
        self.performance_info.set_fp_times(iter_num, perf_info.fp_times)
        self.performance_info.set_int8_names(iter_num, perf_info.int8_names)
        self.performance_info.set_int8_times(iter_num, perf_info.int8_times)

        self.performance_info.update_benefits_status(quant_info, iter_num)

        return self.performance_info

    def _get_key_by_value(self, map_dict, value, hard_check=True):
        keys = [key for key in map_dict.keys() if value in map_dict.get(key)]
        if hard_check:
            if not keys:
                raise RuntimeError('get_key_by_value error!')
            return keys
        if len(keys) == 1:
            if keys[0] not in self.recorded_list:
                return keys[0]
        if len(keys) == 2:
            if keys[0] not in self.recorded_list:
                return keys[0]
            if keys[1] not in self.recorded_list:
                return keys[1]
        return None

    def _supplement_ops(self, main_pb_map_om_1, om_map_pb_1, main_pb_map_om_2,
                        om_map_pb_2):
        for main_pb_name in self.main_layers:
            pb_name_list = []
            for om_name in main_pb_map_om_1[main_pb_name]:
                pb_name_list.extend(om_map_pb_1[om_name])
            for int8_pb_name in pb_name_list:
                fp_om_name = self._get_key_by_value(om_map_pb_2, int8_pb_name,
                                                    False)
                if fp_om_name is not None:
                    main_pb_map_om_2[main_pb_name].append(fp_om_name)
                    self.recorded_list.append(fp_om_name)

    def _fusion_op_mapping(self):
        with open(self.fp_om_file) as file_tmp:
            self.fp_om_map_pb = json.loads(file_tmp.read())
        with open(self.int8_om_file) as file_tmp:
            self.int8_om_map_pb = json.loads(file_tmp.read())

        self.main_pb_map_fp_om = {}
        self.main_pb_map_int8_om = {}
        self.recorded_list = []
        for main_pb_name in self.main_layers:
            fp_om_name = self._get_key_by_value(self.fp_om_map_pb,
                                                main_pb_name)
            self.main_pb_map_fp_om[main_pb_name] = fp_om_name
            self.recorded_list.extend(fp_om_name)
            int8_om_name = self._get_key_by_value(self.int8_om_map_pb,
                                                  main_pb_name)
            self.main_pb_map_int8_om[main_pb_name] = int8_om_name
            self.recorded_list.extend(int8_om_name)

        for index, main_pb_name in enumerate(self.main_layers):
            for quant_pb_name in self.auxi_layers[index]:
                int8_om_name = self._get_key_by_value(self.int8_om_map_pb,
                                                      quant_pb_name, False)
                if int8_om_name is not None:
                    self.main_pb_map_int8_om[main_pb_name].append(int8_om_name)
                    self.recorded_list.append(int8_om_name)

        self._supplement_ops(self.main_pb_map_int8_om, self.int8_om_map_pb,
                             self.main_pb_map_fp_om, self.fp_om_map_pb)
        self._supplement_ops(self.main_pb_map_fp_om, self.fp_om_map_pb,
                             self.main_pb_map_int8_om, self.int8_om_map_pb)
        self._supplement_ops(self.main_pb_map_int8_om, self.int8_om_map_pb,
                             self.main_pb_map_fp_om, self.fp_om_map_pb)

        return self.main_pb_map_fp_om, self.main_pb_map_int8_om

    def _get_performance_info(self):
        fp_times = []
        int8_times = []
        for main_pb_name in self.main_layers:
            fp_time = 0
            int8_time = 0
            for fp_om in self.main_pb_map_fp_om[main_pb_name]:
                if self.fp_profiling.get(fp_om):
                    fp_time = fp_time + self.fp_profiling[fp_om]
            for int8_om in self.main_pb_map_int8_om[main_pb_name]:
                if self.int8_profiling.get(int8_om):
                    int8_time = int8_time + self.int8_profiling[int8_om]
            fp_times.append(fp_time)
            int8_times.append(int8_time)
        fp_names = list(self.main_pb_map_fp_om.values())
        int8_names = list(self.main_pb_map_int8_om.values())

        return PerfInfo._make([fp_names, fp_times, int8_names, int8_times])
