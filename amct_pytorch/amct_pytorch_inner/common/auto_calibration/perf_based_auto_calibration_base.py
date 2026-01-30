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

the base class of performance based auto calibration

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json
import shutil
from datetime import datetime
from datetime import timezone
from datetime import timedelta
from google.protobuf import text_format

from ..performance_utils.ascend_sampler import AscendSampler
from ..performance_utils.operator_matcher import OperatorMatcher
from ..performance_utils.performance_helper import IterationRecorder
from ..performance_utils.knowledge_lib import KnowledgeLib
from ..performance_utils.data_structurer.sampler_proto import \
    SamplerProtoConfig
from ...utils.log import LOGGER # pylint: disable=E0402
from ...utils import log # pylint: disable=E0402
from ..utils import files
from ...proto import scale_offset_record_pb2 # pylint: disable=E0402
from ..utils.record_file_operator import record_activation_scale_offset
from ..utils.record_file_operator import record_weights_scale_offset
from ..utils.record_file_operator import record_shift_bits
from ..utils.record_file_operator import record_skip_status
from ..utils.files import split_save_path
from ..performance_utils.data_structurer.quantize_info import QuantInfo
from ..performance_utils.performance_helper import CsvManager
from ..performance_utils.performance_policer import PerfStrategyController


class PerfBasedAutoCalibrationBase(): # pylint: disable=R0902
    """The superclass of automatic calibration based on performance.

    This superclass contains the main control of automatic calibration
    and a list of functions that subclasses must inherit and implement
    accroding to framework.

    Args:
        sampler_config_file (str): a string of config file path.
        save_dir (str): a string of quantized model file path.
        strategy (Strategy): A instance inherited from 'Strategy' and all the
            methods are implemented.
    """
    def __init__(self, sampler_config_file, save_dir, strategy,
                 framework):
        self.sampler_config_file = os.path.realpath(sampler_config_file)

        self.save_dir, self.prefix_name = split_save_path(save_dir)
        files.create_path(self.save_dir)

        self.amct_log_dir = log.DEFAULT_LOG_FILE_DIR
        self.temp_dir = self._get_temp_dir()
        files.create_path(self.temp_dir)

        self.iter_num = 0
        self.best_iter_num = 0
        self.log_info = ''
        self.performance_info = None
        self.iter_recorder = IterationRecorder(self.temp_dir,
                                               self.prefix_name)
        self.strategy_controller = PerfStrategyController(strategy)
        self.select_sampler(framework)
        self.operator_matcher = OperatorMatcher()
        self.quant_info = QuantInfo()
        self.knowledge_lib = KnowledgeLib()

    @staticmethod
    def print_log(iteration_recorder):
        """Get log information."""
        log_info = ''
        log_info += '*' * 100
        log_info += '\n'
        point_keys = ['iter_num', 'total_time', 'roll_back_layers']
        for key, value in iteration_recorder.items():
            if key in point_keys:
                if key == 'roll_back_layers':
                    log_info += '\t{} num: {}\n'.format(
                        key, 0 if value is None else len(value))
                elif key == 'total_time':
                    log_info += '\t{}: {:.2f} us\n'.format(key, value)
                else:
                    log_info += '\t{}: {}\n'.format(key, value)
        log_info += '*' * 100
        log_info += '\n'
        return log_info

    @staticmethod
    def clean_tmp(path):
        """Clean the tmp file or path."""
        if os.path.isfile(path):
            os.remove(path)
        if os.path.exists(path):
            shutil.rmtree(path)

    def update_iteration_recorder(self):
        """Update iteration's recorder."""
        self.iter_recorder.update(self.iter_num)

    def select_sampler(self, framework):
        """Select performance's sampler."""
        self.sampler_config_file = os.path.realpath(self.sampler_config_file)
        sampler_proto_config = SamplerProtoConfig(
            self.sampler_config_file, framework)
        self.sampler_config = sampler_proto_config.get_sampler_config()

        if 'ascend_sampler_config' in self.sampler_config.keys():
            self.sampler = AscendSampler(
                self.sampler_config['ascend_sampler_config'])

    def extract_record_file(self):
        """
        The quantization factor subset is extracted according to the
        quantization configuration.
        """
        quant_layer_num = 0
        records = scale_offset_record_pb2.ScaleOffsetRecord()
        for layer_name in self.quant_info.get_main_layers():
            quant_config = self.quant_info.get_quant_config().get(layer_name)
            if quant_config.get('quant_enable'):
                quant_layer_num += 1
                quant_factor = self.quant_info.get_total_records().get(
                    layer_name)
                scale_d = quant_factor.get('data_scale')
                offset_d = quant_factor.get('data_offset')
                record_activation_scale_offset(records, layer_name, scale_d,
                                               offset_d)
                if self.quant_info.get_main_types()[layer_name] not in (
                        'AvgPool', ):
                    scale = quant_factor.get('weight_scale').reshape(
                        -1).tolist()
                    offset = quant_factor.get('weight_offset').reshape(
                        -1).tolist()
                    record_weights_scale_offset(records, layer_name, scale, offset)
                if self.quant_info.get_main_types()[layer_name] not in ('AveragePool', ):
                    shift_bit = quant_factor.get('shift_n').reshape(-1).tolist()
                    record_shift_bits(records, layer_name, shift_bit)
            skip_fusion_layers = \
                self.quant_info.get_quant_config().get('skip_fusion_layers')
            if skip_fusion_layers is not None and \
                layer_name in skip_fusion_layers:
                record_skip_status(records, layer_name, True)

        self.iter_recorder.add_items(self.iter_num, 'record_file',
                                     'model_path', '.txt')
        record_file = files.create_empty_file(
            self.iter_recorder.recorders[self.iter_num]['record_file'])
        with open(record_file, 'w') as file_tmp:
            file_tmp.write(text_format.MessageToString(records, as_utf8=True))
        return quant_layer_num

    def get_quant_model_info(self):
        """Get quantized model's information."""
        raise NotImplementedError

    def update_quant_config(self):
        """Update quantize's config."""
        for group_layer, config in self.quant_info.get_search_layers().items():
            for layer_name in group_layer:
                if config['status'] == 'int8':
                    self.quant_info.set_config_enable(layer_name, True)
                else:
                    self.quant_info.set_config_enable(layer_name, False)

        self.iter_recorder.add_items(self.iter_num, 'config_file',
                                     'model_path', '.json')
        config_file = files.create_empty_file(
            self.iter_recorder.recorders[self.iter_num]['config_file'])
        with open(config_file, 'w') as fid:
            json.dump(self.quant_info.get_quant_config(),
                      fid,
                      indent=4,
                      separators=(',', ':'))

        self.iter_recorder.recorders[self.iter_num][
            'roll_back_layers'] = self.quant_info.get_roll_back_layers()
        self.iter_recorder.recorders[self.iter_num][
            'not_quantized_layers'] = self.quant_info.get_not_quant_layers()

    def roll_back_and_save_model(self):
        """
        According to the quantization configuration, the original model is
        quantized, and the mixed precision quantization model is generated.
        """
        raise NotImplementedError

    def summary(self):
        """Summarize and output the final result information."""
        csv_manager = CsvManager()
        csv_heads = ['id', 'op_names', 'op_types']
        for key in self.performance_info.get_status():
            csv_heads.extend([
                'fp_iter{}_names'.format(key),
                'fp_iter{}_times'.format(key),
                'int8_iter{}_names'.format(key),
                'int8_iter{}_times'.format(key),
                'benefits_iter{}'.format(key),
                'status_iter{}'.format(key)])

        csv_rows = []
        for index, layer_name in enumerate(
                self.performance_info.get_main_layers()):
            csv_row_dict = {
                'id': index + 1,
                'op_names': layer_name,
                'op_types': self.performance_info.get_main_types().get(layer_name)}
            for head_index, key in enumerate(csv_heads[3:]):
                out_index = head_index // 6 + 1
                in_index = head_index % 6
                if in_index == 0:
                    csv_row_dict[key] = self.performance_info.get_fp_names().get(out_index)[index]
                elif in_index == 1:
                    csv_row_dict[key] = self.performance_info.get_fp_times().get(out_index)[index]
                elif in_index == 2:
                    csv_row_dict[key] = self.performance_info.get_int8_names().get(out_index)[index]
                elif in_index == 3:
                    csv_row_dict[key] = self.performance_info.get_int8_times().get(out_index)[index]
                elif in_index == 4:
                    csv_row_dict[key] = self.performance_info.get_benefits().get(out_index)[index]
                elif in_index == 5:
                    csv_row_dict[key] = self.performance_info.get_status().get(out_index)[index]
            csv_rows.append(csv_row_dict)
        ouput_csv_file = os.path.join(
            self.amct_log_dir, 'amct_perf_info.csv')
        csv_manager.write(ouput_csv_file, csv_heads, csv_rows)

        self.log_info = 'The following is the model information during ' \
            'different iterations of performance based auto calibration:\n'
        for _, value in self.iter_recorder.recorders.items():
            self.log_info += PerfBasedAutoCalibrationBase.print_log(value)
        sorted_list = sorted(self.iter_recorder.recorders.items(),
                             key=lambda x: x[1]['total_time'],
                             reverse=False)
        self.log_info += '# Iteration with optimal performance is {}. ' \
            'The network time consuming: {:.2f} us.'.format(
                sorted_list[0][0],
                self.iter_recorder.recorders[sorted_list[0][0]]['total_time'])
        self.best_iter_num = sorted_list[0][0]
        LOGGER.logi(self.log_info)

    def run(self):
        """The main control of automatic calibration."""
        # get quant info
        self.get_quant_model_info()
        # generate the working path of iteration 0 and copy the original model
        # to the working path.
        self.update_iteration_recorder()
        while True:
            # obtaining performance data on a model board
            self.sampler.sample_profiling(self.iter_recorder, self.iter_num,
                                          self.quant_info)
            if self.iter_num >= 1:
                if self.iter_num == 1:
                    self.knowledge_lib.pre_fallback(self.quant_info)
                self.performance_info = self.operator_matcher.match(
                    self.quant_info, self.iter_recorder, self.iter_num)
                if self.strategy_controller.is_iter_stop(self.iter_num):
                    break
                self.strategy_controller.analysis(self.performance_info,
                                                  self.quant_info)
            # add 1 to the iterator.
            self.iter_num += 1
            # generate the working path of iteration x (1/2/3).
            self.update_iteration_recorder()
            # updating quantitative configurations
            self.update_quant_config()
            # generating a quantitative model
            self.roll_back_and_save_model()

        self.summary()
        level = os.environ.get('AMCT_LOG_LEVEL')
        if level is None or level.upper() != 'DEBUG':
            shutil.rmtree(self.temp_dir)

    def _get_temp_dir(self):
        time_stamp = datetime.now(tz=timezone(offset=timedelta(hours=8))).strftime('%Y%m%d%H%M%S%f')
        temp_dir = os.path.join(self.amct_log_dir, 'temp{}'.format(time_stamp))
        return temp_dir