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


import os
import copy
import json
import shutil
from datetime import datetime
from datetime import timezone
from datetime import timedelta
import numpy as np

from ..utils import files
from ...utils.log import LOGGER # pylint: disable=E0402
from ..utils.log_base import LOG_FILE_DIR
from ..utils.util import find_repeated_items
from ..utils.util import check_no_repeated
from .auto_calibration_evaluator_base import AutoCalibrationEvaluatorBase
from .auto_calibration_strategy_base import AutoCalibrationStrategyBase
from .sensitivity_base import SensitivityBase


ALL_ROLL_BACK = 0
PARTIAL_ROLL_BACK = 1
METRIC_RET_LEN = 2
MAX_PRINT_INFO_NUM = 10
ROLL_BACK_CONFIG = 'roll_back_config'
METRIC_EVAL = 'metric_eval'


class AccuracyBasedAutoCalibrationBase: # pylint: disable=R0902
    """The superclass of automatic calibration based on accuracy.

    This superclass contains the main control of automatic calibration
    and a list of functions that subclasses must inherit and implement
    accroding to framework.

    Args:
        record_file (str): a string of record file path.
        config_file (str): a string of config file path.
        save_dir (str): a string of quantized model file path.
        evaluator (AutoCalibrationEvaluatorBase): A instance inherited
        from 'AutoCalibrationEvaluatorBase' and all the methods are
        implemented.
        strategy (AutoCalibrationStrategyBase): A instance inherited
        from 'AutoCalibrationStrategyBase' and all the methods are
        implemented.
        sensitivity (SensitivityBase): A instance inherited
        from 'SensitivityBase' and all the methods are implemented.
    """
    def __init__(self, # pylint: disable=R0913
                 record_file: str,
                 config_file: str,
                 save_dir: str,
                 evaluator: AutoCalibrationEvaluatorBase,
                 strategy: AutoCalibrationStrategyBase,
                 sensitivity: SensitivityBase):
        self.record_file = os.path.realpath(record_file)

        self.config_file = os.path.realpath(config_file)
        files.check_file_path(os.path.split(self.config_file)[0], os.path.split(self.config_file)[1])

        self.save_dir = os.path.realpath(save_dir)
        files.create_file_path(self.save_dir)

        self.amct_log_dir = os.path.join(os.getcwd(), LOG_FILE_DIR)
        time_stamp = datetime.now(tz=timezone(offset=timedelta(hours=8))).strftime('%Y%m%d%H%M%S%f')
        self.temp_dir = os.path.join(os.path.split(self.save_dir)[0], 'temp{}'.format(time_stamp))
        files.create_path(self.temp_dir)

        self.evaluator = evaluator
        self.strategy = strategy
        self.sensitivity = sensitivity

        self.history_records = []
        self.history_configs = None
        self.final_config = {}
        self.ranking_info = {}
        self.original_accuracy = None
        self.global_quant_accuracy = None
        self.saved_model_accuracy = None

    @staticmethod
    def get_quant_enable_layers(quant_config):
        """get all quantized layers"""
        quant_enable_layers = []
        for key, value in quant_config.items():
            if isinstance(value, dict) and value['quant_enable']:
                quant_enable_layers.append(key)
        return quant_enable_layers

    @staticmethod
    def is_all_roll_backed(roll_back_config):
        """check whether all layers are roll backed"""
        for _, value in roll_back_config.items():
            if value:
                return False
        return True

    @staticmethod
    def show_record_info(record):
        """print search record on terminal"""
        LOGGER.logi("quant layers:")
        start = 0
        while start < len(record.get(ROLL_BACK_CONFIG).items()):
            end = min(start + MAX_PRINT_INFO_NUM, len(record.get(ROLL_BACK_CONFIG).items()))
            LOGGER.logi(list(record.get(ROLL_BACK_CONFIG).items())[start:end])
            start += MAX_PRINT_INFO_NUM
        LOGGER.logi("metric_eval: {}".format(record.get(METRIC_EVAL)))

    def get_original_accuracy(self):
        """Accroding to framework, get the original model'saccuracy with
        object 'evaluator'.

        Return:
            Any object satisfied parameters of class function
            'evaluator.metric_eval()'.
        """
        raise NotImplementedError

    def get_global_quant_accuracy(self):
        """Accroding to framework, get the global quantized model's
        accuracy with object 'evaluator'.

        Return:
            The model accuracy, any object satisfied parameters of class
            function 'evaluator.metric_eval()'.
        """
        raise NotImplementedError

    def get_ranking_info(self) -> list:
        """Accroding to framework, get cosine similarity between
        quantized model and original model for every layer.

        Return:
            A list object of cosine similarity.
        """
        raise NotImplementedError

    def roll_back_and_evaluate_model(self, roll_back_config: dict):
        """Accroding to 'roll_back_config', create the quantized config
        file, calibration on original model and get the current model's
        accuracy with object 'evaluator'.

        Return:
            The model accuracy, any object satisfied parameters of class
            function 'evaluator.metric_eval()'.
        """
        raise NotImplementedError

    def fine_search(self):
        """Find the best quantized config by 'strategy' instance."""
        strategy_result = self.strategy.update_quant_config(
            self.evaluator.metric_eval(self.original_accuracy, self.global_quant_accuracy))
        self.history_configs = copy.deepcopy(strategy_result[ROLL_BACK_CONFIG])

        fine_search_step = 0
        fine_search_end_flag = False
        while not fine_search_end_flag:
            fine_search_step += 1
            fine_search_end_flag = strategy_result['stop_flag']
            if fine_search_end_flag and (strategy_result[
                    ROLL_BACK_CONFIG] == self.history_configs):
                break
            if self.is_all_roll_backed(strategy_result[ROLL_BACK_CONFIG]):
                metric_eval = self.evaluator.metric_eval(
                    self.original_accuracy, self.original_accuracy)
                record = {
                    ROLL_BACK_CONFIG: copy.deepcopy(strategy_result[ROLL_BACK_CONFIG]),
                    METRIC_EVAL: metric_eval
                }
                for key in record.get(ROLL_BACK_CONFIG).keys():
                    record.get(ROLL_BACK_CONFIG)[key] = False
                self.final_config = copy.deepcopy(record.get(ROLL_BACK_CONFIG))
                self.history_records.append(record)
                return ALL_ROLL_BACK
            current_accuracy = self.roll_back_and_evaluate_model(strategy_result[ROLL_BACK_CONFIG])
            self.saved_model_accuracy = current_accuracy
            metric_eval = self.evaluator.metric_eval(self.original_accuracy, current_accuracy)
            record = {
                ROLL_BACK_CONFIG: copy.deepcopy(strategy_result[ROLL_BACK_CONFIG]),
                METRIC_EVAL: metric_eval
            }

            LOGGER.logi("{} fine search step {} {}".format('*' * 20, fine_search_step, '*' * 20))
            AccuracyBasedAutoCalibrationBase.show_record_info(record)
            self.history_records.append(record)
            self.final_config = copy.deepcopy(strategy_result[ROLL_BACK_CONFIG])
            if not fine_search_end_flag:
                self.history_configs = copy.deepcopy(strategy_result[ROLL_BACK_CONFIG])
                strategy_result = self.strategy.update_quant_config(metric_eval)

        return PARTIAL_ROLL_BACK

    def convert_to_serializable(self, obj):
        if isinstance(obj, np.float32):
            return float(obj)
        elif isinstance(obj, list):
            return [self.convert_to_serializable(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: self.convert_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, tuple):
            obj_left, obj_right = obj
            return (self.convert_to_serializable(obj_left), self.convert_to_serializable(obj_right))
        else:
            return obj

    def save_ranking_info(self):
        """Save the similarity of feature map for each layer and sort
        them after automatic calibration.
        """
        ranking_file = files.create_empty_file(
            os.path.join(os.path.split(self.save_dir)[0], 'accuracy_based_auto_calibration_ranking_information.json'),
            check_exist=True)
        sorted_ranking_info = sorted(self.ranking_info.items(), key=lambda d: d[1])
        sorted_ranking_info = self.convert_to_serializable(sorted_ranking_info)
        with open(ranking_file, 'w') as dump_file:
            json_object = json.dumps(
                sorted_ranking_info, sort_keys=False, indent=4, separators=(',', ':'), ensure_ascii=False)
            dump_file.write(json_object)

    def save_history_records(self):
        """Save the automatic calibration search history."""
        history_record_file = files.create_empty_file(
            os.path.join(self.amct_log_dir, 'accuracy_based_auto_calibration_record.json'),
            check_exist=True)
        with open(history_record_file, 'w') as dump_file:
            for record in self.history_records:
                if isinstance(record[METRIC_EVAL], (tuple, list)) and len(record[METRIC_EVAL]) == METRIC_RET_LEN:
                    record[METRIC_EVAL] = [record[METRIC_EVAL][0], float(record[METRIC_EVAL][1])]
                json_object = json.dumps(record, sort_keys=False, indent=4, separators=(',', ':'), ensure_ascii=False)
                dump_file.write(json_object)


    def save_final_config(self):
        """ save the accuracy based auto calibration final quant config"""
        final_config_file = files.create_empty_file(
            os.path.join(os.path.split(self.save_dir)[0],
                'accuracy_based_auto_calibration_final_config.json'),
                check_exist=True)

        def _detect_repeated_key_hook(json_object):
            '''a hook function for detect repeated key in config file.'''
            keys = [key for key, value in json_object]
            repeat_keys = find_repeated_items(keys)
            check_no_repeated(repeat_keys, self.config_file)
            result = {}
            for key, value in json_object:
                result[key] = value
            return result

        with open(self.config_file, 'r') as fid:
            quant_config = json.load(
                fid, object_pairs_hook=_detect_repeated_key_hook)
        # update the modified config to original config file
        for key, value in self.final_config.items():
            # copy quant config when node is anonymous
            if key not in quant_config:
                LOGGER.logi("Node {} is an anonymous node, its quant config is copied on its type")
                anonymous_node_name = self.original_graph.get_node_by_name(key).type + '::common'
                quant_config[key] = quant_config.get(anonymous_node_name)
            quant_config[key]['quant_enable'] = value
        with open(final_config_file, 'w') as dump_file:
            json_object = json.dumps(
                quant_config, sort_keys=False, indent=4,
                separators=(',', ':'), ensure_ascii=False)
            dump_file.write(json_object)


    def clear(self):
        """Delete the generated model files when all layers are rolled
        back.
        """
        for item in os.listdir(os.path.split(self.save_dir)[0]):
            if 'deploy' in item or 'fake_quant' in item:
                os.remove(os.path.join(os.path.split(self.save_dir)[0], item))
            if item.endswith('quantized.pb'):
                os.remove(os.path.join(os.path.split(self.save_dir)[0], item))

    def run(self):
        """The main control of automatic calibration."""
        self.original_accuracy = self.get_original_accuracy()
        is_satisfied, _ = self.evaluator.metric_eval(self.original_accuracy, self.original_accuracy)
        if not is_satisfied:
            LOGGER.loge(
                "Compare between original_accuracy and original_accuracy can not satisfy the acc requirement, please "
                "check the metric_eval() function.")
            raise ValueError(
                "Compare between original_accuracy and original_accuracy can not satisfy the acc requirement, please "
                "check the metric_eval() function.")

        self.global_quant_accuracy = self.get_global_quant_accuracy()
        self.ranking_info = self.get_ranking_info()
        is_global_satisfied, _ = self.evaluator.metric_eval(self.original_accuracy, self.global_quant_accuracy)
        if is_global_satisfied:
            self.saved_model_accuracy = self.global_quant_accuracy
            LOGGER.logi(
                "The model satisfy the requirement after all layers are quantized, automatic calibration succeed!")
        elif self.ranking_info:
            self.strategy.initialize(self.ranking_info)
            roll_back_status = self.fine_search()
            LOGGER.logi("Fine search finish, automatic calibration succeed!")
            LOGGER.logi("accuracy based auto calibration search record:")
            for record in self.history_records:
                AccuracyBasedAutoCalibrationBase.show_record_info(record)
        else:
            LOGGER.logi('No layer can be rolled back in your model . Accuracy based auto calibration search stopped.')
            roll_back_status = ALL_ROLL_BACK

        self.save_ranking_info()
        self.save_history_records()
        LOGGER.logi("Accuracy of original model is {}".format(self.original_accuracy))
        LOGGER.logi("Accuracy of global quantized model is {}".format(self.global_quant_accuracy))
        if is_global_satisfied or roll_back_status == PARTIAL_ROLL_BACK:
            self.save_final_config()
            LOGGER.logi("Accuracy of saved model is {}".format(self.saved_model_accuracy))
            LOGGER.logi("The generated model is stored in dir: {}".format(os.path.split(self.save_dir)[0]))
            LOGGER.logi("The records file is stored in dir: {}".format(os.path.split(self.record_file)[0]))
        else:
            self.clear()
            LOGGER.logi(
                "No quantized model are generated due to all quant layers are roll backed, "
                "the accuracy target may be too difficult to achieve.")
        shutil.rmtree(self.temp_dir)
