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
__all__ = ['accuracy_based_auto_calibration']

import os
import copy
import json
from collections import OrderedDict

import torch # pylint: disable=E0401
from ..amct_pytorch import quantize_tool
from ..amct_pytorch.quantize_tool import inner_quantize_model
from ..amct_pytorch.quantize_tool import generate_fakequant_module
from ..amct_pytorch.quantize_tool import inner_fuse_bn

from ..amct_pytorch.common.utils import files as files_util
from ..amct_pytorch.common.utils.struct_helper import DumpConfig
from ..amct_pytorch.common.auto_calibration import SensitivityBase
from ..amct_pytorch.common.auto_calibration import AutoCalibrationStrategyBase
from ..amct_pytorch.common.auto_calibration import AutoCalibrationEvaluatorBase
from ..amct_pytorch.common.auto_calibration import AccuracyBasedAutoCalibrationBase
from ..amct_pytorch.common.auto_calibration import BinarySearchStrategy
from ..amct_pytorch.common.auto_calibration import CosineSimilaritySensitivity
from ..amct_pytorch.configuration.configuration import Configuration
from ..amct_pytorch.common.utils.check_params import check_params
from ..amct_pytorch.utils.auto_calibration_helper import AutoCalibrationHelper
from ..amct_pytorch.utils.model_util import ModuleHelper
from ..amct_pytorch.utils.log import LOGGER


AUTO_CALI_SATISFY = 0
AUTO_CALI_KEEP_SEARCH = 1


class AccuracyBasedAutoCalibration(AccuracyBasedAutoCalibrationBase): # pylint: disable=R0902
    """ the class for accuracy_based_auto_calibration API"""
    def __init__(self, # pylint: disable=R0913
                 model,
                 model_evaluator,
                 config_file,
                 record_file,
                 save_dir,
                 strategy,
                 sensitivity,
                 input_data,
                 input_names=None,
                 output_names=None,
                 dynamic_axes=None):
        """ init function for class AccuracyBasedAutoCalibration
        Parameters:
                model: the pytorch model instance
                model_evaluator: the user implemented evaluator instance
                config_file: the quant config json file
                record_file: the scale and offset record file dir
                save_dir: prefix of filename of save model
        """
        super().__init__(
            record_file,
            config_file,
            save_dir,
            model_evaluator,
            strategy,
            sensitivity)
        self.original_model = model
        self.input_data = input_data
        self.input_names = input_names
        self.output_names = output_names
        self.dynamic_axes = dynamic_axes
        self.fake_quant_module = None
        self.quant_layers = []
        self.original_quant_config = None
        self.ranking_info = None
        self.auto_calibration_helper = None
        self.final_config = None
        self.modified_onnx_file = os.path.join(
            self.temp_dir, "modified_model.onnx")

    @staticmethod
    def get_quant_enable_layers(quant_config):
        """ find the quant enable layers from quant config"""
        quant_enable_layers = []
        for key, value in quant_config.items():
            if isinstance(value, dict) and value['quant_enable']:
                quant_enable_layers.append(key)
        return quant_enable_layers

    def get_original_accuracy(self):
        """ get orginal_metrics and test eval model """
        # save the original model to pth file
        self.original_accuracy = self.evaluator.evaluate(self.original_model)
        LOGGER.logi("original evaluation accurcay: {}".format(
            self.original_accuracy), 'auto_calibration')
        return self.original_accuracy

    def get_global_quant_accuracy(self):
        """ do the calibration and get accuracy of fake quant model """
        global_fake_quant_accuracy = self.global_calibration()
        quant_config_torch = Configuration().get_quant_config()
        self.original_quant_config = quant_config_torch
        # record the layers that need quantization
        self.quant_layers = \
            AccuracyBasedAutoCalibration.get_quant_enable_layers(quant_config_torch)
        roll_back_config = OrderedDict()
        for layer in self.quant_layers:
            roll_back_config[layer] = True
        record = {
            'roll_back_config': roll_back_config,
            'metric_eval': self.evaluator.metric_eval(
                self.original_accuracy, global_fake_quant_accuracy)
        }
        self.final_config = roll_back_config
        self.history_records.append(record)
        return global_fake_quant_accuracy


    def global_calibration(self):
        """ do the calibration without joint quantization"""
        # do the calibration and save fake quant deploy model
        model = self.get_original_model()

        # generate a quantize model
        _ = files_util.create_empty_file(self.record_file)
        dump_config = DumpConfig(dump_dir=self.temp_dir, batch_num=None)
        modified_module = inner_quantize_model(
            config_file=self.config_file,
            modfied_onnx_file=self.modified_onnx_file,
            record_file=self.record_file,
            model=model,
            input_data=self.input_data,
            input_names=self.input_names,
            output_names=self.output_names,
            dynamic_axes=self.dynamic_axes,
            dump_config=dump_config,
            weight_fakequant=True)
        # activation calibration process
        self.evaluator.calibration(modified_module)
        quantize_tool.save_model(
            self.modified_onnx_file, self.record_file, self.save_dir)

        # evaluate the generated fake quant model
        self.fake_quant_module = generate_fakequant_module(model, self.config_file, self.record_file,
            self.input_data, self.input_names, self.output_names, self.dynamic_axes)
        # generate the fake quant model in  torch.nn.Module
        current_accuracy = self.evaluator.evaluate(
            self.fake_quant_module)
        LOGGER.logi(
            "global calibration accuracy: {}".format(
                current_accuracy), 'auto_calibration')

        return current_accuracy

    def generate_tmp_file_path(self, file_name):
        """ generate the file path for modified model, fused model"""
        file_path = os.path.join(self.temp_dir, file_name)
        return file_path

    def get_ranking_info(self):
        """ get the ranking information of accuracy based auto calibration"""
        model = self.get_original_model()
        # generate the fused pytorch module for single layer and
        # single fake quant layer cosine similarity compare
        fused_module = inner_fuse_bn(
            model=model,
            config_file=self.config_file,
            record_file=self.record_file,
            input_data=self.input_data,
            input_names=self.input_names,
            output_names=self.output_names,
            dynamic_axes=self.dynamic_axes)

        auto_calibration_helper = AutoCalibrationHelper(
            fused_module,
            self.fake_quant_module,
            self.quant_layers,
            self.record_file,
            self.temp_dir,
            self.sensitivity)
        ranking_info, _ = auto_calibration_helper.calc_ranking_info()
        self.ranking_info = ranking_info
        return ranking_info


    def roll_back_and_evaluate_model(self, roll_back_config):
        """ generate the roll-back fake quant model and
            evaluate the fake quant model
        """
        model = self.get_original_model()
        quant_config = copy.deepcopy(self.original_quant_config)
        for key, value in roll_back_config.items():
            if not value:
                del quant_config[key]

        with open(self.config_file, 'r') as config_file:
            config_dict = json.load(config_file)
            for key, value in roll_back_config.items():
                config_dict[key]['quant_enable'] = value

        with open(self.config_file, 'w') as config_file:
            config_file.write(json.dumps(
                config_dict, sort_keys=False, indent=4,
                separators=(',', ':'), ensure_ascii=False))

        modified_module = inner_quantize_model(
            config_file=self.config_file,
            modfied_onnx_file=self.modified_onnx_file,
            record_file=self.record_file,
            model=model,
            input_data=self.input_data,
            input_names=self.input_names,
            output_names=self.output_names,
            dynamic_axes=self.dynamic_axes,
            dump_config=None,
            weight_fakequant=True)
        # activation calibration process
        self.evaluator.calibration(modified_module)

        quantize_tool.save_model(
            self.modified_onnx_file, self.record_file, self.save_dir)

        self.fake_quant_module = generate_fakequant_module(model, self.config_file, self.record_file,
            self.input_data, self.input_names, self.output_names, self.dynamic_axes)

        current_fq_metric = self.evaluator.evaluate(
            self.fake_quant_module)
        return current_fq_metric


    def get_original_model(self):
        """ get the original model"""
        try:
            model = ModuleHelper.deep_copy(self.original_model)
        except RuntimeError as exception:
            LOGGER.logw(
                exception, "accuracy_based_auto_calibration deep_copy model")
        return model


@check_params(model=torch.nn.Module,
              config_file=str,
              record_file=str,
              save_dir=str,
              input_names=(list, type(None)),
              output_names=(list, type(None)),
              dynamic_axes=(dict, type(None)))
def accuracy_based_auto_calibration( # pylint: disable=too-many-arguments
        model,
        model_evaluator,
        config_file,
        record_file,
        save_dir,
        input_data,
        input_names=None,
        output_names=None,
        dynamic_axes=None,
        strategy='BinarySearch',
        sensitivity='CosineSimilarity'):
    """
    Function:
            calibration the input model automatically, decide which layers need
            to roll back and save the final quantized model
            (fake quant and deploy models)
    Parameters:
            model: the pytorch model instance
            model_evaluator: the user implemented evaluator instance
            config_file: the quant config json file
            record_file: the scale and offset record file path
            save_dir: prefix of file path and filename of save model
            input_data: used to compile model, can be ramdom data
            input_names: list of strings, names to assign to the input nodes
                of the graph, need to be in order
            output_names: names to assign to the output nodes of the graph,
                in order dynamic_axes: a dictionary to specify dynamic axes
                of input/output
            dynamic_axes: a dictionary to specify dynamic axes of input/output
            strategy: union [str, BinarySearchStrategy] the instance
                of strategy to control the search process,
                set 'BinarySearch' to use default value
            sensitivity: union [str, CosineSimilaritySensitivity] the instance
                of sensitivity to measure the quant sensitivity of quantable
                layers, set 'CosineSimilarity' to use default value
    """

    if strategy == 'BinarySearch':
        strategy = BinarySearchStrategy()
    else:
        if not isinstance(strategy, AutoCalibrationStrategyBase):
            raise RuntimeError(
                "strategy is not inherited from base class"
                " AutoCalibrationStrategyBase")

    if sensitivity == 'CosineSimilarity':
        sensitivity = CosineSimilaritySensitivity()
    else:
        if not isinstance(sensitivity, SensitivityBase):
            raise RuntimeError(
                "sensitivity is not inherited from base class SensitivityBase")

    files_util.check_file_path(config_file, 'config_file')
    if not isinstance(model_evaluator, AutoCalibrationEvaluatorBase):
        raise RuntimeError(
            "the model evaluator is not inherited from base class"
            " AutoCalibrationEvaluatorBase")

    auto_calibration_controller = \
        AccuracyBasedAutoCalibration(model=model,
                                     model_evaluator=model_evaluator,
                                     config_file=config_file,
                                     record_file=record_file,
                                     save_dir=save_dir,
                                     strategy=strategy,
                                     sensitivity=sensitivity,
                                     input_data=input_data,
                                     input_names=input_names,
                                     output_names=output_names,
                                     dynamic_axes=dynamic_axes)
    auto_calibration_controller.run()
